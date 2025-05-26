"""
train_end_by_end_amp.py  ─ End-to-End Fine-Tuning (PathCLIP + GPT-2 + AMP)
===========================================================

‣ 與舊 JSON / 路徑映射完全相容
‣ 啟用 PyTorch AMP：顯著減少 GPU 記憶體、提升訓練速度
‣ CPU 環境自動回退到 FP32
‣ 其它功能：梯度累積、早停、tqdm 進度條、Windows pickle 修正
"""

# ────────────── 標準函式庫 & 第三方 ──────────────
import os, json, math, random, time, argparse
import torch
import torchvision.io as tvio
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from functools import partial
from tqdm import tqdm

# ─────────────── 自訂工具 & 架構 ────────────────
from pathclip_loader import load_pathclip
from train_script import ClipCaptionModel


# ======================================================================
#  1. Dataset：舊 JSON → train/<folder>/<image_id>.jpg
# ======================================================================
class TCGAImageCaptionDS(Dataset):
    def __init__(self, ann_json, image_root, preprocess, tokenizer, max_len=128):
        self.data = json.load(open(ann_json, encoding="utf-8"))
        self.image_root = image_root
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        folder = rec["image_id"].rsplit("-", 1)[0]
        img_path = os.path.join(self.image_root, folder, f'{rec["image_id"]}.jpg')
        img = self.preprocess(_fast_read_pil(img_path))
        tok = self.tokenizer.encode(rec["caption"], max_length=self.max_len,
                                    truncation=True)
        return img, torch.tensor(tok, dtype=torch.long)


def _fast_read_pil(path):
    """以 torchvision 讀檔並轉 PIL，速度較 PIL.open() 快。"""
    img = tvio.read_image(path).float() / 255.0
    return T.ToPILImage()(img)


def collate_pad(batch, pad_id):
    """將 batch 中不同長度 caption padding 至同長。"""
    imgs, toks = zip(*batch)
    imgs = torch.stack(imgs)
    maxlen = max(len(t) for t in toks)
    padded = torch.full((len(toks), maxlen), pad_id, dtype=torch.long)
    for i, t in enumerate(toks):
        padded[i, : len(t)] = t
    return imgs, padded


# ======================================================================
#  2. 訓練主流程 (含 AMP)
# ======================================================================
def train(args):
    # ── 裝置與 AMP 設定 ─────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"                       # 只有 GPU 才啟用 AMP
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)   # 建立或停用 GradScaler

    # ── 1) PathCLIP encoder ────────────────────
    clip_model, preprocess = load_pathclip("pt_model/pathclip-base.pt", device)
    for p in clip_model.parameters(): p.requires_grad = True  # 解凍

    # ── 2) Tokenizer / DataLoader ──────────────
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    ds = TCGAImageCaptionDS(args.ann_json, args.image_root, preprocess, tokenizer)
    loader = DataLoader(
        ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True,
        collate_fn=partial(collate_pad, pad_id=tokenizer.pad_token_id)
    )

    # ── 3) Caption Prefix 模型 ──────────────────
    cap_model = ClipCaptionModel(prefix_length=args.prefix_len).to(device)

    # ── 4) Optimizer + Scheduler ───────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": clip_model.parameters(), "lr": args.lr_encoder},
            {"params": cap_model.parameters(),   "lr": args.lr_prefix},
        ], weight_decay=1e-2
    )
    total_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps
    )

    # ── 5) Training Loop ───────────────────────
    best_loss, no_imp, global_step = float("inf"), 0, 0
    clip_model.train(); cap_model.train()

    for ep in range(args.epochs):
        epoch_loss, t0 = 0.0, time.time()
        pbar = tqdm(loader, total=len(loader),
                    desc=f"Epoch {ep+1}/{args.epochs}", unit="batch")

        for imgs, toks in pbar:
            imgs, toks = imgs.to(device), toks.to(device)

            # ---------- Forward (AMP) ----------
            with torch.cuda.amp.autocast(enabled=use_amp):
                feats  = clip_model.encode_image(imgs).float()          # [B,512]
                prefix = cap_model.clip_project(feats).view(
                    imgs.size(0), args.prefix_len, -1)                 # [B,L,H]
                tok_emb = cap_model.gpt.transformer.wte(toks)           # [B,T,H]
                inputs = torch.cat([prefix, tok_emb], dim=1)            # [B,L+T,H]

                labels = torch.cat(
                    [torch.full((toks.size(0), args.prefix_len), -100,
                                dtype=torch.long, device=device), toks],
                    dim=1
                )
                loss = cap_model.gpt(inputs_embeds=inputs,
                                     labels=labels).loss / args.grad_accum

            # ---------- Backward (AMP) ----------
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if (global_step + 1) % args.grad_accum == 0:
                # unscale → clip grad → optimizer.step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(cap_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # tqdm 動態顯示平均 loss
            pbar.set_postfix(loss=f"{epoch_loss / (pbar.n + 1):.4f}")

        # ---------- Epoch 結束 ----------
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {ep+1} 完成｜loss={avg_loss:.4f}｜耗時 {time.time()-t0:.1f}s")

        # 儲存 checkpoint
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(
            {
                "caption_state_dict": cap_model.state_dict(),
                "clip_state_dict":    clip_model.state_dict(),
                "optimizer_state":    optimizer.state_dict(),
                "scaler_state":       scaler.state_dict(),
                "epoch":              ep + 1,
                "loss":               avg_loss,
            },
            os.path.join(args.out_dir, f"end2end_ep{ep+1}.pt")
        )

        # Early-Stopping：3 epoch 無提升即停
        if avg_loss + 1e-5 < best_loss:
            best_loss, no_imp = avg_loss, 0
        else:
            no_imp += 1
            if no_imp >= 3:
                print("Early-Stopping：loss 無提升，結束訓練。")
                break


# ======================================================================
#  3. CLI 參數
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser("End-to-End PathCLIP + AMP 微調")
    p.add_argument("--ann_json",   required=True, help="舊格式 JSON")
    p.add_argument("--image_root", required=True, help="train 影像根目錄")
    p.add_argument("--out_dir",    default="checkpoints_end2end")
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--bs",         type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--prefix_len", type=int, default=10)
    p.add_argument("--lr_encoder", type=float, default=1e-6)
    p.add_argument("--lr_prefix",  type=float, default=1e-4)
    return p.parse_args()


# ======================================================================
#  4. 程式入口
# ======================================================================
if __name__ == "__main__":
    args = parse_args()
    random.seed(42); torch.manual_seed(42)
    train(args)
