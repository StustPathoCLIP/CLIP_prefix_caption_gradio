"""
train_end_by_end_amp_multi_res.py ─ End-to-End Fine-Tuning (PathCLIP + GPT-2 + AMP + Multi-Res)
====================================================================================

‣ 與舊 JSON / 路徑映射 100% 相容  
‣ 預設仍是「單一路徑」訓練；加上 --multi_res 即自動開啟「多解析度 CLIP」  
‣ 多解析度：同一張切片輸出 (224×224) 與 (448×448) 兩種視野  
‣ 兩路徑共用同一個 PathCLIP Encoder → 取平均融合 → Prefix → GPT-2  
‣ 其餘功能（AMP、梯度累積、早停、tqdm、Windows pickle 修正）保持原樣
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
import cv2
from visualization import (
    grad_to_heatmap, overlay_heatmap
)

# ─────────────── 自訂工具 & 架構 ────────────────
from pathclip_loader import load_pathclip
from train_script import ClipCaptionModel

# ======================================================================
#  1. Dataset（新增多解析度支援）
# ======================================================================
class TCGAImageCaptionDS(Dataset):
    """
    若 multi_res=False  → 傳回 (img_low, )；與舊版行為一致
    若 multi_res=True   → 傳回 (img_low, img_high)
    """
    def __init__(self, ann_json, image_root, preprocess, tokenizer,
                 max_len=128, multi_res=False, high_size=448):
        self.data = json.load(open(ann_json, encoding="utf-8"))
        self.image_root   = image_root
        self.preprocess   = preprocess               # 224×224 路徑
        self.multi_res    = multi_res
        self.tokenizer    = tokenizer
        self.max_len      = max_len
        # 另一條高解析度 transform：先較大 resize，再中心裁成 448→再轉 224
        if multi_res:
            self.preprocess_high = T.Compose([
                T.Resize(high_size),                 # 先放大
                T.CenterCrop(high_size),
                T.Resize(preprocess.transforms[0].size),  # 再縮回 224，保留細節
                *preprocess.transforms[1:],          # 後續 Normalize 與 ToTensor
            ])

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        folder = rec["image_id"].rsplit("-", 1)[0]
        img_path = os.path.join(self.image_root, folder, f'{rec["image_id"]}.jpg')
        pil = _fast_read_pil(img_path)
        img_low = self.preprocess(pil)
        if self.multi_res:
            img_high = self.preprocess_high(pil)
            img_pair = (img_low, img_high)
        else:
            img_pair = (img_low,)                   # tuple 方便後續統一處理

        tok = self.tokenizer.encode(rec["caption"], max_length=self.max_len,
                                    truncation=True)
        return img_pair, torch.tensor(tok, dtype=torch.long)


def _fast_read_pil(path):
    img = tvio.read_image(path).float() / 255.0
    return T.ToPILImage()(img)


def collate_pad(batch, pad_id, multi_res=False):
    """batch: ((img_low, [img_high]), tok)"""
    imgs_low, imgs_high, toks = [], [], []
    for img_pair, tok in batch:
        imgs_low.append(img_pair[0])
        if multi_res:
            imgs_high.append(img_pair[1])
        toks.append(tok)

    imgs_low = torch.stack(imgs_low)
    if multi_res:
        imgs_high = torch.stack(imgs_high)

    maxlen = max(len(t) for t in toks)
    padded = torch.full((len(toks), maxlen), pad_id, dtype=torch.long)
    for i, t in enumerate(toks):
        padded[i, :len(t)] = t

    if multi_res:
        return imgs_low, imgs_high, padded
    else:
        return imgs_low, padded

# ======================================================================
#  2. 訓練主流程 (含「可選」AMP + Multi-Res)
# ======================================================================
def train(args):
    # ── 裝置 & AMP ─────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp          # 由 CLI 開關決定是否啟用 AMP
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 1) PathCLIP encoder ────────────────────
    clip_model, preprocess = load_pathclip("pt_model/pathclip-base.pt", device)
    for p in clip_model.parameters(): p.requires_grad = True

    # ── 2) Tokenizer / DataLoader ──────────────
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    ds = TCGAImageCaptionDS(
        args.ann_json, args.image_root, preprocess, tokenizer,
        multi_res=args.multi_res, high_size=args.high_size
    )
    collate_fn = partial(collate_pad, pad_id=tokenizer.pad_token_id,
                         multi_res=args.multi_res)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4,
                        pin_memory=True, collate_fn=collate_fn)

    # ── 3) Caption Prefix 模型 ──────────────────
    cap_model = ClipCaptionModel(prefix_length=args.prefix_len).to(device)

    # ── 4) Optimizer + Scheduler ───────────────
    optimizer = torch.optim.AdamW(
        [{"params": clip_model.parameters(), "lr": args.lr_encoder},
         {"params": cap_model.parameters(),   "lr": args.lr_prefix}],
        weight_decay=1e-2)
    total_steps = math.ceil(len(loader)/args.grad_accum)*args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps*0.1), total_steps)

    # ── 5) Training Loop ───────────────────────
    best_loss, no_imp, g_step = float("inf"), 0, 0
    clip_model.train(); cap_model.train()

    for ep in range(args.epochs):
        epoch_loss, t0 = 0.0, time.time()
        pbar = tqdm(loader, total=len(loader),
                    desc=f"Epoch {ep+1}/{args.epochs}", unit="batch")

        for batch in pbar:
            if args.multi_res:
                imgs_low, imgs_high, toks = batch
                imgs_high = imgs_high.to(device)
            else:
                imgs_low, toks = batch
            imgs_low, toks = imgs_low.to(device), toks.to(device)

            # ---------- Forward (AMP 可選) ----------
            with torch.cuda.amp.autocast(enabled=use_amp):
                feats_low  = clip_model.encode_image(imgs_low).float()   # [B,512]
                if args.multi_res:
                    feats_high = clip_model.encode_image(imgs_high).float()
                    feats = (feats_low + feats_high) / 2                # 加權平均融合
                else:
                    feats = feats_low

                prefix = cap_model.clip_project(feats).view(
                    feats.size(0), args.prefix_len, -1)
                tok_emb = cap_model.gpt.transformer.wte(toks)
                inputs  = torch.cat([prefix, tok_emb], dim=1)

                labels = torch.cat(
                    [torch.full((toks.size(0), args.prefix_len), -100,
                                dtype=torch.long, device=device), toks], dim=1)
                loss = cap_model.gpt(inputs_embeds=inputs,
                                     labels=labels).loss / args.grad_accum

            # ---------- Backward ----------
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if (g_step+1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(cap_model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            g_step += 1

            pbar.set_postfix(loss=f"{epoch_loss / (pbar.n+1):.4f}")

        # ---------- Epoch End ----------
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {ep+1} 完成｜loss={avg_loss:.4f}｜耗時 {time.time()-t0:.1f}s")

        # ========== [新增] 可視化模型注意力 ==========
        if args.visualize_every and (ep + 1) % args.visualize_every == 0:
            clip_model.eval(); cap_model.eval()              # 先關掉 dropout
            # 取目前 batch 的第一張圖；只示範 single-res 版本
            sample_img = imgs_low[0:1].to(device)            # shape [1,3,224,224]
            sample_img.requires_grad_(True)

            # ❶ 停用 AMP，確保梯度正確
            with torch.cuda.amp.autocast(enabled=False):
                feat  = clip_model.encode_image(sample_img).float()
                prefix = cap_model.clip_project(feat).view(1, args.prefix_len, -1)
                logits = cap_model.gpt(inputs_embeds=prefix).logits

            scalar = logits[0, -1].log_softmax(-1).max()     # 單一 scalar score
            scalar.backward()                                # ❷ 反傳梯度

            # ❸ 梯度 → 熱力圖 → 疊圖
            heat_np  = grad_to_heatmap(sample_img.grad.squeeze(), (224, 224))
            # 把張量轉回 PIL（此處忽略反 Normalization，若顏色不對可自行還原）
            pil_img  = T.ToPILImage()(sample_img.detach().cpu().squeeze())
            overlay  = overlay_heatmap(pil_img, heat_np)

            # ❹ 寫檔
            vis_name = f"vis_ep{ep+1}.png"
            vis_path = os.path.join(args.out_dir, vis_name)
            # 1) 顯示你要寫到哪裡
            abs_path = os.path.abspath(vis_path)
            print(f"→ 嘗試寫入熱力圖到：{abs_path}")
            # 2) 真正寫入，並檢查回傳值
            success = cv2.imwrite(
                abs_path,
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )
            print(f"   cv2.imwrite 成功嗎？{success}")

            clip_model.train(); cap_model.train()            # 切回訓練模式
        # ============================================


        # save ckpt
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(
            {"caption_state_dict": cap_model.state_dict(),
             "clip_state_dict":    clip_model.state_dict(),
             "optimizer_state":    optimizer.state_dict(),
             "scaler_state":       scaler.state_dict(),
             "epoch":              ep+1, "loss": avg_loss},
            os.path.join(args.out_dir, f"end2end_ep{ep+1}.pt"))

        if avg_loss+1e-5 < best_loss:
            best_loss, no_imp = avg_loss, 0
        else:
            no_imp += 1
            if no_imp >= 3:
                print("Early-Stopping：loss 無提升，結束訓練。")
                break

# ======================================================================
#  3. CLI 參數（新增 --amp）
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser("End-to-End PathCLIP + AMP (+Multi-Res)")
    p.add_argument("--ann_json",   required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--out_dir",    default="checkpoints_end2end")
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--bs",         type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--prefix_len", type=int, default=10)
    p.add_argument("--lr_encoder", type=float, default=1e-6)
    p.add_argument("--lr_prefix",  type=float, default=1e-4)
    # ★ 新增：AMP 開關 & 多解析度開關 ★
    p.add_argument("--amp", action="store_true",
                   help="啟用 PyTorch Automatic Mixed Precision")
    p.add_argument("--multi_res",  action="store_true",
                   help="啟用多解析度雙路徑訓練")
    p.add_argument("--high_size",  type=int, default=448,
                   help="高解析度路徑先裁到多少邊長再縮回 224")
    p.add_argument("--visualize_every", type=int, default=0,
               help=">0 時，每 N 個 epoch 產生 1 張熱力圖並存檔")
    return p.parse_args()

# ======================================================================
if __name__ == "__main__":
    args = parse_args()
    random.seed(42); torch.manual_seed(42)
    train(args)
