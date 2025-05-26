# ──────────────────── 標準函式庫 & 第三方 ────────────────────
import os, json, math, random, time, argparse
import torch
import torchvision.io as tvio
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from functools import partial              # 為 DataLoader 的 collate_fn 做包裝
from tqdm import tqdm                       # 終端進度條

# ──────────────────── 自訂工具 & 架構 ────────────────────
from pathclip_loader import load_pathclip   # 載入 PathCLIP 權重 + 影像前處理
from train_script import ClipCaptionModel   # Prefix 投影層 + GPT-2 架構


# ======================================================================
#  1. Dataset：讀舊格式 JSON，動態映射圖片路徑
# ======================================================================
class TCGAImageCaptionDS(Dataset):
    """
    預期 JSON 範例：
    [
      { "image_id": "TCGA-3L-AA1B-1", "caption": "..." },
      ...
    ]

    圖片對應規則：
      • folder  = image_id 去掉最後一段 ('-1')   →  TCGA-3L-AA1B
      • 檔名    = f"{image_id}.jpg"
      • 路徑    = image_root / folder / 檔名
    """

    def __init__(self,
                 ann_json: str,
                 image_root: str,
                 preprocess,               # PathCLIP 的 transform
                 tokenizer: GPT2Tokenizer,
                 max_len: int = 128):
        # 讀取 JSON 到 self.data
        with open(ann_json, encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        # ── 影像 ─────────────────────────────
        img_path = self._path_from_id(rec["image_id"])
        img_tensor = self.preprocess(_fast_read_pil(img_path))  # [3,H,W]

        # ── Caption → token IDs ─────────────
        tokens = self.tokenizer.encode(
            rec["caption"], max_length=self.max_len,
            truncation=True
        )
        return img_tensor, torch.tensor(tokens, dtype=torch.long)

    # 根據 image_id 拼出圖片路徑
    def _path_from_id(self, image_id: str) -> str:
        folder = image_id.rsplit("-", 1)[0]  # 例：TCGA-3L-AA1B-1 → TCGA-3L-AA1B
        return os.path.join(self.image_root, folder, f"{image_id}.jpg")


# —— 影像讀取：使用 torchvision.io 速度較快 ——
def _fast_read_pil(path: str):
    img_uint8 = tvio.read_image(path)       # uint8 [C,H,W]
    return T.ToPILImage()(img_uint8.float() / 255.0)


# —— 自訂 collate_fn：將不同長度 token padding 成同長 ——
def collate_pad(batch, pad_id: int):
    imgs, toks = zip(*batch)
    imgs = torch.stack(imgs)                # [B,3,H,W]

    maxlen = max(len(t) for t in toks)
    padded = torch.full((len(toks), maxlen),
                        pad_id, dtype=torch.long)
    for i, t in enumerate(toks):
        padded[i, : len(t)] = t
    return imgs, padded


# ======================================================================
#  2. 訓練主流程
# ======================================================================
def train(args):
    # ── 裝置設定 ─────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1) 載入 PathCLIP encoder & preprocess ─
    clip_model, preprocess = load_pathclip("pt_model/pathclip-base.pt", device)
    # 端到端：整個 encoder 解凍，但 learning-rate 很小
    for p in clip_model.parameters():
        p.requires_grad = True

    # ── 2) Tokenizer, Dataset, DataLoader ─
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # 讓 PAD = EOS
    dataset = TCGAImageCaptionDS(
        args.ann_json, args.image_root, preprocess, tokenizer
    )
    # 用 partial 包裝，避免 lambda 造成 Windows pickle 失敗
    collate_fn = partial(collate_pad, pad_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── 3) 建立 Caption Prefix 模型 ────────────
    cap_model = ClipCaptionModel(prefix_length=args.prefix_len).to(device)

    # ── 4) Optimizer + Scheduler ───────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": clip_model.parameters(), "lr": args.lr_encoder},
            {"params": cap_model.parameters(), "lr": args.lr_prefix},
        ],
        weight_decay=1e-2,
    )
    total_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps
    )

    # ── 5) 進入訓練迴圈 ─────────────────────────
    best_loss, no_imp, global_step = float("inf"), 0, 0
    clip_model.train()
    cap_model.train()

    for ep in range(args.epochs):
        epoch_loss, t0 = 0.0, time.time()

        # tqdm 進度條：顯示 batch 進度 & 即時平均 loss
        pbar = tqdm(loader, total=len(loader),
                    desc=f"Epoch {ep+1}/{args.epochs}", unit="batch")

        for imgs, toks in pbar:
            imgs, toks = imgs.to(device), toks.to(device)

            # ── forward : 影像 → prefix → GPT-2 ──
            feats = clip_model.encode_image(imgs).float()          # [B,512]
            prefix = cap_model.clip_project(feats).view(
                imgs.size(0), args.prefix_len, -1
            )                                                      # [B,L,H]
            tok_emb = cap_model.gpt.transformer.wte(toks)          # [B,T,H]
            inputs_embeds = torch.cat([prefix, tok_emb], dim=1)    # [B,L+T,H]

            # GPT-2 labels：prefix 部分設 -100 忽略 loss
            labels = torch.cat(
                [
                    torch.full(
                        (toks.size(0), args.prefix_len),
                        -100, dtype=torch.long, device=device
                    ),
                    toks
                ],
                dim=1
            )
            loss = cap_model.gpt(inputs_embeds=inputs_embeds,
                                 labels=labels).loss

            # ── backward & 梯度累積 ───────────────
            (loss / args.grad_accum).backward()
            epoch_loss += loss.item()

            if (global_step + 1) % args.grad_accum == 0:
                # 梯度裁剪避免梯度爆炸
                torch.nn.utils.clip_grad_norm_(cap_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # ── tqdm 動態顯示平均 loss ──────────────
            safe_div = epoch_loss / (pbar.n + 1)          # pbar.n 從 0 開始
            pbar.set_postfix(loss=f"{safe_div:.4f}")

        # ── Epoch 結束：列印結果 + checkpoint ────
        avg_loss = epoch_loss / len(loader)
        epoch_time = time.time() - t0
        print(f"Epoch {ep+1} 完成｜loss={avg_loss:.4f}｜耗時 {epoch_time:.1f}s")

        os.makedirs(args.out_dir, exist_ok=True)
        ckpt_path = os.path.join(args.out_dir, f"end2end_ep{ep+1}.pt")
        torch.save(
            {
                "caption_state_dict": cap_model.state_dict(),
                "clip_state_dict": clip_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": ep + 1,
                "loss": avg_loss,
            },
            ckpt_path,
        )

        # ── Early-Stopping：連續 3 Epoch 無提升即停──
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
    p = argparse.ArgumentParser("End-to-End PathCLIP 輕量微調")
    p.add_argument("--ann_json", required=True, help="舊格式 annotation JSON")
    p.add_argument("--image_root", required=True, help="切片圖根目錄 (train/)")
    p.add_argument("--out_dir", default="checkpoints_end2end", help="checkpoint 目錄")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=16, help="batch size")
    p.add_argument("--grad_accum", type=int, default=1, help="梯度累積步數")
    p.add_argument("--prefix_len", type=int, default=10)
    p.add_argument("--lr_encoder", type=float, default=1e-6)
    p.add_argument("--lr_prefix", type=float, default=1e-4)
    return p.parse_args()


# ======================================================================
#  4. 程式入口
# ======================================================================
if __name__ == "__main__":
    args = parse_args()
    # 固定隨機種子，確保可重現
    random.seed(42)
    torch.manual_seed(42)
    train(args)
