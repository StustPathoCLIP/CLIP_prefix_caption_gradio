import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from PIL import Image
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import trange
import clip
from visualization import grad_to_heatmap, overlay_heatmap
import cv2
import numpy as np

from pathclip_loader import load_pathclip

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]

def denorm_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * np.array(STD) + np.array(MEAN)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

#####################################
# 1. 定義模型結構
#####################################

class MLP(nn.Module):
    """
    與原始 ipynb 相同的多層感知器，用於 prefix 長度 <= 10 的情況
    """
    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ClipCaptionModel(nn.Module):
    """
    與原始 ipynb 相同的 ClipCaptionModel。
    - 當 prefix_length > 10，使用 nn.Linear
    - 當 prefix_length <= 10，使用 MLP
    """
    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        # 1) 載入 GPT-2
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        # 2) 根據 prefix_length 決定使用 Linear 或 MLP
        if prefix_length > 10:
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((
                prefix_size,
                (self.gpt_embedding_size * prefix_length) // 2,
                self.gpt_embedding_size * prefix_length
            ))

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor,
                mask: torch.Tensor = None, labels: torch.Tensor = None):
        # GPT-2 的 embedding
        embedding_text = self.gpt.transformer.wte(tokens)
        # 將 CLIP prefix 投影到 GPT-2 embedding 空間後 reshape
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        # 串接 prefix 與原本的 tokens embedding
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        # 有 labels 的情況下，要在前面加上 dummy_token，對齊 prefix
        if labels is not None:
            dummy_token = torch.zeros(tokens.shape[0], self.prefix_length,
                                      dtype=torch.int64, device=tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        # 使用 GPT-2 產生 logits
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


#####################################
# 2. 定義推論函式 (generate_beam, generate2)
#####################################

def generate_beam(
    model: ClipCaptionModel,
    tokenizer: GPT2Tokenizer,
    beam_size: int = 5,
    prompt: str = None,
    embed: torch.Tensor = None,
    entry_length: int = 67,
    temperature: float = 1.0,
    stop_token: str = '.'
):
    """
    與原始 ipynb 基本一致的 beam search 實作
    """
    model.eval()
    device = next(model.parameters()).device
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)

        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                # 將已經停止的 beam 設為 -inf
                logits[is_stopped] = -float("Inf")
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                # 算平均分數
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            # 拼接新 token 的 embedding
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            # 如果遇到 stop_token，就標記停止
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

def generate2(
    model: ClipCaptionModel,
    tokenizer: GPT2Tokenizer,
    tokens: torch.Tensor = None,
    prompt: str = None,
    embed: torch.Tensor = None,
    entry_count: int = 1,
    entry_length: int = 67,
    top_p: float = 0.8,
    temperature: float = 1.0,
    stop_token: str = '.'
):
    """
    與原始 ipynb 相同的 nucleus (top-p) 取樣方法
    """
    from tqdm import trange
    model.eval()
    device = next(model.parameters()).device
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")

    generated_list = []
    with torch.no_grad():
        for _ in trange(entry_count):
            generated = embed
            tokens = None
            for _ in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)

                # 過濾 > top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一個 token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)

                generated = torch.cat((generated, next_token_embed), dim=1)

                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

def filter_state_dict(state_dict, model):
    """
    手動過濾掉與模型結構不匹配的多餘權重
    """
    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    extra_keys = [k for k in state_dict.keys() if k not in model_keys]
    
    # 打印多餘的鍵名
    if extra_keys:
        #print("\nThe following keys are extra and will be removed:")
        for key in extra_keys:
            #print(f"  - {key}")
            pass
    
    return filtered_state_dict

def smart_load_checkpoint(model_path, prefix_length, device):
    """根據檔案內容，自動載回 (caption_model, clip_model, preprocess)。"""
    ckpt = torch.load(model_path, map_location="cpu")
    
    # 判斷格式：End-to-End (= 有 caption_state_dict)
    is_end2end = isinstance(ckpt, dict) and "caption_state_dict" in ckpt
    
    # 1) Caption model --------------------------------------------------
    caption_model = ClipCaptionModel(prefix_length=prefix_length)
    if is_end2end:
        caption_sd = ckpt["caption_state_dict"]
    else:
        caption_sd = ckpt
    caption_model.load_state_dict(filter_state_dict(caption_sd, caption_model), strict=False)
    caption_model.eval().to(device)
    
    # 2) CLIP / PathCLIP encoder ---------------------------------------
    if is_end2end and "clip_state_dict" in ckpt:
        clip_model, preprocess = load_pathclip()           # local base weights
        clip_model.load_state_dict(
            {k: v for k, v in ckpt["clip_state_dict"].items() if k in clip_model.state_dict()},
            strict=False
        )
    else:
        clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval().to(device)

    return caption_model, clip_model, preprocess


def main():
    """
    Inference entry-point  
    ‣ 自動相容 ① 傳統 prefix-only  ② PathCLIP End-to-End checkpoint  
    ‣ 支援 beam search / nucleus sampling  
    ‣ ★ 新增 --multi_res / --high_size：推理時同時用高解析度視野
    """
    import argparse, os
    parser = argparse.ArgumentParser("CLIP Prefix Caption Inference")
    parser.add_argument("--image_path",   required=True,  help="Path to input image")
    parser.add_argument("--model_path",   required=True,  help="Path to .pt / .pth checkpoint")
    parser.add_argument("--device",       default="cuda", help="cuda / cpu (auto-fallback)")
    parser.add_argument("--prefix_length",default=10,     type=int, help="Prefix length used in training")
    parser.add_argument("--use_beam_search",action="store_true", help="Use beam search instead of nucleus sampling")
    parser.add_argument("--beam_size",    default=5,      type=int, help="Beam size when --use_beam_search")
    parser.add_argument("--top_p",        default=0.8,    type=float, help="Top-p for nucleus sampling")
    parser.add_argument("--temperature",  default=1.0,    type=float, help="Sampling temperature")
    parser.add_argument("--stop_token",   default=".",    help="Generation stop token")
    # ── ★ 多解析度旗標 ───────────────────────────
    parser.add_argument("--multi_res",  action="store_true",
                        help="啟用多解析度 (低224 + 高解析度) 雙路推理")
    parser.add_argument("--high_size",  type=int, default=448,
                        help="高解析度路徑先裁成多少邊長再縮回 224 (需搭配 --multi_res)")
    parser.add_argument("--explain", action="store_true",
                    help="同時輸出梯度熱力圖 (saliency.png)")
    args = parser.parse_args()

    # ── 1. 裝置 ─────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 2. 讀取模型 (PathCLIP / Prefix-only 自動判斷) ─
    caption_model, clip_model, preprocess = smart_load_checkpoint(
        model_path     = args.model_path,
        prefix_length  = args.prefix_length,
        device         = device,
    )

    # ── 3. Tokenizer ───────────────────────────
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # ── 4. 影像前處理 → CLIP 特徵 (支援多解析度) ────
    from PIL import Image
    from torchvision import transforms as T          # 避免改動全域 import
    pil_image = Image.open(args.image_path).convert("RGB")

    # 低解析度 (224×224) 視野
    img_low = preprocess(pil_image).unsqueeze(0).to(device)

    # 若開啟 --multi_res，再做一條高解析度視野
    if args.multi_res:
        preprocess_high = T.Compose([
            T.Resize(args.high_size),
            T.CenterCrop(args.high_size),
            T.Resize(preprocess.transforms[0].size),   # 縮回 224
            *preprocess.transforms[1:],               # Normalize / ToTensor
        ])
        img_high = preprocess_high(pil_image).unsqueeze(0).to(device)

    # —— CLIP encode & 融合 ——
    with torch.no_grad():
        feats_low = clip_model.encode_image(img_low).to(dtype=torch.float32)
        if args.multi_res:
            feats_high = clip_model.encode_image(img_high).to(dtype=torch.float32)
            clip_feat = (feats_low + feats_high) / 2          # 簡單平均融合
        else:
            clip_feat = feats_low

        prefix_embed = caption_model.clip_project(clip_feat)\
                                    .view(1, args.prefix_length, -1)

    # ── 5. 產生 Caption ──────────────────────────
    if args.use_beam_search:
        caption = generate_beam(
            model      = caption_model,
            tokenizer  = tokenizer,
            embed      = prefix_embed,
            beam_size  = args.beam_size,
            temperature= args.temperature,
            stop_token = args.stop_token
        )[0]
    else:
        caption = generate2(
            model       = caption_model,
            tokenizer   = tokenizer,
            embed       = prefix_embed,
            entry_length= 67,
            top_p       = args.top_p,
            temperature = args.temperature,
            stop_token  = args.stop_token
        )

    # ── 6. 輸出結果 ──────────────────────────────
    print("\nGenerated caption:")
    print(caption)

    if args.explain:
        # 1) 前處理 + 允許梯度
        pil_img = Image.open(args.image_path).convert("RGB")
        img_t   = preprocess(pil_img).unsqueeze(0).to(device)
        img_t.requires_grad_(True)
        H, W = img_t.shape[-2:]                                  # 224, 224

        # 2) 前向 (保留計算圖)
        feat   = clip_model.encode_image(img_t).float()
        prefix = caption_model.clip_project(feat).view(1, args.prefix_length, -1)
        out    = caption_model.gpt(inputs_embeds=prefix)

        # 3) backward
        logprob = out.logits[0, -1].log_softmax(-1).max()
        logprob.backward()

        # 4) 熱力圖 (224×224)
        heat = grad_to_heatmap(img_t.grad.squeeze(), (H, W))
        orig = denorm_to_pil(img_t.squeeze())                    # 224×224
        vis  = overlay_heatmap(orig, heat)

        # 5) 存檔
        out_name = "saliency.png"
        cv2.imwrite(out_name, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"✔ 已輸出梯度熱力圖：{out_name}")

if __name__ == "__main__":
    main()
