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


def main():
    parser = argparse.ArgumentParser(description="CLIP Prefix Caption Inference (matching original ipynb logic)")
    parser.add_argument('--image_path', required=True, type=str, help="Path to the input image")
    parser.add_argument('--model_path', required=True, type=str, help="Path to the pretrained model weights")
    parser.add_argument('--device', default="cuda", type=str, help="Device to use (cpu or cuda)")
    parser.add_argument('--prefix_length', default=10, type=int, help="Prefix length used in training")
    parser.add_argument('--use_beam_search', action='store_true', help="Whether to use beam search or nucleus sampling")
    args = parser.parse_args()

    # 選擇硬體裝置
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 載入 CLIP 模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # 載入 GPT-2 的 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 建立 ClipCaptionModel
    model = ClipCaptionModel(prefix_length=args.prefix_length)

    # 載入權重，手動刪除多餘的部分
    print("Loading model weights...")
    raw_state_dict = torch.load(args.model_path, map_location="cpu", weights_only=True)
    filtered_state_dict = filter_state_dict(raw_state_dict, model)  # 手動過濾多餘權重
    model.load_state_dict(filtered_state_dict)  # 不使用 strict=False
    print("Model weights loaded successfully!")
    
    # 設置模型為評估模式
    model.eval()
    model = model.to(device)

    # 讀取圖像
    pil_image = Image.open(args.image_path).convert("RGB")
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    # CLIP encode -> prefix
    with torch.no_grad():
        prefix = clip_model.encode_image(image_tensor).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, args.prefix_length, -1)

    # 生成文字
    if args.use_beam_search:
        # beam search
        generated_text = generate_beam(model, tokenizer, embed=prefix_embed, beam_size=5)[0]
    else:
        # nucleus sampling
        generated_text = generate2(model, tokenizer, embed=prefix_embed)

    # 輸出結果
    print("Generated caption:\n", generated_text)


if __name__ == "__main__":
    main()
