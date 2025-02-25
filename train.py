import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union

# AMP 新 API
from torch.amp import autocast, GradScaler

# 使用 PyTorch 自帶的 AdamW 取代 HuggingFace 版本
from torch.optim import AdamW


############################################
#          基礎 Enums & Classes
############################################

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset(Dataset):
    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2", normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)

        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()

        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [c['caption'] for c in captions_raw]

        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for c in captions_raw:
                txt = c["caption"]
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(txt), dtype=torch.int64))
                self.caption2embedding.append(c["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)

        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))], dtype=torch.float)
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # 有效 token = 1
        tokens[~mask] = 0
        mask = mask.float()
        # prefix 部分 mask=1
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self*2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, _ = y.shape
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c//self.num_heads)
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c//self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]

        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self*mlp_ratio), act=act, dropout=dropout)

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x_ = self.attn(self.norm1(x), y, mask)[0]
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim_self: int, num_heads: int, num_layers: int,
                 dim_ref: Optional[int] = None, mlp_ratio: float = 2.,
                 act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super().__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            # 這裡省略 enc_dec 的分支邏輯
            layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for layer in self.layers:
            x = layer(x, y, mask)
        return x


class TransformerMapper(nn.Module):
    def __init__(self, dim_clip: int, dim_embedding: int,
                 prefix_length: int, clip_length: int,
                 num_layers: int = 8):
        super().__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out


class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, clip_length: Optional[int] = None,
                 prefix_size: int = 512, num_layers: int = 8,
                 mapping_type: MappingType = MappingType.MLP):
        super().__init__()
        self.prefix_length = prefix_length
        # 載入 GPT-2
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        if mapping_type == MappingType.MLP:
            self.clip_project = MLP(
                (prefix_size,
                 (self.gpt_embedding_size * prefix_length) // 2,
                 self.gpt_embedding_size * prefix_length)
            )
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size,
                                                  prefix_length, clip_length, num_layers)

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor,
                mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        # 只訓練 prefix 部分
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        # GPT-2 部分維持 eval (凍結參數)
        self.gpt.eval()
        return self


###################################
# ----- 修正版：train(...) -----
###################################
def train(dataset: ClipCocoDataset,
          model: ClipCaptionModel,
          args,
          lr: float = 2e-5,
          warmup_steps: int = 5000,
          output_dir: str = ".",
          output_prefix: str = ""):
    """
    一次性訓練版本。使用 PyTorch AMP + 修正 Scheduler 順序 + PyTorch AdamW。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.bs
    epochs = args.epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = model.to(device)
    model.train()

    # 使用 torch.optim.AdamW
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 透過 transformers 的 get_linear_schedule_with_warmup, 但要注意 step 順序
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=epochs * len(train_dataloader)
    )

    # AMP: GradScaler("cuda")
    scaler = GradScaler("cuda")

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()

        # 你若不想在終端機印出 tqdm，可移除
        from tqdm import tqdm
        progress = tqdm(total=len(train_dataloader), desc=output_prefix, leave=True)

        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            optimizer.zero_grad()

            tokens = tokens.to(device)
            mask = mask.to(device)
            prefix = prefix.to(device, dtype=torch.float32)

            # AMP
            with autocast("cuda"):
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, dataset.prefix_length - 1: -1]
                loss = nnf.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tokens.flatten(),
                    ignore_index=0
                )

            # backward with scaler
            scaler.scale(loss).backward()

            # 先 scaler.step(optimizer) → 再 scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            progress.set_postfix({"loss": loss.item()})
            progress.update()

            # 可定期存檔
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt")
                )

        progress.close()

        # 每個 epoch 結束的存檔
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt")
            )

    return model


###################################
# ----- train_generator(...) -----
###################################
def train_generator(dataset: ClipCocoDataset,
                    model: ClipCaptionModel,
                    args,
                    lr: float = 2e-5,
                    warmup_steps: int = 5000,
                    output_dir: str = ".",
                    output_prefix: str = ""):
    """
    與 train(...) 類似，但改為「產生器」：每個 batch / epoch 都 yield 一次進度。
    可在 gradio_app.py 中使用，於 Gradio 前端即時顯示。
    """
    import sys, os
    import torch
    import torch.nn.functional as nnf
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.bs
    epochs = args.epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=epochs * len(train_dataloader)
    )

    scaler = GradScaler("cuda")

    total_steps = epochs * len(train_dataloader)
    global_step = 0

    for epoch in range(epochs):
        # 在 epoch 開頭 yield
        yield (f"=== Epoch {epoch+1}/{epochs} ===\n", None, None, None, None)

        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            global_step += 1

            optimizer.zero_grad()

            tokens = tokens.to(device)
            mask = mask.to(device)
            prefix = prefix.to(device, dtype=torch.float32)

            with autocast("cuda"):
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, dataset.prefix_length - 1: -1]
                loss = nnf.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tokens.flatten(),
                    ignore_index=0
                )

            scaler.scale(loss).backward()

            # 先 step(optimizer) 再 scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 每個 batch 訓練完就 yield 一次
            yield (None, epoch, global_step, total_steps, loss.item())

            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )

        # 每 epoch 結束，存檔
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt")
            )

    # 結束
    yield (
        f"✔ 訓練完成，總 steps={global_step}。檔案已儲存至 {output_dir}\n",
        None, None, None, None
    )
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')

    args = parser.parse_args()
    prefix_length = args.prefix_length
    dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)

    prefix_dim = 640 if args.is_rn else 512
    # 將 "mlp"/"transformer" 字串轉為 enum
    from enum import Enum
    args.mapping_type = {
        'mlp': MappingType.MLP,
        'transformer': MappingType.Transformer
    }[args.mapping_type]

    if args.only_prefix:
        model = ClipCaptionPrefix(
            prefix_length,
            clip_length=args.prefix_length_clip,
            prefix_size=prefix_dim,
            num_layers=args.num_layers,
            mapping_type=args.mapping_type
        )
        print("Train only prefix")
    else:
        model = ClipCaptionModel(
            prefix_length,
            clip_length=args.prefix_length_clip,
            prefix_size=prefix_dim,
            num_layers=args.num_layers,
            mapping_type=args.mapping_type
        )
        print("Train both prefix and GPT")
        sys.stdout.flush()

    # 一次性訓練
    train(
        dataset=dataset,
        model=model,
        args=args,
        output_dir=args.out_dir,
        output_prefix=args.prefix
    )


if __name__ == '__main__':
    main()
