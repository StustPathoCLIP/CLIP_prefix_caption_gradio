# train_script.py

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class ClipCaptionModel(nn.Module):
    """
    ClipCaptionModel:
    • 將 PathCLIP 編碼後的影像特徵透過線性投影 (clip_project)
      轉換為 GPT-2 的 prefix embedding。
    • 再將 prefix embedding 與文字 token embedding 拼接，
      餵入 GPT-2 進行 caption 生成或微調。

    主要屬性:
      prefix_length:      prefix token 數量
      clip_project:       將影像特徵 (512 維) 投影到 prefix_length × GPT-2 hidden size
      gpt:                內建 GPT2LMHeadModel
    """

    def __init__(
        self,
        prefix_length: int,
        prefix_size: int = 512,
        gpt_model_name: str = "gpt2"
    ):
        """
        參數:
          prefix_length:   prefix 的長度 (token 數)
          prefix_size:     影像特徵維度，PathCLIP-Base 輸出為 512
          gpt_model_name:  GPT-2 模型名稱 (預設 "gpt2")
        """
        super().__init__()
        self.prefix_length = prefix_length

        # 載入預訓練 GPT-2 LM
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        # 取得 GPT-2 hidden size (embedding size)
        self.embedding_size = self.gpt.config.n_embd

        # clip_project: 512 → prefix_length × embedding_size
        self.clip_project = nn.Linear(prefix_size, prefix_length * self.embedding_size)

    def forward(self, inputs_embeds=None, labels=None):
        """
        前向傳播:
          inputs_embeds: [B, prefix_length + seq_len, embedding_size]
          labels:        [B, prefix_length + seq_len]
        直接將 inputs_embeds 與 labels 傳入 GPT-2 即可
        """
        return self.gpt(inputs_embeds=inputs_embeds, labels=labels)
