import torch, open_clip
from typing import Tuple
from torchvision.transforms import Compose

# ------------------ 使用前請確認 ------------------
# pip install --upgrade "open_clip_torch>=2.24.0" timm
# -------------------------------------------------

def load_pathclip(
    weight_path: str = "pt_model/pathclip-base.pt",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Tuple[torch.nn.Module, Compose]:
    """
    讀取 PathCLIP-Base 權重並回傳 (model, preprocess)。
    • 架構 = ViT-B/16（與官方 repo 一致）
    • 介面設計成與 clip.load(...) 相容，方便舊程式直接替換
    """
    # 1) 建立 ViT-B/16 架構並同時載入本地權重
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained=weight_path,
        device=device,
        jit=False,                # 與 clip.load 同步接口
        force_quick_gelu=True     # 官方用這個設定
    )

    model.eval()                 # 關閉 Dropout
    return model, preprocess     # 與 clip.load 回傳格式保持一致
