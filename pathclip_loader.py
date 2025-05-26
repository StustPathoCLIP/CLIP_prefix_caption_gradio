import torch
import open_clip
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
    載入 PathCLIP-Base 權重並回傳 (model, preprocess)
    
    參數:
      weight_path: 權重檔案路徑，請確認為 open_clip 格式
      device:      欲載入至之裝置 (CPU 或 CUDA)
    
    回傳:
      model:      已載入權重並設定為 eval 的 PathCLIP 模型
      preprocess: 相對應的影像前處理 transform
    """
    # 使用 open_clip 內建方法建立 ViT-B/16 架構並載入本地權重
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",       # PathCLIP-Base 架構
        pretrained=weight_path,      # 載入本地 .pt 權重
        device=device,
        jit=False,                   # 關閉 JIT，與 clip.load 同步
        force_quick_gelu=True        # 加速 GELU 計算
    )
    model.eval()                    # 設定為評估模式 (關閉 dropout 等)
    return model, preprocess