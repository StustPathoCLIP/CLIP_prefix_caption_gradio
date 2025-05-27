# visualization.py
import cv2, numpy as np, torch

# ---------- 1) 將梯度張量轉 0~1 numpy ----------
def grad_to_heatmap(grad: torch.Tensor, orig_hw):
    # grad: [3, H, W] 或 [H, W]
    g = grad.detach().abs().sum(0).cpu().numpy()          # → [H,W]
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)        # normalize
    g = cv2.resize(g, orig_hw[::-1])                      # 回到原尺寸
    return g                                              # 0~1

# ---------- 2) 疊圖 ----------
def overlay_heatmap(pil_img, heatmap, alpha=0.45):
    """pil_img: PIL.Image  heatmap: 0~1 numpy"""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    heat = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    out  = cv2.addWeighted(heat, alpha, img, 1-alpha, 0)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)           # 回傳 np.uint8 RGB
