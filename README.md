Linux：
# 1. 先安裝通用依賴
pip install -r requirements.txt

# 2. 再安裝對應 CUDA 輪子（Linux）
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 \
  -f https://download.pytorch.org/whl/cu121/torch_stable.html

Windows（PowerShell / CMD）：

pip install -r requirements.txt
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 `
  --index-url https://download.pytorch.org/whl/cu121
