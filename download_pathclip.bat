@echo off
REM ——————————————————————————————
REM 下載 PathCLIP 權重 (pathclip-base.pt)
REM 存放到 pt_model\pathclip-base.pt
REM ——————————————————————————————

REM 切換到專案目錄（請根據實際路徑修改）
cd /d "E:\Side Project\CLIP\CLIP_prefix_caption_gradio"

REM 如果 pt_model 資料夾不存在就建立
if not exist pt_model (
    mkdir pt_model
)

REM 使用 curl 下載 Hugging Face 上的權重檔案
curl.exe -L ^
    "https://huggingface.co/jamessyx/pathclip/resolve/main/pathclip-base.pt" ^
    -o "pt_model\pathclip-base.pt"

if %ERRORLEVEL% neq 0 (
    echo.
    echo *** 下載失敗！請檢查網路連線或是否已安裝 curl.exe ***
) else (
    echo.
    echo ✔ 權重已成功下載並儲存為 pt_model\pathclip-base.pt
)

pause
