import os
import glob
import torch
import gradio as gr
import clip
import requests
import time
import zipfile
from PIL import Image
from transformers import GPT2Tokenizer

# ------ 推理相關 (inference2.py) ------
from inference import (
    ClipCaptionModel,  # 請勿修改
    generate_beam,
    generate2,
    filter_state_dict
)

# ------ 訓練 & 前處理相關 (train.py) ------
import train as train_script

# ------ parse_coco.py 已支援自訂路徑 ------
try:
    from parse_coco import parse_coco
except ImportError:
    parse_coco = None

model_directory = "pt_model"

class ImageCaptioningApp:
    """
    提供 推理 + 訓練(下載 & parse_coco & train) 的 Gradio 主程式
    """

    def __init__(self):
        # 用於推理時的模型快取
        self.model_cache = {
            "model": None,
            "clip_model": None,
            "tokenizer": None,
            "preprocess": None,
            "device": None,
            "prefix_length": None,
            "model_path": None
        }


    # ===============  1. 推理  ===============
    def get_local_models(self, directory="pt_model"):
        """掃描指定目錄下的 pt / pth 模型檔案。"""
        model_files = []
        for ext in ['*.pt', '*.pth']:
            model_files.extend(glob.glob(os.path.join(directory, ext)))
        return [os.path.basename(f) for f in model_files] + ["自定義路徑"]

    def load_model(self, model_path, prefix_length=10):
        """載入推理模型及相關組件 (CLIP, GPT2 tokenizer)。"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # 由 inference.py 提供的 ClipCaptionModel
        model = ClipCaptionModel(prefix_length=prefix_length)

        print(f"正在載入模型：{model_path}")
        raw_state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        # 過濾掉多餘參數
        filtered_state_dict = filter_state_dict(raw_state_dict, model)
        model.load_state_dict(filtered_state_dict)
        print("模型載入成功！")

        model.eval()
        model = model.to(device)

        return model, clip_model, tokenizer, preprocess, device

    def get_model_path(self, model_selection, custom_path):
        """若使用者在下拉式選單選「自定義路徑」，則使用自訂路徑；否則使用下拉選項。"""
        if model_selection == "自定義路徑":
            return custom_path
        return model_selection

    def inference(
        self,
        image: Image.Image,
        model_selection: str,
        custom_model_path: str,
        prefix_length: int,
        use_beam_search: bool,
        beam_size: int,
        top_p: float,
        temperature: float,
        stop_token: str
    ):
        """
        使用 yield 分段回傳訊息，以在 Gradio 中即時顯示推理過程。
        """
        log = ""

        # 1) 檢查是否有上傳圖片
        if image is None:
            log += "請先上傳圖片。\n"
            yield log
            return

        model_path = self.get_model_path(os.path.join(model_directory, model_selection), custom_model_path)

        # 2) 檢查模型檔案路徑
        if not os.path.isfile(model_path):
            log += f"模型檔案路徑無效：{model_path}\n請確認檔案存在。\n"
            yield log
            return

        # 3) 若尚未載入模型，或 prefix_length / 路徑改變，則重新載入
        need_reload = (
            self.model_cache["model"] is None
            or self.model_cache["prefix_length"] != prefix_length
            or self.model_cache["model_path"] != model_path
        )
        if need_reload:
            log += f"正在載入模型：{model_path} (prefix_length={prefix_length})...\n"
            yield log
            try:
                model, clip_model, tokenizer, preprocess, device = self.load_model(model_path, prefix_length)
                self.model_cache.update({
                    "model": model,
                    "clip_model": clip_model,
                    "tokenizer": tokenizer,
                    "preprocess": preprocess,
                    "device": device,
                    "prefix_length": prefix_length,
                    "model_path": model_path
                })
                log = "模型載入成功！\n"  # 重置 log 只顯示載入成功的訊息
                yield log
            except Exception as e:
                log += f"模型載入失敗：{str(e)}\n"
                yield log
                return
        else:
            log += f"使用快取模型：{model_path}\n"
            yield log

        # 取出快取模型與組件
        model = self.model_cache["model"]
        clip_model = self.model_cache["clip_model"]
        tokenizer = self.model_cache["tokenizer"]
        preprocess = self.model_cache["preprocess"]
        device = self.model_cache["device"]

        try:
            # 4) CLIP Encode
            log = "\n=== 開始 CLIP Encode... ===\n"  # 清空 log 讓推理過程顯示為最新
            yield log
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image_tensor).to(device, dtype=torch.float32)
                prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
            log = "✔ CLIP Encode 完成！\n"  # 更新 log
            yield log

            # 5) 文字生成
            if use_beam_search:
                log = f"\n=== 使用 Beam Search (beam_size={beam_size}) 生成描述... ===\n"
                yield log
                output_text = generate_beam(
                    model=model,
                    tokenizer=tokenizer,
                    beam_size=beam_size,
                    embed=prefix_embed,
                    entry_length=67,
                    temperature=temperature,
                    stop_token=stop_token
                )[0]
            else:
                log = f"\n=== 使用 Nucleus Sampling (top_p={top_p}, temperature={temperature}) 生成描述... ===\n"
                yield log
                output_text = generate2(
                    model=model,
                    tokenizer=tokenizer,
                    embed=prefix_embed,
                    entry_count=1,
                    entry_length=67,
                    top_p=top_p,
                    temperature=temperature,
                    stop_token=stop_token
                )

            # 直接將推理結果覆蓋到輸出框，避免附加
            log = output_text
            yield log

        except Exception as e:
            log = f"生成過程發生錯誤：{str(e)}\n"
            yield log
            return



    # ===============  2. 訓練  ===============
    # 2-1 下載 COCO
    def download_coco_data(self, custom_dir: str):
        """
        使用 generator (yield) 分段回傳進度。
        下載 COCO Dataset (train2014.zip + val2014.zip) 及 train_captions.json
        都顯示「人類可讀大小 + 即時下載速度」到 Gradio UI，
        且關閉終端機進度顯示 (gdown quiet=True)，專注在分段日誌。
        """
        # 固定日誌 (累加)
        stable_log = ""
        # 下載進度 (只顯示最新的一行)
        progress_line = ""

        if not custom_dir.strip():
            custom_dir = "./data/coco"
        coco_dir = os.path.abspath(custom_dir)
        os.makedirs(coco_dir, exist_ok=True)

        stable_log += f"=== 開始下載 COCO 資料集至: {coco_dir} ===\n"
        yield stable_log + "\n" + progress_line

        # 1) 下載 train_captions.json (Google Drive)
        try:
            import gdown
        except ImportError:
            stable_log += "請先安裝 gdown: pip install gdown\n"
            yield stable_log
            return
        
        caption_gdown_id = "1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_"
        caption_output = os.path.join(coco_dir, "annotations_train_caption.json")
        annotations_dir = os.path.join(coco_dir, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)

        stable_log += "\n[1/5] 下載 train_captions.json 中...\n"
        yield stable_log

        try:
            # 關閉 gdown 的終端進度列
            gdown.download(
                url=f"https://drive.google.com/uc?id={caption_gdown_id}",
                output=caption_output,
                quiet=True
            )
            final_train_cap = os.path.join(annotations_dir, "train_caption.json")
            os.replace(caption_output, final_train_cap)
            stable_log += f"✔ 已下載 train_captions.json => {final_train_cap}\n"
            yield stable_log
        except Exception as e:
            stable_log += f"✘ 下載 train_caption.json 失敗: {str(e)}\n"
            yield stable_log
            return

        # 2) 下載 train2014.zip
        train_url = "http://images.cocodataset.org/zips/train2014.zip"
        train_zip_path = os.path.join(coco_dir, "train2014.zip")
        train2014_dir = os.path.join(coco_dir, "train2014")

        if not os.path.exists(train2014_dir):
            stable_log += "\n[2/5] 下載 train2014.zip 中...\n"
            yield stable_log

            try:
                # 逐 chunk 接收最新的進度訊息
                for msg in self._download_file_chunked(train_url, train_zip_path):
                    progress_line = msg  # 每次都覆蓋
                    yield stable_log + "\n" + progress_line

                stable_log += "✔ 已下載 train2014.zip\n"
                progress_line = ""
                yield stable_log

                stable_log += "解壓 train2014.zip 中...\n"
                yield stable_log
                os.makedirs(train2014_dir, exist_ok=True)
                self._unzip_file(train_zip_path, coco_dir)

                stable_log += f"✔ train2014.zip 解壓完成 => {train2014_dir}\n"
                yield stable_log
            except Exception as e:
                stable_log += f"✘ 下載或解壓 train2014.zip 失敗: {str(e)}\n"
                yield stable_log
                return
        else:
            stable_log += "\n[2/5] train2014 資料夾已存在，跳過下載/解壓\n"
            yield stable_log

        # 3) 下載 val2014.zip
        val_url = "http://images.cocodataset.org/zips/val2014.zip"
        val_zip_path = os.path.join(coco_dir, "val2014.zip")
        val2014_dir = os.path.join(coco_dir, "val2014")

        if not os.path.exists(val2014_dir):
            stable_log += "\n[3/5] 下載 val2014.zip 中...\n"
            yield stable_log

            try:
                for msg in self._download_file_chunked(val_url, val_zip_path):
                    progress_line = msg
                    yield stable_log + "\n" + progress_line

                stable_log += "✔ 已下載 val2014.zip\n"
                progress_line = ""
                yield stable_log

                stable_log += "解壓 val2014.zip 中...\n"
                yield stable_log
                os.makedirs(val2014_dir, exist_ok=True)
                self._unzip_file(val_zip_path, coco_dir)

                stable_log += f"✔ val2014.zip 解壓完成 => {val2014_dir}\n"
                yield stable_log
            except Exception as e:
                stable_log += f"✘ 下載或解壓 val2014.zip 失敗: {str(e)}\n"
                yield stable_log
                return
        else:
            stable_log += "\n[3/5] val2014 資料夾已存在，跳過下載/解壓\n"
            yield stable_log

        # 結尾
        stable_log += (
            "\n=== COCO Dataset 下載並解壓完成！===\n"
            f"train_caption.json => {final_train_cap}\n"
            f"train2014 => {train2014_dir}\n"
            f"val2014 => {val2014_dir}\n"
        )
        yield stable_log

    def _unzip_file(self, zip_path: str, dest_dir: str):
        """
        解壓縮檔案的簡易方法 (無分段 yield)。
        若需要顯示解壓進度，也可改成產生器 yield。
        """
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)

    def _download_file_chunked(self, url: str, output_path: str, chunk_size=1024*1024):
        """
        使用 requests 分段下載檔案，並以 yield 傳回「現代化」下載資訊：
        - 已下載 / 總大小 (KB/MB/GB)
        - 下載百分比
        - 即時下載速度 (KB/s 或 MB/s)
        不印任何終端機日誌，只用 yield 回傳給上層去顯示。
        """
        if os.path.exists(output_path):
            yield f"檔案已存在：{output_path}，略過下載"
            return

        # 發送 HTTP GET 請求 (串流)
        r = requests.get(url, stream=True)
        r.raise_for_status()
        total_size = int(r.headers.get('Content-Length', 0))
        downloaded = 0

        start_time = time.time()
        last_time = start_time  # 用於計算每個 chunk 的下載速度

        yield f"開始下載: {url}\n檔案大小: {self.human_readable_size(total_size)}"

        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    chunk_len = len(chunk)
                    downloaded += chunk_len

                    # 計算速度
                    current_time = time.time()
                    dt = current_time - last_time
                    last_time = current_time
                    speed_bps = chunk_len / dt if dt > 0 else 0.0

                    # 進度
                    pct = (downloaded / total_size) * 100 if total_size else 0
                    downloaded_human = self.human_readable_size(downloaded)
                    total_human = self.human_readable_size(total_size)
                    speed_human = self.human_readable_size(speed_bps) + "/s"

                    yield (
                        f"下載進度: {pct:.2f}%\n"
                        f"({downloaded_human}/{total_human})\n"
                        f"速度: {speed_human}"
                    )

        total_time = time.time() - start_time
        yield (
            f"下載完成: {output_path}\n"
            f"總耗時: {total_time:.2f} 秒"
        )

    def human_readable_size(self, size_in_bytes: float) -> str:
        """
        將 bytes 轉換成 KB, MB, GB... (1024進位)
        """
        step_unit = 1024.0
        unit_list = ["Bytes", "KB", "MB", "GB", "TB"]
        i = 0
        size = float(size_in_bytes)
        while size > step_unit and i < len(unit_list) - 1:
            size /= step_unit
            i += 1
        return f"{size:.2f} {unit_list[i]}"

    def _download_file(self, url, output_path, chunk_size=1024*1024):
        """使用 requests chunk 方式下載大檔"""
        import requests
        if os.path.exists(output_path):
            print(f"檔案已存在: {output_path}，略過下載")
            return
        print(f"\n開始下載: {url}")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        total_size = int(r.headers.get('Content-Length', 0))
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    done = int(50 * downloaded / total_size)
                    print(f"\r[{'█' * done}{'.' * (50 - done)}] {downloaded}/{total_size} bytes", end='')
        print(f"\n下載完成: {output_path}")

    def _unzip_file(self, zip_path, dest_dir):
        """使用 python zipfile 解壓"""
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"找不到壓縮檔: {zip_path}")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"解壓完成: {zip_path} 至 {dest_dir}")

    # 2-2 前處理 parse_coco
    def parse_coco_data(self, input_dir: str, output_dir: str, clip_model_type: str, data_type: str):
        """
        1) 呼叫 parse_coco_gen(...)，逐筆取得 (i, total, msg)
        2) 本函式用 stable_log + progress_line 的做法顯示
        3) 新增 data_type (COCO / TCGA)，若使用者沒輸入路徑，就依 data_type 設置預設值
        """
        stable_log = ""
        progress_line = ""

        # 第一次顯示
        stable_log += "=== 開始 parse_coco ===\n"
        yield stable_log

        # 根據 data_type 決定預設路徑
        if not input_dir.strip():
            if data_type == "TCGA":
                input_dir = "./data/tcga"
            else:
                input_dir = "./data/coco"

        if not output_dir.strip():
            if data_type == "TCGA":
                output_dir = "./data/tcga"
            else:
                output_dir = "./data/coco"

        stable_log += f"使用 Input dir: {input_dir}\n"
        stable_log += f"使用 Output dir: {output_dir}\n"
        stable_log += f"Clip Model Type: {clip_model_type}\n"
        stable_log += f"數據類型: {data_type}\n"
        yield stable_log

        try:
            from parse_coco import parse_coco_gen
        except ImportError:
            stable_log += "【失敗】parse_coco.py 未匯入或無法使用。\n"
            yield stable_log
            return

        try:
            gen = parse_coco_gen(input_dir, output_dir, clip_model_type)
            for (i, total, partial_msg) in gen:
                pct = (i / total) * 100 if total > 0 else 0
                progress_line = f"目前進度：{i}/{total} (約 {pct:.2f}%)\n"
                progress_line += partial_msg
                yield stable_log + "\n" + progress_line

            stable_log += "【成功】parse_coco 完成！\n"
            yield stable_log

        except Exception as e:
            stable_log += f"【失敗】parse_coco 過程中出現錯誤：{str(e)}\n"
            yield stable_log
            return


    # 2-3 執行訓練
    def run_training(
        self,
        data_path: str,
        out_dir: str,
        prefix: str,
        epochs: int,
        save_every: int,
        prefix_length: int,
        prefix_length_clip: int,
        bs: int,
        only_prefix: bool,
        mapping_type_str: str,
        num_layers: int,
        is_rn: bool,
        normalize_prefix: bool
    ):
        """
        我們將在這裡呼叫 train_script.train_generator(...)，
        將其回傳的進度資訊 (epoch, step, loss) 一一 yield 出去。
        """
        log = ""

        class ArgsObj: pass

        args = ArgsObj()
        args.data = data_path
        args.out_dir = out_dir
        args.prefix = prefix
        args.epochs = epochs
        args.save_every = save_every
        args.prefix_length = prefix_length
        args.prefix_length_clip = prefix_length_clip
        args.bs = bs
        args.only_prefix = only_prefix
        args.mapping_type = (
            train_script.MappingType.MLP if mapping_type_str == "mlp"
            else train_script.MappingType.Transformer
        )
        args.num_layers = num_layers
        args.is_rn = is_rn
        args.normalize_prefix = normalize_prefix

        # 前面跟原本一樣：顯示參數
        log += "=== 開始準備訓練 ===\n"
        log += f"data_path: {args.data}\n"
        log += f"out_dir: {args.out_dir}\n"
        log += f"epochs: {args.epochs}, save_every: {args.save_every}\n"
        log += f"prefix_length: {args.prefix_length}, prefix_length_clip: {args.prefix_length_clip}\n"
        log += f"batch_size: {args.bs}\n"
        log += f"only_prefix: {args.only_prefix}\n"
        log += f"mapping_type: {mapping_type_str}\n"
        log += f"num_layers: {args.num_layers}\n"
        log += f"is_rn: {args.is_rn}\n"
        log += f"normalize_prefix: {args.normalize_prefix}\n"
        yield log  # 第一次 yield (把參數印到前端)

        try:
            # 建立 dataset (跟你原本一樣)
            log += "\n=== 建立資料集 (ClipCocoDataset) 中... ===\n"
            yield log
            dataset = train_script.ClipCocoDataset(
                data_path=args.data,
                prefix_length=args.prefix_length,
                normalize_prefix=args.normalize_prefix
            )
            log += "✔ 資料集建立完成。\n"
            yield log

            prefix_dim = 640 if args.is_rn else 512
            log += "\n=== 建立模型中... ===\n"
            yield log

            if args.only_prefix:
                model = train_script.ClipCaptionPrefix(
                    prefix_length=args.prefix_length,
                    clip_length=args.prefix_length_clip,
                    prefix_size=prefix_dim,
                    num_layers=args.num_layers,
                    mapping_type=args.mapping_type
                )
                log += "Train only prefix (凍結 GPT-2)\n"
            else:
                model = train_script.ClipCaptionModel(
                    prefix_length=args.prefix_length,
                    clip_length=args.prefix_length_clip,
                    prefix_size=prefix_dim,
                    num_layers=args.num_layers,
                    mapping_type=args.mapping_type
                )
                log += "Train both prefix and GPT-2 (完整微調 GPT-2)\n"
            yield log

            log += "\n=== 開始訓練... ===\n"
            yield log

            # ---- 重點：呼叫 generator 版本 ----
            gen = train_script.train_generator(
                dataset=dataset,
                model=model,
                args=args,
                output_dir=args.out_dir,
                output_prefix=args.prefix
            )

            for (msg_str, epoch_i, step_i, total_steps, loss_val) in gen:
                # msg_str = 像 "=== Epoch 1 ===" 或 "✔ 訓練完成" 之類
                # epoch_i, step_i, total_steps, loss_val = batch-level 進度資訊
                if msg_str is not None:
                    # 代表是 "epoch開始" 或 "訓練完成" 之類的特殊訊息
                    log += msg_str
                    yield log
                else:
                    # 這次 yield 的是 batch 進度
                    if loss_val is not None:
                        pct = (step_i / total_steps) * 100 if total_steps else 0
                        progress_line = (
                            f"Epoch {epoch_i+1}, Step {step_i}/{total_steps} "
                            f"({pct:.2f}%), loss={loss_val:.4f}\n"
                        )
                        # 把這行「進度」附加到 log + yield
                        yield log + "\n" + progress_line

        except Exception as e:
            log += f"✘ 訓練過程發生錯誤：{str(e)}\n"
            yield log
            return

def update_parse_instructions(data_type):
    """
    根據用戶選擇的數據類型返回相應的指導說明。
    """
    if data_type == "TCGA":
        return (
            "【注意】您選擇的是 TCGA 數據，請確保：\n"
            "1. JSON 文件命名為 **train_caption_tcgaCOAD.json** 並放在 annotations 資料夾中；\n"
            "2. 圖片存放格式必須為 **train/<folder>/<image_id>.jpg**，例如：\n"
            "   TCGA-3L-AA1B-1 對應 train/TCGA-3L-AA1B/TCGA-3L-AA1B-1.jpg"
        )
    else:
        return (
            "【注意】您選擇的是 COCO 數據，請確保數據格式符合 COCO 標準，\n"
            "並確認 annotations 資料夾中存在 **train_caption.json**。"
        )

def update_data_path_placeholders(data_type):
    """
    根據數據類型更新「資料路徑」及「輸出 pkl 目錄」的 placeholder。
    當選擇 TCGA 時，預設為 "./data/tcga"；否則為 "./data/coco"。
    """
    if data_type == "TCGA":
        return gr.update(placeholder="./data/tcga"), gr.update(placeholder="./data/tcga")
    else:
        return gr.update(placeholder="./data/coco"), gr.update(placeholder="./data/coco")

def create_gradio_interface(app: ImageCaptioningApp):
    """建立 Gradio 前端介面：包含推理、訓練（使用COCO格式的數據或自定義數據）兩大功能。"""

    def update_custom_path_visibility(choice):
        """若選擇 '自定義路徑' 才顯示模型路徑輸入框。"""
        return gr.update(visible=(choice == "自定義路徑"))

    def toggle_top_p(use_beam):
        """勾選 Beam Search 時停用 top_p；否則啟用。"""
        if use_beam:
            return gr.update(interactive=False)
        else:
            return gr.update(interactive=True)

    with gr.Blocks() as demo:
        gr.Markdown("# CLIP_prefix_caption")
        gr.Markdown("本系統提供 **推理** 和 **訓練** ，您可以使用 COCO 格式的數據或自定義數據 (例如 TCGA COAD)。")

        with gr.Tabs():
            # ==================  (A) 推理 Tab  ================== #
            with gr.Tab("推理"):
                gr.Markdown("### 上傳圖片並選擇或輸入已訓練的模型")

                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="上傳圖片", type="pil")
                        prefix_length = gr.Number(
                            label="Prefix Length (推理用)",
                            value=10,
                            precision=0
                        )
                        use_beam_search = gr.Checkbox(
                            label="使用 Beam Search",
                            value=False
                        )
                        beam_size = gr.Slider(
                            label="Beam Size",
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=5
                        )
                        top_p = gr.Slider(
                            label="Top-p (Nucleus Sampling)",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.8
                        )
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.1,
                            maximum=2.0,
                            step=0.1,
                            value=1.0
                        )
                        stop_token = gr.Textbox(
                            label="停止符號 (Stop Token)",
                            value=".",
                            lines=1
                        )

                    with gr.Column():
                        model_selection = gr.Dropdown(
                            choices=app.get_local_models(directory = "pt_model"),
                            value=("自定義路徑" if not app.get_local_models(directory = "pt_model") else app.get_local_models(directory = "pt_model")[0]),
                            label="選擇本地模型"
                        )
                        custom_model_path = gr.Textbox(
                            label="自定義模型路徑",
                            placeholder="請輸入完整的模型檔案路徑",
                            visible=False
                        )
                        output_text = gr.Textbox(label="生成的描述", lines=5)
                        generate_button = gr.Button("產生描述", variant="primary")

                model_selection.change(
                    fn=update_custom_path_visibility,
                    inputs=model_selection,
                    outputs=custom_model_path
                )

                use_beam_search.change(
                    fn=toggle_top_p,
                    inputs=use_beam_search,
                    outputs=top_p
                )

                generate_button.click(
                    fn=app.inference,
                    inputs=[
                        image_input,
                        model_selection,
                        custom_model_path,
                        prefix_length,
                        use_beam_search,
                        beam_size,
                        top_p,
                        temperature,
                        stop_token
                    ],
                    outputs=output_text
                )

            # ==================  (B) 訓練 Tab  ================== #
            with gr.Tab("訓練"):
                gr.Markdown("### 步驟1 → 步驟2 → 步驟3")
                gr.Markdown("您可以使用COCO格式的數據集，或者提供您自己的數據集 (從 Step2. 開始)，只需遵循COCO格式。")

                # --- Step 1: 下載 COCO dataset --- 
                gr.Markdown("#### Step1. 下載資料 (COCO images + train_captions.json)")

                with gr.Row():
                    with gr.Column():
                        download_dir_input = gr.Textbox(
                            label="下載目錄 (空白預設: ./data/coco)",
                            placeholder="./data/coco"
                        )
                        download_button = gr.Button(
                            "下載 COCO 資料 (train2014 + val2014 + train_caption.json)",
                            variant="primary"
                        )
                    with gr.Column():
                        download_output = gr.Textbox(
                            label="下載/解壓 輸出紀錄",
                            lines=6,
                            interactive=False
                        )

                download_button.click(
                    fn=app.download_coco_data,
                    inputs=[download_dir_input],
                    outputs=download_output
                )

                # --- Step 2: parse_coco 產生 pkl ---
                gr.Markdown("#### Step2. 前處理 (產生 pkl)")
                with gr.Row():
                    with gr.Column():
                        # 新增選擇數據類型的控件
                        data_type_radio = gr.Radio(
                            label="訓練數據類型", 
                            choices=["COCO", "TCGA"], 
                            value="COCO"
                        )
                        input_dir_box = gr.Textbox(
                            label="資料路徑", 
                            placeholder="./data/coco"
                        )
                        output_dir_box = gr.Textbox(
                            label="輸出 pkl 目錄", 
                            placeholder="./data/coco"
                        )
                        clip_model_type_input = gr.Radio(
                            label="Clip Model Type", 
                            choices=["ViT-B/32", "RN50", "RN101", "RN50x4"], 
                            value="ViT-B/32"
                        )
                        parse_button = gr.Button(
                            "Parse (pkl)", 
                            variant="primary"
                        )
                    with gr.Column():
                        data_type_instructions = gr.Markdown(update_parse_instructions("COCO"))
                        parse_output = gr.Textbox(label="parse 輸出", lines=6, interactive=False)
                # 更新說明及 placeholder：當用戶切換數據類型時
                data_type_radio.change(fn=update_parse_instructions, inputs=data_type_radio, outputs=data_type_instructions)
                data_type_radio.change(fn=update_data_path_placeholders, inputs=data_type_radio, outputs=[input_dir_box, output_dir_box])
                parse_button.click(
                    fn=app.parse_coco_data,
                    inputs=[input_dir_box, output_dir_box, clip_model_type_input, data_type_radio],
                    outputs=parse_output
                )

                # --- Step 3: 執行訓練 ---
                gr.Markdown("#### Step3. 訓練模型")

                with gr.Row():
                    with gr.Column():
                        data_path = gr.Textbox(
                            label="訓練資料 (pkl) 路徑",
                            value="./data/coco/oscar_split_ViT-B_32_train.pkl"
                        )
                        out_dir = gr.Textbox(
                            label="輸出檔案目錄 (out_dir)",
                            value="./checkpoints"
                        )
                        prefix = gr.Textbox(
                            label="輸出檔案前綴 (prefix)",
                            value="coco_prefix"
                        )
                        epochs = gr.Slider(
                            label="訓練 Epoch 數",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=10
                        )
                        save_every = gr.Slider(
                            label="每幾個 Epoch 存檔",
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=1
                        )
                        prefix_length_train = gr.Slider(
                            label="Prefix Length (訓練用)",
                            minimum=1,
                            maximum=40,
                            step=1,
                            value=10
                        )
                        prefix_length_clip = gr.Slider(
                            label="Prefix Length for CLIP",
                            minimum=1,
                            maximum=40,
                            step=1,
                            value=10
                        )
                        bs = gr.Slider(
                            label="Batch Size",
                            minimum=1,
                            maximum=256,
                            step=1,
                            value=40
                        )
                    with gr.Column():
                        only_prefix = gr.Checkbox(
                            label="只訓練 Prefix (凍結 GPT-2)",
                            value=False
                        )
                        mapping_type_str = gr.Radio(
                            label="Mapping Type",
                            choices=["mlp", "transformer"],
                            value="mlp"
                        )
                        num_layers = gr.Slider(
                            label="Transformer 層數 (num_layers)",
                            minimum=1,
                            maximum=12,
                            step=1,
                            value=8
                        )
                        is_rn = gr.Checkbox(
                            label="是否使用 RN 模型 (prefix_dim=640)",
                            value=False
                        )
                        normalize_prefix = gr.Checkbox(
                            label="是否 Normalized Prefix",
                            value=False
                        )
                        start_training_button = gr.Button(
                            "開始訓練",
                            variant="primary"
                        )
                        training_output = gr.Textbox(
                            label="訓練資訊輸出",
                            lines=10,
                            interactive=False
                        )

                start_training_button.click(
                    fn=app.run_training,
                    inputs=[
                        data_path,
                        out_dir,
                        prefix,
                        epochs,
                        save_every,
                        prefix_length_train,
                        prefix_length_clip,
                        bs,
                        only_prefix,
                        mapping_type_str,
                        num_layers,
                        is_rn,
                        normalize_prefix
                    ],
                    outputs=training_output
                )

        gr.Markdown("---")
        gr.Markdown("**系統提示**：如有 GPU，將自動使用 CUDA；否則使用 CPU。更多資訊，請參見 [GitHub 原始碼](https://github.com/rmokady/CLIP_prefix_caption)。")
        

    return demo



def main():
    app = ImageCaptioningApp()
    demo = create_gradio_interface(app)
    # 若需要指定其他 host/port，可於下方修改
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
