# parse_coco.py

import torch
import clip
from PIL import Image
import pickle
import json
import os
import time
from pathclip_loader import load_pathclip

def parse_coco_gen(input_dir: str, output_dir: str, clip_model_type: str):
    """
    與原本 parse_coco 類似，但改為 generator。
    每處理部分資料就回傳 (i, total, msg)，讓上層可自訂如何顯示。
    
    新增支援自己的數據：當 annotations 資料夾內存在 "train_caption_tcgaCOAD.json" 時，
    表示使用自己的數據，其 JSON 格式為：
    
    [
        {
            "image_id": "TCGA-3L-AA1B-1",
            "id": 1,
            "caption": "..."
        },
        ...
    ]
    
    此時對應的圖片檔案位置為：
        train/<folder>/<image_id>.jpg
    例如：image_id "TCGA-3L-AA1B-1" 對應的路徑為 "train/TCGA-3L-AA1B/TCGA-3L-AA1B-1.jpg"
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === (A) 挑選影像編碼器 ===
    if clip_model_type == "PathCLIP":
        clip_model, preprocess = load_pathclip("pt_model/pathclip-base.pt", device)
    else:
        clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    
    # 判斷使用哪個 annotation 文件
    tcga_annotation_path = os.path.join(input_dir, 'annotations', 'train_caption_tcgaCOAD.json')
    coco_annotation_path = os.path.join(input_dir, 'annotations', 'train_caption.json')
    if os.path.isfile(tcga_annotation_path):
        captions_path = tcga_annotation_path
        dataset_type = 'tcga'
    else:
        captions_path = coco_annotation_path
        dataset_type = 'coco'

    with open(captions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    yield (0, total, f"載入 {total} 筆 captions：{captions_path}\n")
    yield (0, total, f"使用 CLIP 模型：{clip_model_type}\n")
    yield (0, total, f"資料集類型：{dataset_type}\n")

    all_embeddings = []
    all_captions = []
    start_time = time.time()

    for i, d in enumerate(data, start=1):
        img_id = d["image_id"]

        if dataset_type == 'tcga':
            # 根據 image_id 取得子資料夾名稱（去除最後一個 '-' 後的部分）
            folder_name = img_id.rsplit("-", 1)[0]
            filename = os.path.join(input_dir, "train", folder_name, f"{img_id}.jpg")
        else:
            # COCO 資料集處理：先找 train2014，再找 val2014
            try:
                int_img_id = int(img_id)
            except ValueError:
                int_img_id = 0
            filename = os.path.join(input_dir, 'train2014', f"COCO_train2014_{int_img_id:012d}.jpg")
            if not os.path.isfile(filename):
                filename = os.path.join(input_dir, 'val2014', f"COCO_val2014_{int_img_id:012d}.jpg")

        if not os.path.isfile(filename):
            yield (i, total, f"警告: 找不到圖片檔案 {filename}\n")
            continue

        image = Image.open(filename).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image_tensor).cpu()

        d["clip_embedding"] = i - 1
        all_embeddings.append(prefix)
        all_captions.append(d)

        # 每 100 筆回報一次
        if i % 100 == 0:
            elapsed = time.time() - start_time
            msg = f"已處理 {i}/{total} 筆，耗時 {elapsed:.1f} 秒...\n"
            yield (i, total, msg)

    # 全部處理完成
    model_name_sanitized = clip_model_type.replace('/', '_')
    out_path = os.path.join(output_dir, f"oscar_split_{model_name_sanitized}_train.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump({
            "clip_embedding": torch.cat(all_embeddings, dim=0),
            "captions": all_captions
        }, f)

    elapsed_all = time.time() - start_time
    final_msg = (
        f"parse_coco 完成！總共 {total} 筆 => {out_path}\n"
        f"總耗時 {elapsed_all:.1f} 秒\n"
    )
    yield (total, total, final_msg)
