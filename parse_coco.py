# parse_coco.py

import torch
import clip
from PIL import Image
import pickle
import json
import os
import time

def parse_coco_gen(input_dir: str, output_dir: str, clip_model_type: str):
    """
    與原本 parse_coco 類似，但改為 generator。
    每處理部分資料就回傳 (i, total, msg)，讓上層可自訂如何顯示。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    captions_path = os.path.join(input_dir, 'annotations', 'train_caption.json')
    with open(captions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    yield (0, total, f"載入 {total} 筆 captions：{captions_path}\n")
    yield (0, total, f"使用 CLIP 模型：{clip_model_type}\n")

    all_embeddings = []
    all_captions = []

    start_time = time.time()

    for i, d in enumerate(data, start=1):
        img_id = d["image_id"]

        # train2014 / val2014
        filename = os.path.join(input_dir, 'train2014', f"COCO_train2014_{int(img_id):012d}.jpg")
        if not os.path.isfile(filename):
            filename = os.path.join(input_dir, 'val2014', f"COCO_val2014_{int(img_id):012d}.jpg")

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
