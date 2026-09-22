"""
Script: setup_dataset.py
Chức năng: Chuẩn hóa và sao chép toàn bộ dữ liệu từ E:\\trainAI vào thư mục dataset/ trong dự án
theo cấu trúc chuẩn mực của YOLO:
    dataset/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
"""

import os
import sys
import shutil
import yaml

# Dam bao encoding tren Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def setup_dataset():
    src_base = r"E:\trainAI"
    dst_base = os.path.abspath("dataset")

    print("=" * 60)
    print("CHUAN HOA DU LIEU VAO DU AN: RoadDetectionAI/dataset")
    print(f"Nguon: {src_base}")
    print(f"Dich : {dst_base}")
    print("=" * 60)

    if not os.path.exists(src_base):
        print(f"[ERROR] Không tìm thấy thư mục nguồn: {src_base}")
        return

    # 1. Tạo cấu trúc thư mục đích
    splits = [
        ("train", "train"),
        ("valid", "val"),
        ("test", "test")
    ]

    for src_split, dst_split in splits:
        img_dst = os.path.join(dst_base, "images", dst_split)
        lbl_dst = os.path.join(dst_base, "labels", dst_split)
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        # Sao chep anh
        img_src = os.path.join(src_base, src_split, "images")
        if os.path.exists(img_src):
            print(f"[COPY] Dang sao chep {src_split}/images -> images/{dst_split}...")
            for f in os.listdir(img_src):
                s = os.path.join(img_src, f)
                d = os.path.join(img_dst, f)
                shutil.copy2(s, d)

        # Sao chep labels
        lbl_src = os.path.join(src_base, src_split, "labels")
        if os.path.exists(lbl_src):
            print(f"[COPY] Dang sao chep {src_split}/labels -> labels/{dst_split}...")
            for f in os.listdir(lbl_src):
                s = os.path.join(lbl_src, f)
                d = os.path.join(lbl_dst, f)
                shutil.copy2(s, d)

    # 2. Tao file data.yaml chuan hoa ben trong thu muc dataset
    data_yaml_content = {
        "path": os.path.abspath("dataset").replace("\\", "/"),  # Duong dan tuyet doi chuan POSIX
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 4,
        "names": ["car", "motorcycle", "bus", "truck"]
    }

    yaml_path = os.path.join(dst_base, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)

    print("\n" + "=" * 60)
    print("CHUAN HOA HOAN TAT!")
    print(f"- File cau hinh: {yaml_path}")
    print(f"- So anh train : {len(os.listdir(os.path.join(dst_base, 'images', 'train')))}")
    print(f"- So anh val   : {len(os.listdir(os.path.join(dst_base, 'images', 'val')))}")
    if os.path.exists(os.path.join(dst_base, "images", "test")):
        print(f"- So anh test  : {len(os.listdir(os.path.join(dst_base, 'images', 'test')))}")
    print("=" * 60)

if __name__ == "__main__":
    setup_dataset()
