"""
Script: train.py
Chức năng: Huấn luyện hoặc Fine-tune mô hình YOLOv8 trên tập dữ liệu phương tiện giao thông tùy chỉnh.
Môn học: Trí Tuệ Nhân Tạo (Artificial Intelligence)

Cách chạy:
    python train.py --data dataset/data.yaml --epochs 50 --batch 16 --imgsz 640
"""

import os
import argparse
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train custom YOLOv8 model for vehicle detection")
    parser.add_argument("--data", type=str, default="dataset/data.yaml",
                        help="Đường dẫn tới file cấu hình data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="Mô hình gốc (yolov8s.pt cho transfer learning hoặc yolov8s.yaml để train từ đầu)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Số lượng epoch huấn luyện (khuyên dùng: 50 - 150, có Early Stopping tự dừng)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (tùy thuộc dung lượng VRAM GPU, khuyên dùng 8 cho 4GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Kích thước ảnh đầu vào (mặc định 640x640)")
    default_device = "0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device,
                        help="Thiết bị: '0' (GPU CUDA), 'cpu', hoặc để trống")
    parser.add_argument("--name", type=str, default="traffic_model",
                        help="Tên thư mục lưu kết quả thí nghiệm")
    return parser.parse_args()


def train_model():
    args = parse_args()

    print("=" * 60)
    print(" BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN XE (YOLOv8)")
    print("=" * 60)
    print(f"- Data config : {args.data}")
    print(f"- Base model  : {args.model}")
    print(f"- Epochs      : {args.epochs}")
    print(f"- Batch size  : {args.batch}")
    print(f"- Image size  : {args.imgsz}")
    print(f"- Device      : {args.device if args.device else 'Tự động'}")
    print("=" * 60)

    # 1. Khởi tạo mô hình
    # Nếu truyền file .pt: Dùng kỹ thuật Transfer Learning (Khuyên dùng)
    # Nếu truyền file .yaml: Khởi tạo trọng số ngẫu nhiên để train từ đầu (Train from scratch)
    model = YOLO(args.model)

    # 2. Bắt đầu huấn luyện
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        name=args.name,
        save=True,               # Tự động lưu checkpoint và best.pt
        save_period=10,          # Lưu checkpoint mỗi 10 epochs
        plots=True,              # Tự động vẽ biểu đồ Loss, PR Curve, Confusion Matrix
        optimizer="auto",        # Tự động chọn SGD hoặc AdamW
        verbose=True,
        # === Data Augmentation nâng cao ===
        mixup=0.1,               # Trộn 2 ảnh để tăng tính tổng quát hóa (generalization)
        copy_paste=0.1,          # Copy-paste đối tượng giữa các ảnh để tăng đa dạng
        degrees=10.0,            # Xoay ảnh ngẫu nhiên ±10° (mô phỏng camera nghiêng)
        shear=2.0,               # Biến dạng nghiêng nhẹ
        # === Learning Rate Schedule ===
        cos_lr=True,             # Cosine Annealing LR: hội tụ mượt và ổn định hơn linear decay
    )

    print("\n" + "=" * 60)
    print(" HUẤN LUYỆN HOÀN TẤT!")
    print(f"Trọng số tốt nhất (Best weights): runs/detect/{args.name}/weights/best.pt")
    print(f"Biểu đồ kết quả huấn luyện     : runs/detect/{args.name}/results.png")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
