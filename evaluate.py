"""
Script: evaluate.py
Chức năng: Đánh giá một hoặc nhiều mô hình YOLOv8 trên tập val/test và tự động
           xuất báo cáo Markdown benchmark so sánh chi tiết giữa các mô hình.

Cách chạy:
    # Đánh giá mô hình mặc định trên tập test
    python evaluate.py

    # Đánh giá trên tập validation
    python evaluate.py --split val

    # Benchmark nhiều mô hình
    python evaluate.py --models models/best.pt models/yolov8n.pt models/yolov8s.pt

    # Chỉ định file báo cáo Markdown
    python evaluate.py --models models/best.pt yolov8n.pt --report reports/benchmark.md
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from statistics import mean

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 models on test/val dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Đường dẫn tới một file trọng số mô hình (.pt). Giữ để tương thích lệnh cũ.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Danh sách file trọng số để benchmark nhiều mô hình.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Đường dẫn tới file cấu hình data.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Tập dữ liệu để đánh giá: 'test' hoặc 'val'",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size khi đánh giá",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Ngưỡng confidence thấp để tính mAP chính xác",
    )
    parser.add_argument("--iou", type=float, default=0.6, help="Ngưỡng IoU cho NMS")
    default_device = "0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device, help="Thiết bị: '0' (GPU), 'cpu'")
    parser.add_argument(
        "--name",
        type=str,
        default="evaluate_results",
        help="Tên thư mục gốc lưu kết quả đánh giá trong runs/detect/",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Đường dẫn file Markdown benchmark. Mặc định: runs/detect/<name>/benchmark_report.md",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Tắt xuất confusion matrix, PR curve, F1 curve để chạy nhanh hơn.",
    )
    return parser.parse_args()


def model_label(model_path):
    path = Path(model_path)
    if path.parent.name == "weights":
        return f"{path.parent.parent.name}/{path.stem}"
    return path.stem


def fmt_float(value, digits=4):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_percent(value):
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def fmt_ms(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def sanitize_run_name(name):
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return safe.strip("_") or "model"


def get_speed(metrics):
    speed = getattr(metrics, "speed", None)
    if not isinstance(speed, dict):
        return {}
    return {
        "preprocess_ms": speed.get("preprocess"),
        "inference_ms": speed.get("inference"),
        "loss_ms": speed.get("loss"),
        "postprocess_ms": speed.get("postprocess"),
    }


def to_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def metric_value(value):
    if callable(value):
        value = value()
    if hasattr(value, "item"):
        return value.item()
    return value


def collect_result(model_path, model, metrics, run_name, save_dir):
    box = metrics.box
    class_names = model.names
    ap50_values = to_list(getattr(box, "ap50", None))
    map_values = to_list(getattr(box, "maps", None))

    per_class = []
    for class_id, class_name in class_names.items():
        ap50 = ap50_values[class_id] if class_id < len(ap50_values) else None
        map_50_95 = map_values[class_id] if class_id < len(map_values) else None
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "ap50": ap50,
                "map50_95": map_50_95,
            }
        )

    return {
        "model": str(model_path),
        "label": model_label(model_path),
        "run_name": run_name,
        "save_dir": str(save_dir),
        "precision": getattr(box, "mp", None),
        "recall": getattr(box, "mr", None),
        "map50": getattr(box, "map50", None),
        "map50_95": getattr(box, "map", None),
        "fitness": metric_value(getattr(metrics, "fitness", None)),
        "per_class": per_class,
        **get_speed(metrics),
    }


def sort_key(result):
    return (
        result.get("map50_95") if result.get("map50_95") is not None else -1,
        result.get("map50") if result.get("map50") is not None else -1,
        result.get("precision") if result.get("precision") is not None else -1,
    )


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(results, args, report_path):
    ranked = sorted(results, key=sort_key, reverse=True)
    best = ranked[0] if ranked else None
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_rows = []
    for rank, item in enumerate(ranked, start=1):
        summary_rows.append(
            [
                rank,
                item["label"],
                item["model"],
                fmt_float(item["precision"]),
                fmt_float(item["recall"]),
                fmt_float(item["map50"]),
                fmt_float(item["map50_95"]),
                fmt_float(item["fitness"]),
                fmt_ms(item.get("inference_ms")),
                item["save_dir"],
            ]
        )

    lines = [
        "# YOLOv8 Benchmark Report",
        "",
        f"- Generated at: `{now}`",
        f"- Data config: `{args.data}`",
        f"- Split: `{args.split}`",
        f"- Image size: `{args.imgsz}`",
        f"- Batch size: `{args.batch}`",
        f"- Confidence threshold: `{args.conf}`",
        f"- IoU threshold: `{args.iou}`",
        f"- Device: `{args.device}`",
        f"- Report path: `{report_path}`",
        "",
    ]

    if best:
        lines.extend(
            [
                "## Best Model",
                "",
                f"**{best['label']}** đạt mAP@50-95 cao nhất với `{fmt_float(best['map50_95'])}` "
                f"(mAP@50 `{fmt_float(best['map50'])}`, Precision `{fmt_float(best['precision'])}`, "
                f"Recall `{fmt_float(best['recall'])}`).",
                "",
            ]
        )

    lines.extend(
        [
            "## Overall Benchmark",
            "",
            markdown_table(
                [
                    "Rank",
                    "Model",
                    "Path",
                    "Precision",
                    "Recall",
                    "mAP@50",
                    "mAP@50-95",
                    "Fitness",
                    "Inference ms/img",
                    "Run output",
                ],
                summary_rows,
            ),
            "",
            "## Percent Metrics",
            "",
            markdown_table(
                ["Model", "Precision", "Recall", "mAP@50", "mAP@50-95"],
                [
                    [
                        item["label"],
                        fmt_percent(item["precision"]),
                        fmt_percent(item["recall"]),
                        fmt_percent(item["map50"]),
                        fmt_percent(item["map50_95"]),
                    ]
                    for item in ranked
                ],
            ),
            "",
            "## Per-Class Detail",
            "",
        ]
    )

    for item in ranked:
        per_class_rows = [
            [row["class_id"], row["class_name"], fmt_float(row["ap50"]), fmt_float(row["map50_95"])]
            for row in item["per_class"]
        ]
        avg_ap50 = mean([row["ap50"] for row in item["per_class"] if row["ap50"] is not None]) if per_class_rows else None
        avg_map = mean([row["map50_95"] for row in item["per_class"] if row["map50_95"] is not None]) if per_class_rows else None
        lines.extend(
            [
                f"### {item['label']}",
                "",
                markdown_table(["Class ID", "Class", "AP@50", "mAP@50-95"], per_class_rows),
                "",
                f"- Average AP@50 by class: `{fmt_float(avg_ap50)}`",
                f"- Average mAP@50-95 by class: `{fmt_float(avg_map)}`",
                f"- Output folder: `{item['save_dir']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- Ranking ưu tiên mAP@50-95, sau đó mAP@50 và Precision.",
            "- Mỗi model được lưu vào một thư mục riêng trong `runs/detect/` để tránh ghi đè kết quả.",
            "- Các biểu đồ như confusion matrix, PR curve và F1 curve nằm trong từng thư mục output nếu không dùng `--no-plots`.",
            "",
        ]
    )
    return "\n".join(lines)


def print_result(result):
    print("\n" + "=" * 60)
    print(f" KẾT QUẢ ĐÁNH GIÁ: {result['label']}")
    print("=" * 60)
    print(f"{'Chỉ số':<30} {'Giá trị':>10}")
    print("-" * 42)
    print(f"{'Precision:':<30} {fmt_float(result['precision']):>10}")
    print(f"{'Recall:':<30} {fmt_float(result['recall']):>10}")
    print(f"{'mAP@50:':<30} {fmt_float(result['map50']):>10}")
    print(f"{'mAP@50-95:':<30} {fmt_float(result['map50_95']):>10}")
    print(f"{'Inference ms/img:':<30} {fmt_ms(result.get('inference_ms')):>10}")

    print(f"\n{'Lớp':<20} {'AP@50':>10} {'mAP@50-95':>12}")
    print("-" * 44)
    for row in result["per_class"]:
        print(f"{row['class_name']:<20} {fmt_float(row['ap50']):>10} {fmt_float(row['map50_95']):>12}")


def evaluate():
    args = parse_args()
    model_paths = args.models if args.models else [args.model]

    print("=" * 70)
    print(" ĐÁNH GIÁ / BENCHMARK MÔ HÌNH NHẬN DIỆN PHƯƠNG TIỆN (YOLOv8)")
    print("=" * 70)
    print(f"- Models      : {', '.join(model_paths)}")
    print(f"- Data config : {args.data}")
    print(f"- Split       : {args.split}")
    print(f"- Image size  : {args.imgsz}")
    print(f"- Batch size  : {args.batch}")
    print(f"- Device      : {args.device}")
    print("=" * 70)

    missing = [path for path in model_paths if not os.path.exists(path)]
    if missing:
        print("\n[ERROR] Không tìm thấy file mô hình:")
        for path in missing:
            print(f"        - {path}")
        print("[HINT]  Hãy kiểm tra đường dẫn hoặc huấn luyện trước bằng lệnh: python train.py")
        return

    results = []
    for index, model_path in enumerate(model_paths, start=1):
        label = model_label(model_path)
        run_name = f"{args.name}_{index:02d}_{sanitize_run_name(label)}" if len(model_paths) > 1 else args.name

        print(f"\n[INFO] ({index}/{len(model_paths)}) Đang tải mô hình: {model_path}")
        model = YOLO(model_path)

        print(f"[INFO] Đánh giá '{label}' trên tập '{args.split}'...")
        metrics = model.val(
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=args.device if args.device else None,
            name=run_name,
            plots=not args.no_plots,
            verbose=True,
        )

        save_dir = getattr(metrics, "save_dir", Path("runs") / "detect" / run_name)
        result = collect_result(model_path, model, metrics, run_name, save_dir)
        results.append(result)
        print_result(result)

    report_path = Path(args.report) if args.report else Path("runs") / "detect" / args.name / "benchmark_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(results, args, report_path), encoding="utf-8")

    ranked = sorted(results, key=sort_key, reverse=True)
    print("\n" + "=" * 70)
    print(" BẢNG XẾP HẠNG BENCHMARK")
    print("=" * 70)
    for rank, item in enumerate(ranked, start=1):
        print(
            f"{rank}. {item['label']:<28} "
            f"mAP@50-95={fmt_float(item['map50_95'])}  "
            f"mAP@50={fmt_float(item['map50'])}  "
            f"P={fmt_float(item['precision'])}  "
            f"R={fmt_float(item['recall'])}"
        )
    print("=" * 70)
    print(f"[OK] Báo cáo Markdown đã lưu tại: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
