"""
Script: compare_runs.py
Chức năng: Tự động trích xuất và so sánh các lần chạy YOLO trong runs/detect/.

Cách chạy:
    # So sánh tất cả run trong runs/detect và xuất Markdown mặc định
    python compare_runs.py

    # Chỉ định thư mục runs và file output
    python compare_runs.py --runs-dir runs/detect --output reports/runs_benchmark.md

    # Xuất thêm CSV
    python compare_runs.py --csv reports/runs_benchmark.csv
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


METRIC_ALIASES = {
    "precision": ["metrics/precision(B)", "metrics/precision"],
    "recall": ["metrics/recall(B)", "metrics/recall"],
    "map50": ["metrics/mAP50(B)", "metrics/mAP50"],
    "map50_95": ["metrics/mAP50-95(B)", "metrics/mAP50-95"],
    "train_box_loss": ["train/box_loss"],
    "train_cls_loss": ["train/cls_loss"],
    "train_dfl_loss": ["train/dfl_loss"],
    "val_box_loss": ["val/box_loss"],
    "val_cls_loss": ["val/cls_loss"],
    "val_dfl_loss": ["val/dfl_loss"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare YOLO runs under runs/detect/")
    parser.add_argument("--runs-dir", default="runs/detect", help="Thư mục chứa các lần chạy YOLO")
    parser.add_argument(
        "--output",
        default="runs/detect/runs_benchmark_report.md",
        help="Đường dẫn file Markdown báo cáo",
    )
    parser.add_argument("--csv", default=None, help="Đường dẫn CSV tổng hợp nếu cần xuất thêm")
    parser.add_argument(
        "--sort-by",
        default="map50_95",
        choices=["map50_95", "map50", "precision", "recall", "fitness", "epoch"],
        help="Metric dùng để xếp hạng",
    )
    parser.add_argument(
        "--best-epoch",
        action="store_true",
        help="Lấy epoch tốt nhất theo mAP@50-95 thay vì dòng cuối cùng của results.csv",
    )
    return parser.parse_args()


def load_yaml(path):
    if not path.exists():
        return {}

    try:
        import yaml

        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except Exception:
        metadata = {}
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip("'\"")
        return metadata


def as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value):
    number = as_float(value)
    return int(number) if number is not None else None


def get_metric(row, key):
    for column in METRIC_ALIASES[key]:
        if column in row:
            return as_float(row[column])
    return None


def fitness(row):
    map50 = get_metric(row, "map50") or 0.0
    map50_95 = get_metric(row, "map50_95") or 0.0
    return 0.1 * map50 + 0.9 * map50_95


def read_results_csv(path, best_epoch=False):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = [{key.strip(): value for key, value in row.items()} for row in reader]

    if not rows:
        return None, []

    selected = max(rows, key=fitness) if best_epoch else rows[-1]
    return selected, rows


def detect_artifacts(run_dir):
    candidates = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
    ]
    return [name for name in candidates if (run_dir / name).exists()]


def collect_run(run_dir, best_epoch=False):
    results_path = run_dir / "results.csv"
    args_path = run_dir / "args.yaml"

    if not results_path.exists():
        return None

    selected, all_rows = read_results_csv(results_path, best_epoch=best_epoch)
    if selected is None:
        return None

    metadata = load_yaml(args_path)
    total_epochs = len(all_rows)
    selected_epoch = as_int(selected.get("epoch"))
    total_time = as_float(selected.get("time"))

    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "model": metadata.get("model", "-"),
        "data": metadata.get("data", "-"),
        "mode": metadata.get("mode", "-"),
        "task": metadata.get("task", "-"),
        "epochs_config": metadata.get("epochs", "-"),
        "epochs_done": total_epochs,
        "selected_epoch": selected_epoch,
        "imgsz": metadata.get("imgsz", "-"),
        "batch": metadata.get("batch", "-"),
        "device": metadata.get("device", "-"),
        "optimizer": metadata.get("optimizer", "-"),
        "precision": get_metric(selected, "precision"),
        "recall": get_metric(selected, "recall"),
        "map50": get_metric(selected, "map50"),
        "map50_95": get_metric(selected, "map50_95"),
        "fitness": fitness(selected),
        "train_box_loss": get_metric(selected, "train_box_loss"),
        "train_cls_loss": get_metric(selected, "train_cls_loss"),
        "train_dfl_loss": get_metric(selected, "train_dfl_loss"),
        "val_box_loss": get_metric(selected, "val_box_loss"),
        "val_cls_loss": get_metric(selected, "val_cls_loss"),
        "val_dfl_loss": get_metric(selected, "val_dfl_loss"),
        "time_seconds": total_time,
        "artifacts": detect_artifacts(run_dir),
    }


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_time(seconds):
    if seconds is None:
        return "-"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} min"
    return f"{minutes / 60:.2f} h"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def sort_runs(runs, sort_by):
    return sorted(
        runs,
        key=lambda item: item.get(sort_by) if item.get(sort_by) is not None else -1,
        reverse=True,
    )


def build_report(runs, args, output_path):
    ranked = sort_runs(runs, args.sort_by)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best = ranked[0] if ranked else None

    lines = [
        "# YOLO Runs Benchmark",
        "",
        f"- Generated at: `{now}`",
        f"- Runs directory: `{args.runs_dir}`",
        f"- Selection mode: `{'best epoch by mAP@50-95' if args.best_epoch else 'last epoch'}`",
        f"- Sort by: `{args.sort_by}`",
        f"- Output: `{output_path}`",
        "",
    ]

    if best:
        lines.extend(
            [
                "## Best Run",
                "",
                f"**{best['run']}** đang đứng đầu theo `{args.sort_by}` với "
                f"mAP@50-95 `{fmt(best['map50_95'])}`, mAP@50 `{fmt(best['map50'])}`, "
                f"Precision `{fmt(best['precision'])}`, Recall `{fmt(best['recall'])}`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Ranking",
            "",
            markdown_table(
                [
                    "Rank",
                    "Run",
                    "Model",
                    "Epoch",
                    "Precision",
                    "Recall",
                    "mAP@50",
                    "mAP@50-95",
                    "Fitness",
                    "Time",
                    "Path",
                ],
                [
                    [
                        rank,
                        item["run"],
                        item["model"],
                        item["selected_epoch"],
                        fmt(item["precision"]),
                        fmt(item["recall"]),
                        fmt(item["map50"]),
                        fmt(item["map50_95"]),
                        fmt(item["fitness"]),
                        fmt_time(item["time_seconds"]),
                        item["path"],
                    ]
                    for rank, item in enumerate(ranked, start=1)
                ],
            ),
            "",
            "## Training And Validation Loss",
            "",
            markdown_table(
                [
                    "Run",
                    "Train Box",
                    "Train Cls",
                    "Train DFL",
                    "Val Box",
                    "Val Cls",
                    "Val DFL",
                ],
                [
                    [
                        item["run"],
                        fmt(item["train_box_loss"]),
                        fmt(item["train_cls_loss"]),
                        fmt(item["train_dfl_loss"]),
                        fmt(item["val_box_loss"]),
                        fmt(item["val_cls_loss"]),
                        fmt(item["val_dfl_loss"]),
                    ]
                    for item in ranked
                ],
            ),
            "",
            "## Run Configuration",
            "",
            markdown_table(
                ["Run", "Data", "Image Size", "Batch", "Device", "Optimizer", "Epochs Done", "Epochs Config"],
                [
                    [
                        item["run"],
                        item["data"],
                        item["imgsz"],
                        item["batch"],
                        item["device"],
                        item["optimizer"],
                        item["epochs_done"],
                        item["epochs_config"],
                    ]
                    for item in ranked
                ],
            ),
            "",
            "## Available Artifacts",
            "",
            markdown_table(
                ["Run", "Artifacts"],
                [[item["run"], ", ".join(item["artifacts"]) if item["artifacts"] else "-"] for item in ranked],
            ),
            "",
            "## Notes",
            "",
            "- `Fitness` được tính theo công thức YOLO phổ biến: `0.1 * mAP@50 + 0.9 * mAP@50-95`.",
            "- Mặc định script lấy dòng cuối cùng trong `results.csv`; dùng `--best-epoch` để lấy epoch tốt nhất theo fitness.",
            "- Các run không có `results.csv` sẽ được bỏ qua.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(runs, csv_path):
    fieldnames = [
        "run",
        "model",
        "data",
        "selected_epoch",
        "epochs_done",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "fitness",
        "train_box_loss",
        "train_cls_loss",
        "train_dfl_loss",
        "val_box_loss",
        "val_cls_loss",
        "val_dfl_loss",
        "time_seconds",
        "imgsz",
        "batch",
        "device",
        "optimizer",
        "path",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in runs:
            writer.writerow({field: item.get(field) for field in fieldnames})


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)

    if not runs_dir.exists():
        print(f"[ERROR] Không tìm thấy thư mục runs: {runs_dir}")
        return

    runs = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        result = collect_run(run_dir, best_epoch=args.best_epoch)
        if result:
            runs.append(result)

    if not runs:
        print(f"[WARN] Không tìm thấy run nào có results.csv trong: {runs_dir}")
        return

    ranked = sort_runs(runs, args.sort_by)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(ranked, args, output_path), encoding="utf-8")

    if args.csv:
        write_csv(ranked, Path(args.csv))

    print("=" * 70)
    print(" SO SÁNH CÁC LẦN CHẠY YOLO")
    print("=" * 70)
    for rank, item in enumerate(ranked, start=1):
        print(
            f"{rank}. {item['run']:<24} "
            f"mAP@50-95={fmt(item['map50_95'])}  "
            f"mAP@50={fmt(item['map50'])}  "
            f"P={fmt(item['precision'])}  "
            f"R={fmt(item['recall'])}"
        )
    print("=" * 70)
    print(f"[OK] Báo cáo Markdown đã lưu tại: {output_path}")
    if args.csv:
        print(f"[OK] CSV tổng hợp đã lưu tại: {args.csv}")


if __name__ == "__main__":
    main()
