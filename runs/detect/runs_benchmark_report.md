# YOLO Runs Benchmark

- Generated at: `2026-09-22 12:51:36`
- Runs directory: `runs/detect`
- Selection mode: `best epoch by mAP@50-95`
- Sort by: `map50_95`
- Output: `runs\detect\runs_benchmark_report.md`

## Best Run

**traffic_model-2** đang đứng đầu theo `map50_95` với mAP@50-95 `0.6843`, mAP@50 `0.8886`, Precision `0.8250`, Recall `0.8229`.

## Ranking

| Rank | Run | Model | Epoch | Precision | Recall | mAP@50 | mAP@50-95 | Fitness | Time | Path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | traffic_model-2 | yolov8n.pt | 30 | 0.8250 | 0.8229 | 0.8886 | 0.6843 | 0.7047 | 1.02 h | runs\detect\traffic_model-2 |
| 2 | traffic_model_v2 | yolov8s.pt | 3 | 0.8356 | 0.6699 | 0.7498 | 0.5140 | 0.5376 | 14.8 min | runs\detect\traffic_model_v2 |
| 3 | traffic_model | yolov8n.pt | 1 | 0.7951 | 0.6003 | 0.6753 | 0.4700 | 0.4905 | 14.1 min | runs\detect\traffic_model |

## Training And Validation Loss

| Run | Train Box | Train Cls | Train DFL | Val Box | Val Cls | Val DFL |
| --- | --- | --- | --- | --- | --- | --- |
| traffic_model-2 | 0.7848 | 0.4258 | 0.9050 | 0.9067 | 0.5156 | 0.9516 |
| traffic_model_v2 | 1.1779 | 0.7876 | 1.0735 | 1.1626 | 0.7355 | 1.0323 |
| traffic_model | 1.1546 | 1.4388 | 1.0331 | 1.0343 | 0.8184 | 0.9813 |

## Run Configuration

| Run | Data | Image Size | Batch | Device | Optimizer | Epochs Done | Epochs Config |
| --- | --- | --- | --- | --- | --- | --- | --- |
| traffic_model-2 | dataset/data.yaml | 640 | 8 | 0 | auto | 30 | 30 |
| traffic_model_v2 | dataset/data.yaml | 640 | 8 | 0 | auto | 3 | 100 |
| traffic_model | dataset/data.yaml | 640 | 8 |  | auto | 1 | 30 |

## Available Artifacts

| Run | Artifacts |
| --- | --- |
| traffic_model-2 | results.png, confusion_matrix.png, confusion_matrix_normalized.png, BoxPR_curve.png, BoxF1_curve.png, BoxP_curve.png, BoxR_curve.png |
| traffic_model_v2 | - |
| traffic_model | - |

## Notes

- `Fitness` được tính theo công thức YOLO phổ biến: `0.1 * mAP@50 + 0.9 * mAP@50-95`.
- Mặc định script lấy dòng cuối cùng trong `results.csv`; dùng `--best-epoch` để lấy epoch tốt nhất theo fitness.
- Các run không có `results.csv` sẽ được bỏ qua.
