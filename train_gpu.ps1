# Chạy huấn luyện YOLOv8 bằng GPU NVIDIA GTX 1650
Write-Host "=========================================================" -ForegroundColor Green
Write-Host "KHOI CHAY HUAN LUYEN TREN GPU: NVIDIA GeForce GTX 1650" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Green

.\.venv_gpu\Scripts\python.exe train.py --epochs 30 --batch 8 --imgsz 640 --device 0
