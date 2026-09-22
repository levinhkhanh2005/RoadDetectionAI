@echo off
echo =========================================================
echo CHAY HUAN LUYEN YOLOv8 TREN GPU (NVIDIA GTX 1650)
echo =========================================================
.\.venv_gpu\Scripts\python.exe train.py --epochs 30 --batch 8 --imgsz 640 --device 0
pause
