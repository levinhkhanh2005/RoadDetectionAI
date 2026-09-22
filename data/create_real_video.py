"""
Tao video thuc te tu anh traffic.jpg bang ky thuat camera panning
de dam bao YOLOv8 nhan dien duoc tat ca cac xe thuc te va ByteTrack tracking lien tuc!
"""

import cv2
import numpy as np

def create_panning_traffic_video(image_path="data/samples/traffic.jpg", output_path="data/samples/traffic.mp4"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: {image_path} not found")
        return
    
    h, w = img.shape[:2]
    # Kich thuoc viewport video
    crop_w, crop_h = int(w * 0.85), int(h * 0.85)
    
    fps = 20
    duration = 5 # 5 giay
    total_frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))
    
    # Panning tu tren xuong duoi (mo phong goc quay flycam/giam sat chuyen dong nhe)
    max_dy = h - crop_h
    max_dx = w - crop_w
    
    print(f"[INFO] Tao video panning giao thong thuc te tu {image_path}...")
    for f in range(total_frames):
        alpha = f / total_frames
        y_offset = int(alpha * max_dy)
        x_offset = int(alpha * max_dx * 0.5)
        
        frame = img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
        out.write(frame)
        
    out.release()
    print(f"[SUCCESS] Da tao xong video: {output_path}")

if __name__ == "__main__":
    create_panning_traffic_video()
