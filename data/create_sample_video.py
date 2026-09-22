"""
Script tao video mau mo phong giao thong (Sample Traffic Video Generator)
Tao ra file video data/samples/traffic.mp4 voi cac xe di chuyen qua vach dem
de nguoi dung co the test truc tiep chuc nang Object Tracking va Counting Line.
"""

import os
import cv2
import numpy as np

def generate_traffic_video(output_path="data/samples/traffic.mp4", duration_sec=8, fps=25):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    num_frames = duration_sec * fps
    
    # Tao nen duong pho (asphalt road with lanes)
    def draw_background():
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Mat duong mau xam
        frame[:, :] = (55, 55, 60)
        # Vi he 2 ben
        frame[:, :80] = (140, 140, 145)
        frame[:, width-80:] = (140, 140, 145)
        # Vach ke duong mau vang o giua
        cv2.line(frame, (width // 2 - 3, 0), (width // 2 - 3, height), (0, 215, 255), 2)
        cv2.line(frame, (width // 2 + 3, 0), (width // 2 + 3, height), (0, 215, 255), 2)
        # Vach trang phan lan (net dut)
        for y in range(0, height, 40):
            cv2.line(frame, (width // 4, y), (width // 4, y + 20), (255, 255, 255), 2)
            cv2.line(frame, (3 * width // 4, y), (3 * width // 4, y + 20), (255, 255, 255), 2)
        return frame

    # Danh sach xe mo phong (x, start_y, speed, color, w, h, label)
    vehicles = [
        {"x": 160, "y": -80, "speed": 3.5, "color": (40, 40, 220), "w": 65, "h": 120, "type": "car"},
        {"x": 300, "y": -220, "speed": 4.2, "color": (220, 150, 30), "w": 40, "h": 70, "type": "motorcycle"},
        {"x": 180, "y": -400, "speed": 2.8, "color": (30, 200, 220), "w": 80, "h": 190, "type": "bus"},
        {"x": 580, "y": height + 60, "speed": -3.8, "color": (200, 200, 200), "w": 65, "h": 120, "type": "car"},
        {"x": 720, "y": height + 180, "speed": -4.5, "color": (40, 220, 50), "w": 40, "h": 75, "type": "motorcycle"},
        {"x": 600, "y": height + 350, "speed": -2.5, "color": (180, 50, 180), "w": 85, "h": 180, "type": "truck"},
    ]

    base_frame = draw_background()

    print(f"[INFO] Dang khoi tao video mo phong tai: {output_path} ({num_frames} frames)...")
    for f in range(num_frames):
        frame = base_frame.copy()
        for v in vehicles:
            v["y"] += v["speed"]
            vy = int(v["y"])
            vx = int(v["x"])
            vw = v["w"]
            vh = v["h"]
            
            # Neu xe dang o trong khung hinh
            if -vh < vy < height + vh:
                # Ve than xe
                cv2.rectangle(frame, (vx - vw//2, vy - vh//2), (vx + vw//2, vy + vh//2), v["color"], -1)
                # Kính chắn gió
                cv2.rectangle(frame, (vx - vw//2 + 5, vy - vh//4), (vx + vw//2 - 5, vy + vh//4), (80, 80, 80), -1)
                # Đèn xe
                if v["speed"] > 0: # Đi xuống
                    cv2.circle(frame, (vx - vw//2 + 8, vy + vh//2 - 5), 4, (0, 255, 255), -1)
                    cv2.circle(frame, (vx + vw//2 - 8, vy + vh//2 - 5), 4, (0, 255, 255), -1)
                else: # Đi lên
                    cv2.circle(frame, (vx - vw//2 + 8, vy - vh//2 + 5), 4, (0, 255, 255), -1)
                    cv2.circle(frame, (vx + vw//2 - 8, vy - vh//2 + 5), 4, (0, 255, 255), -1)

        out.write(frame)

    out.release()
    print(f"[SUCCESS] Da tao xong video mau: {output_path}")

if __name__ == "__main__":
    generate_traffic_video()
