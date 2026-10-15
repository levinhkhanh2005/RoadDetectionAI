"""
Script tai va chuan bi video mau giao thong thuc te (Real Traffic Sample Video Downloader)
Cung cap cac video giam sat CCTV duong pho thuc te de kiem thu YOLOv8, ByteTrack va Vach dem xe.
"""

import os
import urllib.request
import cv2

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

HIGHWAY_URL = "https://raw.githubusercontent.com/AarohiSingla/Speed-detection-of-vehicles/main/highway.mp4"
MOTORCYCLE_URL = "https://raw.githubusercontent.com/agung-madani/yolov8-motorcycle-counting/main/gerbanguin.mp4"


def download_file(url, target_path, label):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 100000:
        print(f"[INFO] {label} da ton tai ({target_path})")
        return True
    print(f"[INFO] Dang tai {label} tu {url}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"[SUCCESS] Da tai {label}: {target_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Loi tai {label}: {e}")
        return False


def setup_real_samples():
    print("=" * 60)
    print("CHUAN BI VIDEO GIAO THONG THUC TE (REAL TRAFFIC DATASETS)")
    print("=" * 60)

    # 1. Tai video cao toc Full HD
    highway_path = os.path.join(SAMPLE_DIR, "highway.mp4")
    download_file(HIGHWAY_URL, highway_path, "Video Cao Toc (highway.mp4)")

    # 2. Tai video xe may do thi
    moto_path = os.path.join(SAMPLE_DIR, "motorcycles.mp4")
    download_file(MOTORCYCLE_URL, moto_path, "Video Xe May Do Thi (motorcycles.mp4)")

    # 3. Tao ban traffic.mp4 (720p, 16s) toi uu hoa cho CPU & Streamlit tu highway.mp4
    traffic_path = os.path.join(SAMPLE_DIR, "traffic.mp4")
    if not os.path.exists(traffic_path) or os.path.getsize(traffic_path) < 1000000:
        if os.path.exists(highway_path):
            print("[INFO] Dang tao ban toi uu traffic.mp4 (720p @ 25 FPS, 16s)...")
            cap = cv2.VideoCapture(highway_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            target_w, target_h = 1280, 720
            max_frames = int(fps * 16)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(traffic_path, fourcc, fps, (target_w, target_h))

            count = 0
            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                resized = cv2.resize(frame, (target_w, target_h))
                out.write(resized)
                count += 1

            cap.release()
            out.release()
            print(f"[SUCCESS] Da tao xong traffic.mp4 ({count} frames)")

    print("\n[HOAN TAT] Danh sach video mau thuc te tai data/samples/:")
    for f in os.listdir(SAMPLE_DIR):
        if f.endswith((".mp4", ".jpg", ".png")):
            sz_mb = os.path.getsize(os.path.join(SAMPLE_DIR, f)) / (1024 * 1024)
            print(f" - {f}: {sz_mb:.2f} MB")


if __name__ == "__main__":
    setup_real_samples()
