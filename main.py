"""
Chương trình: Road Vehicle Detection & Counting AI
Môn học: Trí Tuệ Nhân Tạo (Artificial Intelligence)

Hướng dẫn chạy:
    1. Chạy với Webcam:
        python main.py --source 0
    2. Chạy với Video:
        python main.py --source data/samples/traffic.mp4
    3. Chạy với Ảnh:
        python main.py --source data/samples/traffic.jpg
    4. Lưu video kết quả:
        python main.py --source data/samples/traffic.mp4 --save
"""

import os
import sys
import time
import argparse
import cv2

from src.detector import VehicleDetector, VEHICLE_COLORS
from src.tracker import VehicleCounter
from src.visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="AI Road Vehicle Detection & Counting")
    parser.add_argument("--source", type=str, default="0",
                        help="Nguon du lieu: '0' (Webcam), duong dan den file video (.mp4) hoac anh (.jpg, .png)")
    parser.add_argument("--model", type=str, default="models/yolov8n.pt",
                        help="Duong dan den file mo hinh YOLO (.pt)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Nguong tu tin confidence threshold (0.1 - 1.0)")
    parser.add_argument("--line-y", type=float, default=0.65,
                        help="Vi tri vach dem theo truc Y (ti le 0.0 - 1.0 tu tren xuong)")
    parser.add_argument("--save", action="store_true",
                        help="Luu video/anh ket qua vao thu muc data/output/")
    parser.add_argument("--no-show", action="store_true",
                        help="Khong hien thi cua so OpenCV (phu hop khi xu ly ngam hoac headless)")
    parser.add_argument("--device", type=str, default="",
                        help="Thiet bi tinh toan ('cpu', 'cuda', hoac de trong)")
    return parser.parse_args()


def process_image(detector, visualizer, image_path, conf_threshold, save=False, no_show=False):
    """Xu ly nhan dien tren 1 anh tinh."""
    print(f"[INFO] Dang doc anh: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Khong the doc anh tu duong dan: {image_path}")
        return

    start_time = time.time()
    detections = detector.detect(image, conf_threshold=conf_threshold)
    elapsed = time.time() - start_time

    # Thong ke so luong theo loai
    counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0}
    for d in detections:
        cls_en = d.get("class_en", "car")
        if cls_en in counts:
            counts[cls_en] += 1

    # Ve ket qua
    output_img = visualizer.draw_detections(image, detections, VEHICLE_COLORS)
    output_img = visualizer.draw_dashboard_hud(output_img, counts, len(detections), fps=1.0 / max(elapsed, 0.001))

    print("\n--- KET QUA NHAN DIEN ANH ---")
    print(f"- Thoi gian xu ly: {elapsed * 1000:.2f} ms")
    print(f"- Tong so phuong tien: {len(detections)}")
    for k, v in counts.items():
        if v > 0:
            print(f"  + {k.capitalize()}: {v}")

    if save:
        os.makedirs("data/output", exist_ok=True)
        out_path = os.path.join("data/output", "result_" + os.path.basename(image_path))
        cv2.imwrite(out_path, output_img)
        print(f"[INFO] Da luu ket qua tai: {out_path}")

    if not no_show:
        cv2.imshow("Road Vehicle Detection - AI Project", output_img)
        print("\n[INFO] Nhan phim bat ky tren cua so anh de thoat...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video_or_camera(args):
    """Xu ly video luong thoi gian thuc hoac camera."""
    # Xac dinh nguon video
    is_webcam = False
    source = args.source
    if source.isdigit():
        source = int(source)
        is_webcam = True

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Khong the mo nguon video: {args.source}")
        return

    # Lay thong so video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"[INFO] Da ket noi nguon video: {args.source} ({frame_width}x{frame_height} @ {fps_video:.1f} FPS)")

    # Khoi tao cac module
    detector = VehicleDetector(model_path=args.model, device=args.device)
    visualizer = Visualizer()
    
    # Thiet lap vach dem cat ngang duong (ti le theo line-y)
    line_y = int(frame_height * args.line_y)
    counting_line = ((0, line_y), (frame_width, line_y))
    counter = VehicleCounter(line_coords=counting_line)

    # Thiet lap ghi video neu bat --save
    writer = None
    if args.save:
        os.makedirs("data/output", exist_ok=True)
        out_filename = "output_detected.mp4" if not isinstance(source, str) else "output_" + os.path.basename(source)
        out_path = os.path.join("data/output", out_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_video, (frame_width, frame_height))
        print(f"[INFO] Dang ghi video ra: {out_path}")

    print("\n" + "=" * 50)
    print("CHUONG TRINH NHAN DIEN PHUONG TIEN GIAO THONG (AI)")
    print("Phim tat dieu khien:")
    print("  - 'q' hoac ESC: Thoat chuong trinh")
    print("  - 'p'         : Tam dung / Tiep tuc")
    print("  - 's'         : Chup anh luu lai snapshot")
    print("  - 'r'         : Reset bo dem xe ve 0")
    print("=" * 50 + "\n")

    paused = False
    prev_time = time.time()
    fps_calc = 0.0
    frame_count = 0

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Da ket thuc luong video.")
                break

            frame_count += 1

            # Tinh FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                fps_calc = 0.9 * fps_calc + 0.1 * (1.0 / dt) if fps_calc > 0 else 1.0 / dt
            prev_time = curr_time

            # 1. Nhan dien & Theo doi (Tracking)
            tracked_objects = detector.track(frame, conf_threshold=args.conf, persist=True)

            # 2. Cap nhat bo dem xe qua vach
            count_info = counter.update(tracked_objects)

            # 3. Truc quan hoa
            # - Ve bounding boxes & nhan xe
            disp_frame = visualizer.draw_detections(frame, tracked_objects, VEHICLE_COLORS)
            # - Ve vet quy dao di chuyen
            disp_frame = visualizer.draw_trajectories(disp_frame, counter.tracks_history)
            # - Ve vach dem xe
            disp_frame = visualizer.draw_counting_line(disp_frame, counter.line_coords)
            # - Ve bang thong ke HUD
            disp_frame = visualizer.draw_dashboard_hud(
                disp_frame,
                count_info["counts"],
                count_info["total"],
                fps=fps_calc,
                directions=count_info["directions"]
            )

            # Ghi video neu can
            if writer is not None:
                writer.write(disp_frame)

        # Hien thi cua so neu khong phai no-show
        if not args.no_show:
            cv2.imshow("Road Vehicle Detection & Counting (AI)", disp_frame)

            # Bat phim tat
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' hoac ESC
                break
            elif key == ord('p'):      # Tam dung / tiep tuc
                paused = not paused
                print("[INFO] Tinh trang video: " + ("Tam dung" if paused else "Tiep tuc"))
            elif key == ord('r'):      # Reset bo dem
                counter.reset()
                print("[INFO] Da reset bo dem xe ve 0.")
            elif key == ord('s'):      # Chup anh
                os.makedirs("data/output", exist_ok=True)
                snap_path = f"data/output/snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_path, disp_frame)
                print(f"[INFO] Da luu anh chup tai: {snap_path}")

    # Giai phong tai nguyen
    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_show:
        cv2.destroyAllWindows()
    print("\n--- TONG KET KET QUA ---")
    print(f"- Tong so phuong tien da dem: {counter.get_total_count()}")
    for k, v in counter.counts.items():
        print(f"  + {k.capitalize()}: {v}")
    print(f"- Luot vao (In): {counter.direction_counts.get('in', 0)} | Luot ra (Out): {counter.direction_counts.get('out', 0)}")


def main():
    args = parse_args()
    
    # Kiem tra neu source la file anh
    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if isinstance(args.source, str) and args.source.lower().endswith(img_exts):
        detector = VehicleDetector(model_path=args.model, device=args.device)
        visualizer = Visualizer()
        process_image(detector, visualizer, args.source, args.conf, save=args.save, no_show=args.no_show)
    else:
        process_video_or_camera(args)


if __name__ == "__main__":
    main()
