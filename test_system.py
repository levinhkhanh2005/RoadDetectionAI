"""
Script kiem tra toan dien he thong (Verification Test)
1. Kiem tra tai model YOLOv8
2. Kiem tra nhan dien tren anh giao thong mau data/samples/traffic.jpg
3. Kiem tra thuat toan cat vach dem xe VehicleCounter
4. Xuat ket qua ra data/output/test_result.jpg
"""

import os
import cv2
from src.detector import VehicleDetector, VEHICLE_COLORS
from src.tracker import VehicleCounter
from src.visualizer import Visualizer


def run_tests():
    print("=== BẮT ĐẦU KIỂM THỬ HỆ THỐNG AI NHẬN DIỆN PHƯƠNG TIỆN ===")
    
    # 1. Kiem tra Detector & Tai weights
    print("\n[BƯỚC 1] Khởi tạo mô hình YOLOv8n...")
    detector = VehicleDetector(model_path="models/yolov8n.pt")
    assert detector.model is not None, "Model chua duoc nap thanh cong!"
    print("=> Tải model YOLOv8 thành công!")

    # 2. Kiem tra nhan dien tren anh
    image_path = "data/samples/traffic.jpg"
    print(f"\n[BƯỚC 2] Kiểm tra nhận diện trên ảnh: {image_path}...")
    assert os.path.exists(image_path), f"Khong tim thay file anh {image_path}!"
    img = cv2.imread(image_path)
    assert img is not None, "Khong the doc du lieu anh!"
    
    detections = detector.detect(img, conf_threshold=0.3)
    print(f"=> Số lượng phương tiện phát hiện được: {len(detections)}")
    assert len(detections) > 0, "Không phát hiện được phương tiện nào!"
    
    # Thống kê phân loại
    counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0}
    for d in detections:
        cls_en = d.get("class_en", "car")
        counts[cls_en] = counts.get(cls_en, 0) + 1
        print(f"   + Phát hiện: {d['class_vi']} ({d['class_en']}) - Conf: {d['confidence']*100:.1f}% - BBox: {d['bbox']}")

    print("=> Bảng tổng hợp theo loại xe:")
    for k, v in counts.items():
        if v > 0:
            print(f"     - {k}: {v}")

    # 3. Kiem tra thuat toan dem xe qua vach (VehicleCounter)
    print("\n[BƯỚC 3] Kiểm tra thuật toán cắt vạch đếm xe (Line-crossing)...")
    h, w = img.shape[:2]
    line_y = int(h * 0.6)
    counter = VehicleCounter(line_coords=((0, line_y), (w, line_y)))
    
    # Giả lập 2 frame di chuyển của 1 xe ô tô đi qua vạch (y đi từ line_y - 20 xuống line_y + 20)
    fake_track_frame1 = [{
        "bbox": [100, line_y - 30, 160, line_y - 10],
        "class_en": "car",
        "class_vi": "O to",
        "track_id": 1
    }]
    fake_track_frame2 = [{
        "bbox": [100, line_y + 10, 160, line_y + 30],
        "class_en": "car",
        "class_vi": "O to",
        "track_id": 1
    }]
    
    counter.update(fake_track_frame1)
    res_count = counter.update(fake_track_frame2)
    assert counter.counts["car"] == 1, f"Lỗi đếm xe: kết quả mong đợi 1, thực tế {counter.counts['car']}"
    assert counter.get_total_count() == 1, "Tổng số xe đếm được không khớp!"
    print(f"=> Kiểm thử đếm xe cắt vạch thành công! Tổng xe đếm được: {counter.get_total_count()}")

    # 4. Kiem tra Visualizer & Xuat file ket qua
    print("\n[BƯỚC 4] Vẽ Bounding Box, vạch đếm, Dashboard HUD và lưu ảnh...")
    visualizer = Visualizer()
    out_img = visualizer.draw_detections(img, detections, VEHICLE_COLORS)
    out_img = visualizer.draw_counting_line(out_img, counter.line_coords)
    out_img = visualizer.draw_dashboard_hud(out_img, counts, len(detections), fps=30.0)
    
    os.makedirs("data/output", exist_ok=True)
    out_path = "data/output/test_result.jpg"
    cv2.imwrite(out_path, out_img)
    print(f"=> Đã lưu ảnh kết quả trực quan hóa tại: {out_path}")

    # Đồng thời sao chép ra artifact để hiển thị trong báo cáo
    artifact_img_path = r"C:\Users\kle45\.gemini\antigravity-ide\brain\f407a859-09ef-4899-b3a8-30768dca56ac\test_result.jpg"
    cv2.imwrite(artifact_img_path, out_img)

    print("\n=== TOÀN BỘ CÁC BƯỚC KIỂM THỬ ĐÃ HOÀN THÀNH XUẤT SẮC! ===")


if __name__ == "__main__":
    run_tests()
