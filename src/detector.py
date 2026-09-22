"""
Module: VehicleDetector
Chức năng: Tải mô hình YOLOv8, phát hiện và phân loại các phương tiện giao thông đường bộ.
"""

from typing import List, Dict, Any, Optional
import os
import cv2
import numpy as np
from ultralytics import YOLO


# Từ điển dịch tên class từ tiếng Anh sang tiếng Việt
CLASS_NAMES_VI = {
    "bicycle": "Xe dap",
    "car": "O to",
    "motorcycle": "Xe may",
    "bus": "Xe buyt",
    "truck": "Xe tai"
}

# Bảng màu đại diện cho từng loại xe (BGR format cho OpenCV)
VEHICLE_COLORS = {
    "car": (0, 255, 128),         # Xanh lục sáng
    "motorcycle": (255, 165, 0),   # Cam
    "bus": (0, 191, 255),          # Xanh dương sáng (Deep Sky Blue)
    "truck": (255, 50, 100),       # Đỏ hồng
    "bicycle": (255, 255, 0)       # Vàng chanh
}


class VehicleDetector:
    """
    Bộ nhận diện phương tiện giao thông sử dụng kiến trúc mạng YOLOv8.
    """

    def __init__(self, model_path: str = "models/yolov8n.pt", device: str = ""):
        """
        Khởi tạo detector.
        :param model_path: Đường dẫn tới file weights của YOLO (.pt).
        :param device: Thiết bị tính toán ('cpu', 'cuda', hoặc '' để tự động chọn).
        """
        self.model_path = model_path
        self.device = device
        
        # Đảm bảo thư mục models tồn tại
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        
        # Nạp mô hình YOLOv8
        print(f"[INFO] Dang tai mo hinh YOLO tu: {model_path}...")
        self.model = YOLO(model_path)
        print("[INFO] Tai mo hinh thanh cong!")

        # Đọc động các class name từ chính mô hình (Hỗ trợ cả COCO và Custom model)
        self.target_class_ids = []
        valid_vehicle_classes = set(CLASS_NAMES_VI.keys())
        for cls_id, cls_name in self.model.names.items():
            if cls_name.lower() in valid_vehicle_classes:
                self.target_class_ids.append(cls_id)

    def get_vehicle_name(self, class_id: int, lang: str = "vi") -> str:
        """Lấy tên phương tiện theo ngôn ngữ ('vi' hoặc 'en')."""
        name_en = self.model.names.get(class_id, "vehicle")
        if lang == "en":
            return name_en
        return CLASS_NAMES_VI.get(name_en.lower(), "Phuong tien")

    def get_vehicle_color(self, class_name_en: str) -> tuple:
        """Lấy màu sắc trực quan hóa cho loại phương tiện."""
        return VEHICLE_COLORS.get(class_name_en, (200, 200, 200))

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.35, iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
        """
        Nhận diện phương tiện trên 1 ảnh tĩnh hoặc 1 frame video.
        
        :param frame: Khung hình dạng numpy array (BGR).
        :param conf_threshold: Ngưỡng tự tin tối thiểu (0.0 - 1.0).
        :param iou_threshold: Ngưỡng NMS IoU.
        :return: Danh sách các phương tiện được phát hiện.
        """
        results = self.model.predict(
            source=frame,
            classes=self.target_class_ids,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device if self.device else None,
            verbose=False
        )

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                name_en = self.model.names.get(cls_id, "vehicle").lower()
                name_vi = CLASS_NAMES_VI.get(name_en, "Phuong tien")

                detections.append({
                    "bbox": xyxy.tolist(),  # [x1, y1, x2, y2]
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_en": name_en,
                    "class_vi": name_vi,
                    "track_id": None
                })

        return detections

    def track(self, frame: np.ndarray, conf_threshold: float = 0.35, iou_threshold: float = 0.45,
              persist: bool = True, tracker: str = "bytetrack.yaml") -> List[Dict[str, Any]]:
        """
        Theo dõi phương tiện liên tục qua các frame video (Object Tracking).
        Sử dụng thuật toán ByteTrack tích hợp để gán ID duy nhất cho mỗi phương tiện.
        
        :param frame: Khung hình hiện tại.
        :param conf_threshold: Ngưỡng tự tin.
        :param iou_threshold: Ngưỡng IoU.
        :param persist: Giữ lại track history giữa các frame.
        :param tracker: Cấu hình tracker ('bytetrack.yaml' hoặc 'botsort.yaml').
        :return: Danh sách đối tượng kèm ID theo dõi (track_id).
        """
        results = self.model.track(
            source=frame,
            classes=self.target_class_ids,
            conf=conf_threshold,
            iou=iou_threshold,
            persist=persist,
            tracker=tracker,
            device=self.device if self.device else None,
            verbose=False
        )

        tracked_objects = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Lấy track_id nếu có
                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())

                name_en = self.model.names.get(cls_id, "vehicle").lower()
                name_vi = CLASS_NAMES_VI.get(name_en, "Phuong tien")

                tracked_objects.append({
                    "bbox": xyxy.tolist(),
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_en": name_en,
                    "class_vi": name_vi,
                    "track_id": track_id
                })

        return tracked_objects
