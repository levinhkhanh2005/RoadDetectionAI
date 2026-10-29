"""
Module: Visualizer
Chức năng: Trực quan hóa kết quả nhận diện, vẽ Bounding Box, vẽ vạch đếm ảo,
quỹ đạo di chuyển (Trajectory trails) và hiển thị bảng thống kê HUD (Heads-Up Display) hiện đại.
"""

from typing import List, Dict, Tuple, Any, Optional
import cv2
import numpy as np


class Visualizer:
    """
    Module hỗ trợ vẽ các thành phần đồ họa trực quan lên khung hình video/ảnh.
    """

    def __init__(self, font_scale: float = 0.55, thickness: int = 2):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness

    def draw_rounded_rectangle(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                               color: Tuple[int, int, int], thickness: int = 2, radius: int = 8):
        """Vẽ hình chữ nhật bo góc viền hiện đại."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Bounding box thông thường với viền mượt mà
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Thêm 4 góc nhấn mạnh (Corner brackets) để tạo phong cách AI cao cấp
        corner_len = min(15, int((x2 - x1) / 4), int((y2 - y1) / 4))
        # Góc trên-trái
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
        # Góc trên-phải
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
        # Góc dưới-trái
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
        # Góc dưới-phải
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)

    def draw_label(self, img: np.ndarray, text: str, pt: Tuple[int, int],
                   bg_color: Tuple[int, int, int], text_color: Tuple[int, int, int] = (255, 255, 255)):
        """Vẽ thẻ nhãn (badge) có nền màu nổi bật."""
        x, y = pt
        (tw, th), baseline = cv2.getTextSize(text, self.font, self.font_scale, 1)
        
        # Tọa độ nền nhãn
        pad = 4
        y_top = max(0, y - th - pad * 2)
        y_bottom = y
        x_right = min(img.shape[1], x + tw + pad * 2)

        # Vẽ nền nhãn
        cv2.rectangle(img, (x, y_top), (x_right, y_bottom), bg_color, -1)
        # Viền nhãn
        cv2.rectangle(img, (x, y_top), (x_right, y_bottom), (0, 0, 0), 1)
        # Vẽ chữ
        cv2.putText(img, text, (x + pad, y_bottom - pad), self.font, self.font_scale, text_color, 1, cv2.LINE_AA)

    def draw_detections(self, frame: np.ndarray, objects: List[Dict[str, Any]],
                        color_map: Dict[str, Tuple[int, int, int]],
                        speed_warning_threshold: float = 50.0) -> np.ndarray:
        """
        Vẽ Bounding Box, nhãn phân loại và ID đối tượng lên khung hình.
        """
        out = frame.copy()
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            cls_en = obj.get("class_en", "car")
            cls_vi = obj.get("class_vi", "Xe")
            conf = obj.get("confidence", 0.0)
            track_id = obj.get("track_id")
            speed_kmh = float(obj.get("speed_kmh", 0.0) or 0.0)

            color = color_map.get(cls_en, (0, 255, 0))
            if speed_kmh >= speed_warning_threshold:
                color = (0, 0, 255)

            # Bounding box
            self.draw_rounded_rectangle(out, (x1, y1), (x2, y2), color, thickness=self.thickness)

            # Chuẩn bị nội dung nhãn
            if track_id is not None:
                label = f"#{track_id} {cls_vi} {conf:.2f} {speed_kmh:.1f}km/h"
            else:
                label = f"{cls_vi} {conf:.2f}"

            self.draw_label(out, label, (x1, y1), color)

            # Vẽ tâm đối tượng
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(out, (cx, cy), 4, color, -1)

        return out

    def draw_trajectories(self, frame: np.ndarray, tracks_history: Dict[int, List[Tuple[int, int]]],
                          color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """Vẽ vệt quỹ đạo di chuyển của các xe theo thời gian."""
        out = frame  # intentional: mutates in-place for perf, called after draw_detections copies
        for track_id, points in tracks_history.items():
            if len(points) < 2:
                continue
            for i in range(1, len(points)):
                # Độ đậm nhạt tăng dần theo thời gian gần nhất
                alpha = i / len(points)
                thick = max(1, int(alpha * 3))
                cv2.line(out, points[i - 1], points[i], color, thick)
        return out

    def draw_counting_line(self, frame: np.ndarray, line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
                           color: Tuple[int, int, int] = (0, 0, 255), is_active: bool = False) -> np.ndarray:
        """Vẽ vạch đếm ảo qua đường."""
        if line_coords is None:
            return frame
        out = frame  # intentional: mutates in-place for perf, called after draw_detections copies
        pt1, pt2 = line_coords
        
        # Vẽ hiệu ứng vạch đếm sáng
        cv2.line(out, pt1, pt2, color, 3)
        cv2.circle(out, pt1, 6, (0, 255, 255), -1)
        cv2.circle(out, pt2, 6, (0, 255, 255), -1)
        
        # Chữ ghi chú vạch
        mid_x = int((pt1[0] + pt2[0]) / 2)
        mid_y = int((pt1[1] + pt2[1]) / 2) - 10
        cv2.putText(out, "VACH DEM XE (COUNTING LINE)", (mid_x - 120, mid_y),
                    self.font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return out

    def draw_dashboard_hud(self, frame: np.ndarray, counts: Dict[str, int], total_count: int,
                           fps: float = 0.0, directions: Optional[Dict[str, int]] = None,
                           speed_summary: Optional[Dict[str, float]] = None,
                           speed_warning_threshold: float = 50.0) -> np.ndarray:
        """
        Vẽ bảng điều khiển thống kê HUD (Heads-Up Display) bán trong suốt ở góc trên bên trái.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # Kích thước khung HUD
        hud_w = 260
        hud_h = 250
        hud_x = 20
        hud_y = 20

        # Tạo lớp overlay mờ (Glassmorphism dark theme)
        sub_img = out[hud_y:hud_y + hud_h, hud_x:hud_x + hud_w]
        dark_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        # Pha trộn 70% đen, 30% nền
        res = cv2.addWeighted(sub_img, 0.25, dark_rect, 0.75, 1.0)
        out[hud_y:hud_y + hud_h, hud_x:hud_x + hud_w] = res

        # Viền neon sang trọng cho HUD
        cv2.rectangle(out, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 255, 200), 1)

        # Tiêu đề HUD
        cv2.putText(out, "AI TRAFFIC MONITOR", (hud_x + 12, hud_y + 24),
                    self.font, 0.55, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.line(out, (hud_x + 10, hud_y + 32), (hud_x + hud_w - 10, hud_y + 32), (100, 100, 100), 1)

        # FPS
        fps_color = (0, 255, 0) if fps >= 20 else (0, 165, 255)
        cv2.putText(out, f"FPS: {fps:.1f}", (hud_x + 180, hud_y + 24),
                    self.font, 0.45, fps_color, 1, cv2.LINE_AA)

        # Danh sách số lượng phương tiện theo loại
        label_names = [
            ("O to (Car):", counts.get("car", 0), (0, 255, 128)),
            ("Xe may (Motorbike):", counts.get("motorcycle", 0), (255, 165, 0)),
            ("Xe buyt (Bus):", counts.get("bus", 0), (0, 191, 255)),
            ("Xe tai (Truck):", counts.get("truck", 0), (255, 50, 100)),
            ("Xe dap (Bicycle):", counts.get("bicycle", 0), (255, 255, 0))
        ]

        curr_y = hud_y + 52
        for title, val, color in label_names:
            # Chấm màu phân loại
            cv2.circle(out, (hud_x + 18, curr_y - 4), 4, color, -1)
            # Tên phương tiện
            cv2.putText(out, title, (hud_x + 28, curr_y), self.font, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
            # Số lượng
            cv2.putText(out, str(val), (hud_x + hud_w - 40, curr_y), self.font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            curr_y += 22

        cv2.line(out, (hud_x + 10, curr_y - 8), (hud_x + hud_w - 10, curr_y - 8), (100, 100, 100), 1)

        # Tổng số xe đếm được
        cv2.putText(out, "TONG SO XE:", (hud_x + 15, curr_y + 14),
                    self.font, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"{total_count:03d}", (hud_x + hud_w - 55, curr_y + 14),
                    self.font, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        # Nếu có thống kê hướng di chuyển
        if directions:
            in_count = directions.get("in", 0)
            out_count = directions.get("out", 0)
            dir_text = f"Vao: {in_count} | Ra: {out_count}"
            cv2.putText(out, dir_text, (hud_x + 15, curr_y + 35),
                        self.font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        if speed_summary:
            avg_speed = speed_summary.get("average_kmh", 0.0)
            max_speed = speed_summary.get("max_kmh", 0.0)
            warning_count = int(speed_summary.get("warning_count", 0))
            speed_y = curr_y + 58
            speed_color = (0, 0, 255) if warning_count else (180, 255, 180)
            cv2.putText(out, f"Speed avg/max: {avg_speed:.1f}/{max_speed:.1f} km/h",
                        (hud_x + 15, speed_y), self.font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(out, f"WARNING > {speed_warning_threshold:.0f}: {warning_count}",
                        (hud_x + 15, speed_y + 20), self.font, 0.4, speed_color, 2, cv2.LINE_AA)

        return out

    def draw_heatmap(self, frame: np.ndarray, tracks_history: Dict[int, List[Tuple[int, int]]],
                     alpha: float = 0.45) -> np.ndarray:
        """Vẽ bản đồ nhiệt mật độ giao thông (Traffic Density Heatmap) từ lịch sử di chuyển."""
        h, w = frame.shape[:2]
        heat_mask = np.zeros((h, w), dtype=np.float32)
        has_points = False

        for points in tracks_history.values():
            for pt in points:
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(heat_mask, pt, radius=18, color=1.0, thickness=-1)
                    has_points = True

        if not has_points:
            return frame

        cv2.normalize(heat_mask, heat_mask, 0, 255, cv2.NORM_MINMAX)
        heat_mask = cv2.GaussianBlur(heat_mask.astype(np.uint8), (41, 41), 0)
        color_map = cv2.applyColorMap(heat_mask, cv2.COLORMAP_JET)

        mask_threshold = heat_mask > 15
        out = frame.copy()
        out[mask_threshold] = cv2.addWeighted(frame, 1.0 - alpha, color_map, alpha, 0)[mask_threshold]
        return out

    def draw_polygon_roi(self, frame: np.ndarray, polygon_coords: Optional[List[Tuple[int, int]]],
                         color: Tuple[int, int, int] = (255, 0, 255), label: str = "VUNG ROI") -> np.ndarray:
        """Vẽ vùng đa giác kiểm soát (Polygon ROI) với hiệu ứng bán trong suốt."""
        if polygon_coords is None or len(polygon_coords) < 3:
            return frame
        out = frame.copy()
        pts = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        for pt in polygon_coords:
            cv2.circle(out, pt, 4, (0, 255, 255), -1)
        first_pt = polygon_coords[0]
        cv2.putText(out, label, (first_pt[0], max(15, first_pt[1] - 8)),
                    self.font, 0.45, color, 1, cv2.LINE_AA)
        return out
