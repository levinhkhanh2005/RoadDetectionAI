"""
Module: VehicleCounter
Chức năng: Quản lý quỹ đạo di chuyển (Trajectory) của phương tiện,
xử lý thuật toán cắt vạch ảo (Line Crossing) để đếm số lượng xe theo loại và hướng di chuyển.
"""

from typing import List, Dict, Tuple, Set, Any, Optional
from collections import defaultdict
import math
import time
import csv
import os
import cv2
import numpy as np


def ccw(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int]) -> bool:
    """Kiểm tra hướng quay của 3 điểm (Counter-Clockwise)."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int], D: Tuple[int, int]) -> bool:
    """
    Kiểm tra xem 2 đoạn thẳng AB và CD có cắt nhau hay không.
    AB: Quỹ đạo di chuyển của tâm phương tiện từ frame trước tới frame này.
    CD: Vạch đếm ảo trên đường.
    """
    def orientation(P, Q, R):
        value = (Q[0] - P[0]) * (R[1] - P[1]) - (Q[1] - P[1]) * (R[0] - P[0])
        return (value > 0) - (value < 0)

    def on_segment(P, Q, R):
        return (
            min(P[0], R[0]) <= Q[0] <= max(P[0], R[0])
            and min(P[1], R[1]) <= Q[1] <= max(P[1], R[1])
        )

    o1 = orientation(A, B, C)
    o2 = orientation(A, B, D)
    o3 = orientation(C, D, A)
    o4 = orientation(C, D, B)
    if o1 == 0 and on_segment(A, C, B):
        return True
    if o2 == 0 and on_segment(A, D, B):
        return True
    if o3 == 0 and on_segment(C, A, D):
        return True
    if o4 == 0 and on_segment(C, B, D):
        return True
    return o1 != o2 and o3 != o4


class VehicleCounter:
    """
    Bộ đếm phương tiện thông minh qua vạch kiểm soát ảo.
    """

    def __init__(self, line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                 polygon_coords: Optional[List[Tuple[int, int]]] = None,
                 pixels_per_meter: float = 10.0):
        """
        Khởi tạo bộ đếm.
        :param line_coords: Tọa độ vạch đếm ((x1, y1), (x2, y2)).
        :param polygon_coords: Tọa độ đa giác vùng kiểm soát [(x, y), ...].
        """
        self.line_coords = line_coords  # ((x1, y1), (x2, y2))
        self.polygon_coords = polygon_coords  # [(x, y), ...]
        self.pixels_per_meter = max(float(pixels_per_meter), 1e-6)
        
        # Lưu vết tâm của các đối tượng: {track_id: [(cx, cy), ...]}
        self.tracks_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.track_timestamps: Dict[int, List[float]] = defaultdict(list)
        self.speed_history: Dict[int, List[float]] = defaultdict(list)
        self.current_speeds: Dict[int, float] = {}
        
        # Danh sách các ID đã đếm qua vạch để tránh đếm lặp
        self.counted_ids: Set[int] = set()
        
        # Thống kê tổng số lượng xe theo từng loại
        self.counts: Dict[str, int] = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
            "bicycle": 0
        }
        
        # Thống kê theo hướng: 'in' (vào), 'out' (ra)
        self.direction_counts: Dict[str, int] = {"in": 0, "out": 0}
        
        # Lịch sử sự kiện gần nhất để hiển thị thông báo
        self.recent_events: List[Dict[str, Any]] = []
        # Toàn bộ lịch sử các sự kiện cắt vạch để xuất báo cáo CSV
        self.events_log: List[Dict[str, Any]] = []

    def set_line(self, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        """Cập nhật tọa độ vạch kiểm soát ảo."""
        self.line_coords = (pt1, pt2)

    def set_polygon(self, pts: List[Tuple[int, int]]):
        """Cập nhật tọa độ vùng đa giác kiểm soát ROI."""
        self.polygon_coords = pts

    def reset(self):
        """Khởi tạo lại toàn bộ bộ đếm và lịch sử."""
        self.tracks_history.clear()
        self.track_timestamps.clear()
        self.speed_history.clear()
        self.current_speeds.clear()
        self.counted_ids.clear()
        for k in self.counts:
            self.counts[k] = 0
        self.direction_counts = {"in": 0, "out": 0}
        self.recent_events.clear()
        self.events_log.clear()

    def get_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Tính điểm tâm đáy của Bounding Box (phù hợp với vị trí bánh xe tiếp đất)."""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = y2  # tâm đáy: mép dưới bbox
        return (cx, cy)

    def determine_direction(self, pt_prev: Tuple[int, int], pt_curr: Tuple[int, int]) -> str:
        """
        Xác định hướng di chuyển so với vạch kẻ.
        Mặc định: dy > 0 là đi từ trên xuống dưới ('in'), ngược lại là đi lên ('out').
        """
        if self.line_coords is None:
            return "unknown"
            
        (x1, y1), (x2, y2) = self.line_coords
        line_dx = x2 - x1
        line_dy = y2 - y1
        
        move_dx = pt_curr[0] - pt_prev[0]
        move_dy = pt_curr[1] - pt_prev[1]
        
        # Tích có hướng vector để xác định chiều cắt vạch
        cross_prod = line_dx * move_dy - line_dy * move_dx
        return "in" if cross_prod > 0 else "out"

    def update(self, tracked_objects: List[Dict[str, Any]], timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Cập nhật quỹ đạo và kiểm tra xe vượt qua vạch đếm.
        
        :param tracked_objects: Danh sách các đối tượng từ VehicleDetector.track().
        :return: Thông tin tổng hợp số đếm và các sự kiện mới.
        """
        current_frame_counts = {"in": 0, "out": 0}
        new_crossed_events = []
        frame_timestamp = time.monotonic() if timestamp is None else float(timestamp)

        current_ids = set()

        for obj in tracked_objects:
            track_id = obj.get("track_id")
            if track_id is None:
                continue

            current_ids.add(track_id)
            bbox = obj["bbox"]
            cls_en = obj.get("class_en", "car")
            center = self.get_center(bbox)
            object_timestamp = float(obj.get("timestamp", frame_timestamp))
            previous_points = self.tracks_history[track_id]
            previous_timestamps = self.track_timestamps[track_id]
            speed_kmh = 0.0
            if previous_points and previous_timestamps:
                elapsed = object_timestamp - previous_timestamps[-1]
                if elapsed > 0:
                    distance_pixels = math.hypot(
                        center[0] - previous_points[-1][0],
                        center[1] - previous_points[-1][1],
                    )
                    raw_speed = distance_pixels / self.pixels_per_meter / elapsed * 3.6
                    self.speed_history[track_id].append(raw_speed)
                    self.speed_history[track_id] = self.speed_history[track_id][-5:]
                    speed_kmh = sum(self.speed_history[track_id]) / len(self.speed_history[track_id])
            self.current_speeds[track_id] = speed_kmh
            obj["speed_kmh"] = round(speed_kmh, 1)

            # Lưu vào lịch sử tâm
            self.tracks_history[track_id].append(center)
            self.track_timestamps[track_id].append(object_timestamp)
            # Giữ tối đa 30 điểm lịch sử gần nhất để tiết kiệm RAM
            if len(self.tracks_history[track_id]) > 30:
                self.tracks_history[track_id].pop(0)
                self.track_timestamps[track_id].pop(0)

            # Kiểm tra cắt vạch hoặc đi vào vùng đa giác ROI
            has_prev = len(self.tracks_history[track_id]) >= 2
            pt_prev = self.tracks_history[track_id][-2] if has_prev else center
            pt_curr = center

            crossed_line = False
            if self.line_coords is not None and has_prev:
                (line_a, line_b) = self.line_coords
                crossed_line = intersect(pt_prev, pt_curr, line_a, line_b)

            entered_polygon = False
            if self.polygon_coords is not None and len(self.polygon_coords) >= 3:
                poly_np = np.array(self.polygon_coords, dtype=np.int32)
                entered_polygon = cv2.pointPolygonTest(poly_np, (float(center[0]), float(center[1])), False) >= 0

            # Nếu cắt vạch hoặc vào ROI và chưa từng được đếm
            if track_id not in self.counted_ids and (crossed_line or entered_polygon):
                self.counted_ids.add(track_id)
                direction = self.determine_direction(pt_prev, pt_curr) if crossed_line else "roi"

                # Tăng biến đếm
                if cls_en in self.counts:
                    self.counts[cls_en] += 1
                else:
                    self.counts[cls_en] = 1

                self.direction_counts[direction] = self.direction_counts.get(direction, 0) + 1
                current_frame_counts[direction] = current_frame_counts.get(direction, 0) + 1

                event_info = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "frame_timestamp": object_timestamp,
                    "track_id": track_id,
                    "class_en": cls_en,
                    "class_vi": obj.get("class_vi", "Xe"),
                    "direction": direction,
                    "point": center,
                    "speed_kmh": round(speed_kmh, 1),
                    "speed_avg_kmh": round(self.get_track_speed_average(track_id), 1),
                    "speed_max_kmh": round(self.get_track_speed_max(track_id), 1)
                }
                new_crossed_events.append(event_info)
                self.events_log.append(event_info)
                self.recent_events.append(event_info)
                if len(self.recent_events) > 10:
                    self.recent_events.pop(0)

        # Xóa bớt lịch sử của những track_id không còn xuất hiện trong hơn 50 frame
        for sid in set(self.track_timestamps) - current_ids:
            self.track_timestamps.pop(sid, None)
            self.speed_history.pop(sid, None)
            self.current_speeds.pop(sid, None)
        if len(self.tracks_history) > 200:
            active_ids = set(self.tracks_history.keys())
            stale_ids = active_ids - current_ids
            for sid in list(stale_ids)[:50]:
                del self.tracks_history[sid]
                self.track_timestamps.pop(sid, None)
                self.speed_history.pop(sid, None)
                self.current_speeds.pop(sid, None)

        total_vehicles = sum(self.counts.values())
        return {
            "counts": self.counts.copy(),
            "total": total_vehicles,
            "directions": self.direction_counts.copy(),
            "new_events": new_crossed_events
        }

    def get_total_count(self) -> int:
        """Tổng số phương tiện đã đếm qua vạch."""
        return sum(self.counts.values())

    def get_track_speed_average(self, track_id: int) -> float:
        speeds = self.speed_history.get(track_id, [])
        return sum(speeds) / len(speeds) if speeds else 0.0

    def get_track_speed_max(self, track_id: int) -> float:
        return max(self.speed_history.get(track_id, []), default=0.0)

    def get_speed_summary(self, warning_threshold: float = 50.0) -> Dict[str, float]:
        speeds = list(self.current_speeds.values())
        return {
            "average_kmh": sum(speeds) / len(speeds) if speeds else 0.0,
            "max_kmh": max(speeds, default=0.0),
            "warning_count": sum(1 for speed in speeds if speed >= warning_threshold),
        }

    def export_csv(self, filepath: str) -> bool:
        """Xuất danh sách tất cả các lượt xe qua vạch ra file CSV."""
        if not self.events_log:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "track_id", "class_en", "class_vi", "direction", "x", "y", "speed_kmh", "speed_avg_kmh", "speed_max_kmh"])
            for e in self.events_log:
                writer.writerow([e["timestamp"], e["track_id"], e["class_en"], e["class_vi"], e["direction"], e["point"][0], e["point"][1], e.get("speed_kmh", 0.0), e.get("speed_avg_kmh", 0.0), e.get("speed_max_kmh", 0.0)])
        return True
