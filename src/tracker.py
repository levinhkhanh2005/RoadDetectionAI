"""
Module: VehicleCounter
Chức năng: Quản lý quỹ đạo di chuyển (Trajectory) của phương tiện,
xử lý thuật toán cắt vạch ảo (Line Crossing) để đếm số lượng xe theo loại và hướng di chuyển.
"""

from typing import List, Dict, Tuple, Set, Any, Optional
from collections import defaultdict
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
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class VehicleCounter:
    """
    Bộ đếm phương tiện thông minh qua vạch kiểm soát ảo.
    """

    def __init__(self, line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None):
        """
        Khởi tạo bộ đếm.
        :param line_coords: Tọa độ vạch đếm ((x1, y1), (x2, y2)).
        """
        self.line_coords = line_coords  # ((x1, y1), (x2, y2))
        
        # Lưu vết tâm của các đối tượng: {track_id: [(cx, cy), ...]}
        self.tracks_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        
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

    def set_line(self, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        """Cập nhật tọa độ vạch kiểm soát ảo."""
        self.line_coords = (pt1, pt2)

    def reset(self):
        """Khởi tạo lại toàn bộ bộ đếm và lịch sử."""
        self.tracks_history.clear()
        self.counted_ids.clear()
        for k in self.counts:
            self.counts[k] = 0
        self.direction_counts = {"in": 0, "out": 0}
        self.recent_events.clear()

    def get_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Tính điểm tâm đáy của Bounding Box (phù hợp với vị trí bánh xe tiếp đất)."""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
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

    def update(self, tracked_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Cập nhật quỹ đạo và kiểm tra xe vượt qua vạch đếm.
        
        :param tracked_objects: Danh sách các đối tượng từ VehicleDetector.track().
        :return: Thông tin tổng hợp số đếm và các sự kiện mới.
        """
        current_frame_counts = {"in": 0, "out": 0}
        new_crossed_events = []

        current_ids = set()

        for obj in tracked_objects:
            track_id = obj.get("track_id")
            if track_id is None:
                continue

            current_ids.add(track_id)
            bbox = obj["bbox"]
            cls_en = obj.get("class_en", "car")
            center = self.get_center(bbox)

            # Lưu vào lịch sử tâm
            self.tracks_history[track_id].append(center)
            # Giữ tối đa 30 điểm lịch sử gần nhất để tiết kiệm RAM
            if len(self.tracks_history[track_id]) > 30:
                self.tracks_history[track_id].pop(0)

            # Kiểm tra cắt vạch nếu có vạch đếm và có ít nhất 2 điểm lịch sử
            if self.line_coords is not None and len(self.tracks_history[track_id]) >= 2:
                pt_prev = self.tracks_history[track_id][-2]
                pt_curr = self.tracks_history[track_id][-1]
                (line_a, line_b) = self.line_coords

                # Nếu cắt vạch và chưa từng được đếm
                if track_id not in self.counted_ids and intersect(pt_prev, pt_curr, line_a, line_b):
                    self.counted_ids.add(track_id)
                    direction = self.determine_direction(pt_prev, pt_curr)

                    # Tăng biến đếm
                    if cls_en in self.counts:
                        self.counts[cls_en] += 1
                    else:
                        self.counts[cls_en] = 1

                    self.direction_counts[direction] = self.direction_counts.get(direction, 0) + 1
                    current_frame_counts[direction] += 1

                    event_info = {
                        "track_id": track_id,
                        "class_en": cls_en,
                        "class_vi": obj.get("class_vi", "Xe"),
                        "direction": direction,
                        "point": center
                    }
                    new_crossed_events.append(event_info)
                    self.recent_events.append(event_info)
                    if len(self.recent_events) > 10:
                        self.recent_events.pop(0)

        # Xóa bớt lịch sử của những track_id không còn xuất hiện trong hơn 50 frame
        if len(self.tracks_history) > 200:
            active_ids = set(self.tracks_history.keys())
            stale_ids = active_ids - current_ids
            for sid in list(stale_ids)[:50]:
                del self.tracks_history[sid]

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
