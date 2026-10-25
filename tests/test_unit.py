"""
Unit tests cho hệ thống RoadDetectionAI (sử dụng thư viện chuẩn unittest).
Không yêu cầu GPU hay tải mô hình YOLO nặng, chạy siêu nhanh (<0.2s).
Chạy lệnh:
    python -m unittest discover tests
"""

import os
import unittest
import numpy as np

from src.tracker import VehicleCounter, intersect, ccw
from src.detector import VEHICLE_COLORS, CLASS_NAMES_VI
from src.visualizer import Visualizer


class TestGeometryAndTracker(unittest.TestCase):
    def test_ccw(self):
        # 3 điểm ngược chiều kim đồng hồ
        self.assertTrue(ccw((0, 0), (1, 0), (1, 1)))
        # 3 điểm cùng chiều kim đồng hồ
        self.assertFalse(ccw((0, 0), (1, 1), (1, 0)))

    def test_intersect(self):
        # 2 đoạn thẳng vuông góc cắt nhau tại (5, 5)
        ab = ((0, 5), (10, 5))
        cd = ((5, 0), (5, 10))
        self.assertTrue(intersect(ab[0], ab[1], cd[0], cd[1]))

        # 2 đoạn thẳng song song không cắt nhau
        cd_parallel = ((0, 8), (10, 8))
        self.assertFalse(intersect(ab[0], ab[1], cd_parallel[0], cd_parallel[1]))

    def test_get_center_is_bottom_center(self):
        counter = VehicleCounter()
        bbox = [100, 200, 300, 400]
        cx, cy = counter.get_center(bbox)
        self.assertEqual(cx, 200)
        self.assertEqual(cy, 400)  # Tâm đáy tiếp đất (y2)

    def test_line_crossing_counter(self):
        line = ((0, 300), (600, 300))
        counter = VehicleCounter(line_coords=line)

        # Frame 1: xe ở phía trên vạch (y = 280)
        frame1 = [{
            "track_id": 10,
            "class_en": "car",
            "class_vi": "O to",
            "bbox": [150, 240, 210, 280]
        }]
        counter.update(frame1)
        self.assertEqual(counter.get_total_count(), 0)

        # Frame 2: xe di chuyển cắt qua vạch xuống dưới (y = 320)
        frame2 = [{
            "track_id": 10,
            "class_en": "car",
            "class_vi": "O to",
            "bbox": [150, 280, 210, 320]
        }]
        info = counter.update(frame2)
        self.assertEqual(counter.get_total_count(), 1)
        self.assertEqual(counter.counts["car"], 1)
        self.assertEqual(counter.direction_counts["in"], 1)
        self.assertEqual(len(counter.events_log), 1)
        self.assertEqual(counter.events_log[0]["track_id"], 10)

        # Frame 3: xe tiếp tục đi, không được đếm lặp
        frame3 = [{
            "track_id": 10,
            "class_en": "car",
            "class_vi": "O to",
            "bbox": [150, 320, 210, 360]
        }]
        counter.update(frame3)
        self.assertEqual(counter.get_total_count(), 1)

    def test_polygon_roi_counter(self):
        # Vùng hình chữ nhật: x từ 100..300, y từ 100..300
        roi = [(100, 100), (300, 100), (300, 300), (100, 300)]
        counter = VehicleCounter(polygon_coords=roi)

        # Ngoài ROI -> Chưa đếm
        outside_obj = [{
            "track_id": 99,
            "class_en": "motorcycle",
            "class_vi": "Xe may",
            "bbox": [10, 10, 40, 50]
        }]
        counter.update(outside_obj)
        self.assertEqual(counter.get_total_count(), 0)

        # Đi vào ROI -> Đếm
        inside_obj = [{
            "track_id": 99,
            "class_en": "motorcycle",
            "class_vi": "Xe may",
            "bbox": [180, 150, 220, 200]
        }]
        counter.update(inside_obj)
        self.assertEqual(counter.get_total_count(), 1)
        self.assertEqual(counter.counts["motorcycle"], 1)
        self.assertEqual(counter.events_log[-1]["direction"], "roi")

    def test_export_csv(self):
        counter = VehicleCounter(line_coords=((0, 100), (200, 100)))
        counter.update([{"track_id": 1, "class_en": "bus", "class_vi": "Xe buyt", "bbox": [50, 60, 90, 90]}])
        counter.update([{"track_id": 1, "class_en": "bus", "class_vi": "Xe buyt", "bbox": [50, 90, 90, 120]}])

        test_path = "data/output/test_unit_log.csv"
        self.assertTrue(counter.export_csv(test_path))
        self.assertTrue(os.path.exists(test_path))

        with open(test_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
            self.assertEqual(lines[0], "timestamp,track_id,class_en,class_vi,direction,x,y")
            self.assertIn("bus,Xe buyt", lines[1])


class TestVisualizerAndDetector(unittest.TestCase):
    def test_detector_constants(self):
        expected_classes = {"car", "motorcycle", "bus", "truck", "bicycle"}
        self.assertTrue(expected_classes.issubset(set(CLASS_NAMES_VI.keys())))
        self.assertTrue(expected_classes.issubset(set(VEHICLE_COLORS.keys())))

    def test_visualizer_heatmap_and_polygon_roi(self):
        vis = Visualizer()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Test polygon vẽ không crash
        roi = [(50, 50), (200, 50), (200, 200), (50, 200)]
        out_poly = vis.draw_polygon_roi(dummy_frame, roi)
        self.assertEqual(out_poly.shape, dummy_frame.shape)

        # Test heatmap vẽ không crash
        history = {1: [(100, 100), (105, 110), (110, 120)]}
        out_heat = vis.draw_heatmap(dummy_frame, history)
        self.assertEqual(out_heat.shape, dummy_frame.shape)


if __name__ == "__main__":
    unittest.main()
