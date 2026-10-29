"""
Unit tests cho hệ thống RoadDetectionAI (sử dụng thư viện chuẩn unittest).
Không yêu cầu GPU hay tải mô hình YOLO nặng, chạy siêu nhanh (<0.2s).
Chạy lệnh:
    python -m unittest discover tests
"""

import os
import tempfile
import unittest
import numpy as np

from src.tracker import VehicleCounter, intersect, ccw
from src.detector import VEHICLE_COLORS, CLASS_NAMES_VI
from src.visualizer import Visualizer
from src.video_processing import cleanup_temp_video


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

        # Chạm đúng đầu mút vẫn được tính là giao nhau.
        self.assertTrue(intersect((0, 0), (10, 0), (10, 0), (10, 10)))

        # Hai đoạn thẳng thẳng hàng nhưng tách rời không giao nhau.
        self.assertFalse(intersect((0, 0), (10, 0), (11, 0), (20, 0)))

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
            self.assertEqual(lines[0], "timestamp,track_id,class_en,class_vi,direction,x,y,speed_kmh,speed_avg_kmh,speed_max_kmh")
            self.assertIn("bus,Xe buyt", lines[1])

    def test_stationary_vehicle_speed_is_zero(self):
        counter = VehicleCounter(pixels_per_meter=10)
        obj = {"track_id": 7, "class_en": "car", "bbox": [100, 100, 140, 140]}
        counter.update([obj.copy()], timestamp=0.0)
        second = obj.copy()
        counter.update([second], timestamp=1.0)
        self.assertEqual(second["speed_kmh"], 0.0)
        self.assertEqual(counter.current_speeds[7], 0.0)

    def test_moving_vehicle_speed_is_attached_to_object(self):
        counter = VehicleCounter(pixels_per_meter=10)
        counter.update([{"track_id": 6, "class_en": "car", "bbox": [0, 0, 20, 20]}], timestamp=0.0)
        moving = {"track_id": 6, "class_en": "car", "bbox": [0, 10, 20, 30]}
        counter.update([moving], timestamp=1.0)
        self.assertAlmostEqual(moving["speed_kmh"], 3.6, places=1)

    def test_missing_timestamp_uses_safe_fallback(self):
        counter = VehicleCounter(pixels_per_meter=10)
        counter.update([{"track_id": 8, "class_en": "car", "bbox": [0, 0, 20, 20]}])
        counter.update([{"track_id": 8, "class_en": "car", "bbox": [0, 0, 20, 20]}])
        self.assertEqual(counter.current_speeds[8], 0.0)

    def test_lost_track_is_removed_from_live_speed_summary(self):
        counter = VehicleCounter(pixels_per_meter=10)
        counter.update([{"track_id": 9, "class_en": "car", "bbox": [0, 0, 20, 20]}], timestamp=0.0)
        counter.update([{"track_id": 9, "class_en": "car", "bbox": [0, 0, 20, 30]}], timestamp=1.0)
        self.assertIn(9, counter.current_speeds)
        counter.update([], timestamp=2.0)
        self.assertNotIn(9, counter.current_speeds)
        self.assertEqual(counter.get_speed_summary()["warning_count"], 0)


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


class TestNoiseProcessing(unittest.TestCase):
    def setUp(self):
        from src.image_processing import ImageProcessor
        self.ip = ImageProcessor()
        # Ảnh test đơn giản 60x60x3
        self.img = np.full((60, 60, 3), 120, dtype=np.uint8)

    def test_noise_generation_shapes_and_types(self):
        g = self.ip.add_gaussian_noise(self.img, mean=0, sigma=15)
        self.assertEqual(g.shape, self.img.shape)
        self.assertEqual(g.dtype, np.uint8)

        sp = self.ip.add_salt_pepper_noise(self.img, amount=0.05)
        self.assertEqual(sp.shape, self.img.shape)
        self.assertEqual(sp.dtype, np.uint8)

        p = self.ip.add_poisson_noise(self.img)
        self.assertEqual(p.shape, self.img.shape)
        self.assertEqual(p.dtype, np.uint8)

        sk = self.ip.add_speckle_noise(self.img, sigma=0.1)
        self.assertEqual(sk.shape, self.img.shape)
        self.assertEqual(sk.dtype, np.uint8)

        u = self.ip.add_uniform_noise(self.img, low=-20, high=20)
        self.assertEqual(u.shape, self.img.shape)
        self.assertEqual(u.dtype, np.uint8)

    def test_compute_psnr_mse(self):
        # Ảnh giống nhau -> PSNR vô hạn, MSE = 0
        res_ident = self.ip.compute_psnr_mse(self.img, self.img)
        self.assertEqual(res_ident["mse"], 0.0)
        self.assertEqual(res_ident["psnr"], float('inf'))

        # Ảnh nhiễu -> MSE > 0, PSNR hữu hạn
        noisy = self.ip.add_gaussian_noise(self.img, sigma=20)
        res_noisy = self.ip.compute_psnr_mse(self.img, noisy)
        self.assertGreater(res_noisy["mse"], 0)
        self.assertGreater(res_noisy["psnr"], 0)
        self.assertLess(res_noisy["psnr"], 60)

    def test_denoising_methods(self):
        noisy = self.ip.add_salt_pepper_noise(self.img, amount=0.05)
        for method in ["nlm", "median", "bilateral", "mean", "gaussian"]:
            denoised = self.ip.apply_denoising(noisy, method=method)
            self.assertEqual(denoised.shape, self.img.shape)
            self.assertEqual(denoised.dtype, np.uint8)

    def test_demonstrate_noise_processing(self):
        results = self.ip.demonstrate_noise_processing(self.img)
        expected_keys = {"Gaussian", "Salt-and-Pepper", "Poisson", "Speckle", "Uniform"}
        self.assertEqual(set(results.keys()), expected_keys)
        for k, v in results.items():
            self.assertIn("noisy", v)
            self.assertIn("denoised", v)
            self.assertIn("filter", v)
            self.assertIn("psnr_noisy", v)
            self.assertIn("psnr_denoised", v)


class TestVideoProcessing(unittest.TestCase):
    def test_cleanup_temp_video_removes_owned_temp_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        cleanup_temp_video(temp_path)
        self.assertFalse(os.path.exists(temp_path))


if __name__ == "__main__":
    unittest.main()
