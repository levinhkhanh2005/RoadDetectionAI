"""
Module: ImageProcessing (Xử lý Ảnh Cổ điển)
Môn học: Xử lý Ảnh và Thị giác Máy tính (Image Processing & Computer Vision)

Chức năng: Triển khai các kỹ thuật xử lý ảnh cổ điển (không dùng Deep Learning)
để phát hiện phương tiện giao thông, bao gồm:

1. Tiền xử lý ảnh (Preprocessing):
   - Chuyển đổi không gian màu (Color Space Conversion)
   - Lọc nhiễu Gaussian (Gaussian Blur)
   - Cân bằng lược đồ xám (Histogram Equalization / CLAHE)

2. Phát hiện cạnh (Edge Detection):
   - Toán tử Sobel (Sobel Operator)
   - Thuật toán Canny (Canny Edge Detector)
   - Toán tử Laplacian of Gaussian (LoG)

3. Phép toán hình thái học (Morphological Operations):
   - Giãn nở (Dilation), Co (Erosion)
   - Mở (Opening), Đóng (Closing)

4. Trừ nền chuyển động (Background Subtraction):
   - MOG2 (Mixture of Gaussians v2)
   - KNN Background Subtractor

5. Phát hiện contour & Bounding Box:
   - Tìm đường viền (Contour Detection)
   - Lọc kích thước hợp lệ (Area/Aspect Ratio Filtering)

6. Phân vùng theo màu sắc (Color-based Segmentation):
   - Phân vùng mặt đường (Road Region Segmentation)
   - Phát hiện phương tiện theo dải màu HSV

7. Biến đổi phối cảnh (Perspective Transform):
   - Chuyển đổi góc nhìn Bird's Eye View

Mỗi hàm trả về cả kết quả xử lý lẫn ảnh trung gian (intermediate steps)
để phục vụ mục đích giảng dạy và trình bày trực quan.
"""

from typing import List, Dict, Tuple, Any, Optional
import cv2
import numpy as np


class ImageProcessor:
    """
    Bộ xử lý ảnh cổ điển (Classical Image Processing Pipeline)
    cho bài toán nhận diện phương tiện giao thông đường bộ.
    """

    def __init__(self):
        """Khởi tạo các tham số và bộ trừ nền."""
        # Bộ trừ nền MOG2 (Background Subtraction)
        self.bg_subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )
        # Bộ trừ nền KNN
        self.bg_subtractor_knn = cv2.createBackgroundSubtractorKNN(
            history=500,
            dist2Threshold=400.0,
            detectShadows=True
        )
        # Kernel chuẩn cho phép hình thái học
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # =====================================================================
    # 1. TIỀN XỬ LÝ ẢNH (IMAGE PREPROCESSING)
    # =====================================================================

    def convert_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi ảnh sang thang xám (Grayscale).
        Công thức: Gray = 0.299*R + 0.587*G + 0.114*B
        """
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def convert_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi ảnh sang không gian màu HSV (Hue, Saturation, Value).
        HSV tách biệt thông tin màu sắc và cường độ sáng,
        phù hợp hơn RGB cho bài toán phân vùng theo màu.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def apply_gaussian_blur(self, image: np.ndarray, ksize: int = 5) -> np.ndarray:
        """
        Áp dụng bộ lọc Gaussian (Gaussian Blur) để làm mịn ảnh và loại nhiễu.
        Bộ lọc sử dụng hàm phân phối Gaussian 2D làm nhân tích chập (kernel).
        """
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def apply_bilateral_filter(self, image: np.ndarray, d: int = 9,
                               sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        Bộ lọc Bilateral: Làm mịn nhưng BẢO TOÀN CẠNH viền.
        Khác Gaussian Blur, Bilateral cân nhắc cả khoảng cách không gian
        lẫn khoảng cách cường độ pixel để tránh làm mờ biên.
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Cân bằng lược đồ xám (Histogram Equalization).
        Tái phân bố các mức xám để tăng độ tương phản toàn cục,
        giúp phân biệt rõ phương tiện và nền đường.
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        return cv2.equalizeHist(gray)

    def apply_clahe(self, image: np.ndarray, clip_limit: float = 3.0,
                    tile_size: int = 8) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Cải tiến của Histogram Equalization: chia ảnh thành các ô nhỏ (tiles)
        và cân bằng lược đồ cục bộ, có giới hạn contrast để tránh khuếch đại nhiễu.
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(gray)

    # =====================================================================
    # 2. PHÁT HIỆN CẠNH (EDGE DETECTION)
    # =====================================================================

    def detect_edges_canny(self, image: np.ndarray, low_threshold: int = 50,
                           high_threshold: int = 150) -> np.ndarray:
        """
        Thuật toán Canny Edge Detection (J. Canny, 1986).
        Pipeline: Gaussian Blur → Tính Gradient (Sobel) → Non-Maximum Suppression
                  → Double Thresholding → Hysteresis Edge Tracking.
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return edges

    def detect_edges_sobel(self, image: np.ndarray, ksize: int = 3) -> Dict[str, np.ndarray]:
        """
        Toán tử Sobel: Tính gradient theo 2 hướng X và Y.
        Sobel_X phát hiện cạnh dọc (vertical edges),
        Sobel_Y phát hiện cạnh ngang (horizontal edges).
        Gradient magnitude = sqrt(Gx² + Gy²)
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Gradient theo hướng X (cạnh dọc)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_x_abs = cv2.convertScaleAbs(sobel_x)

        # Gradient theo hướng Y (cạnh ngang)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel_y_abs = cv2.convertScaleAbs(sobel_y)

        # Kết hợp magnitude
        magnitude = cv2.magnitude(sobel_x.astype(np.float64), sobel_y.astype(np.float64))
        magnitude = cv2.convertScaleAbs(magnitude)

        return {
            "sobel_x": sobel_x_abs,
            "sobel_y": sobel_y_abs,
            "magnitude": magnitude
        }

    def detect_edges_laplacian(self, image: np.ndarray) -> np.ndarray:
        """
        Toán tử Laplacian (đạo hàm bậc hai).
        Nhạy với nhiễu hơn Sobel, nhưng phát hiện cạnh đẳng hướng (isotropic).
        Thường kết hợp với Gaussian Blur → Laplacian of Gaussian (LoG).
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

    # =====================================================================
    # 3. PHÉP TOÁN HÌNH THÁI HỌC (MORPHOLOGICAL OPERATIONS)
    # =====================================================================

    def apply_morphology(self, mask: np.ndarray, operation: str = "close",
                         kernel_size: str = "medium", iterations: int = 2) -> np.ndarray:
        """
        Áp dụng phép toán hình thái học trên ảnh nhị phân/mask.

        Các phép toán:
        - 'erode': Co nhỏ vùng trắng, loại bỏ nhiễu nhỏ.
        - 'dilate': Giãn nở vùng trắng, lấp đầy lỗ hổng nhỏ.
        - 'open': Erosion → Dilation. Loại nhiễu hạt tiêu (salt noise).
        - 'close': Dilation → Erosion. Lấp đầy lỗ hổng bên trong đối tượng.
        - 'gradient': Dilation - Erosion. Trích xuất đường viền hình thái.
        """
        kernels = {
            "small": self.kernel_small,
            "medium": self.kernel_medium,
            "large": self.kernel_large
        }
        kernel = kernels.get(kernel_size, self.kernel_medium)

        ops = {
            "erode": cv2.MORPH_ERODE,
            "dilate": cv2.MORPH_DILATE,
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT
        }
        morph_op = ops.get(operation, cv2.MORPH_CLOSE)

        return cv2.morphologyEx(mask, morph_op, kernel, iterations=iterations)

    def demonstrate_morphology_steps(self, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Trình diễn lần lượt từng phép toán hình thái học.
        Mục đích: Minh họa trực quan cho bài giảng / báo cáo.
        """
        return {
            "original": mask.copy(),
            "erosion": cv2.erode(mask, self.kernel_medium, iterations=1),
            "dilation": cv2.dilate(mask, self.kernel_medium, iterations=1),
            "opening": cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_medium, iterations=1),
            "closing": cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_medium, iterations=1),
            "gradient": cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, self.kernel_medium, iterations=1),
        }

    # =====================================================================
    # 4. TRỪ NỀN CHUYỂN ĐỘNG (BACKGROUND SUBTRACTION)
    # =====================================================================

    def apply_background_subtraction(self, frame: np.ndarray,
                                     method: str = "MOG2") -> Dict[str, np.ndarray]:
        """
        Trừ nền để tách phương tiện đang di chuyển (foreground) ra khỏi nền tĩnh.

        MOG2 (Mixture of Gaussians v2, Zivkovic 2004):
            Mô hình hóa mỗi pixel bằng hỗn hợp nhiều phân phối Gaussian.
            Tự động xác định số thành phần Gaussian tối ưu.

        KNN (K-Nearest Neighbors):
            Phân loại pixel foreground/background dựa trên k-láng giềng gần nhất
            trong lịch sử các giá trị pixel.
        """
        if method == "MOG2":
            fg_mask = self.bg_subtractor_mog2.apply(frame)
        else:
            fg_mask = self.bg_subtractor_knn.apply(frame)

        # Loại bỏ vùng bóng (shadow, giá trị = 127 trong mask)
        _, fg_mask_clean = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Áp dụng hình thái học để loại nhiễu và lấp lỗ hổng
        fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_OPEN, self.kernel_small, iterations=1)
        fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_CLOSE, self.kernel_large, iterations=2)
        fg_mask_clean = cv2.dilate(fg_mask_clean, self.kernel_medium, iterations=1)

        return {
            "fg_mask_raw": fg_mask,
            "fg_mask_clean": fg_mask_clean,
        }

    def reset_background_subtractor(self, method: str = "both"):
        """Khởi tạo lại bộ trừ nền (khi đổi video/source)."""
        if method in ("MOG2", "both"):
            self.bg_subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=50, detectShadows=True
            )
        if method in ("KNN", "both"):
            self.bg_subtractor_knn = cv2.createBackgroundSubtractorKNN(
                history=500, dist2Threshold=400.0, detectShadows=True
            )

    # =====================================================================
    # 5. PHÁT HIỆN CONTOUR & BOUNDING BOX
    # =====================================================================

    def detect_contours(self, mask: np.ndarray,
                        min_area: int = 800, max_area: int = 100000,
                        min_aspect_ratio: float = 0.2, max_aspect_ratio: float = 5.0
                        ) -> List[Dict[str, Any]]:
        """
        Tìm và lọc các đường viền (contour) trên ảnh nhị phân.

        Quy trình:
        1. Tìm tất cả contour bằng cv2.findContours().
        2. Lọc theo diện tích (area) để loại bỏ nhiễu nhỏ và vùng quá lớn.
        3. Lọc theo tỷ lệ khung hình (aspect ratio) để chỉ giữ hình dạng xe hợp lệ.
        4. Tính Bounding Box cho mỗi contour hợp lệ.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / max(h, 1)

            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue

            # Tính tâm và các đặc trưng
            M = cv2.moments(contour)
            cx = int(M["m10"] / max(M["m00"], 1))
            cy = int(M["m01"] / max(M["m00"], 1))

            detections.append({
                "bbox": [x, y, x + w, y + h],
                "area": area,
                "aspect_ratio": round(aspect_ratio, 2),
                "center": (cx, cy),
                "contour": contour,
                "class_vi": "Phuong tien",
                "class_en": "vehicle",
                "confidence": round(min(area / 5000.0, 1.0), 2),  # Ước lượng độ tin cậy từ diện tích
            })

        # Sắp xếp theo diện tích giảm dần
        detections.sort(key=lambda d: d["area"], reverse=True)
        return detections

    def draw_contour_detections(self, frame: np.ndarray,
                                detections: List[Dict[str, Any]],
                                color: Tuple[int, int, int] = (0, 255, 0),
                                draw_contour: bool = True) -> np.ndarray:
        """Vẽ Bounding Box và Contour lên ảnh gốc."""
        out = frame.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]

            # Vẽ bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Vẽ contour
            if draw_contour and "contour" in det:
                cv2.drawContours(out, [det["contour"]], -1, (255, 255, 0), 1)

            # Vẽ nhãn
            label = f"#{i+1} A={det['area']:.0f}"
            cv2.putText(out, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # Vẽ tâm
            cx, cy = det["center"]
            cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        # HUD tổng kết
        hud_text = f"Classical CV: {len(detections)} phuong tien"
        (tw, th), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (10, 10), (20 + tw, 20 + th + 10), (0, 0, 0), -1)
        cv2.putText(out, hud_text, (15, 30 + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

        return out

    # =====================================================================
    # 6. PHÂN VÙNG THEO MÀU SẮC (COLOR-BASED SEGMENTATION)
    # =====================================================================

    def segment_by_color_range(self, image: np.ndarray,
                               lower_hsv: Tuple[int, int, int],
                               upper_hsv: Tuple[int, int, int]) -> np.ndarray:
        """
        Phân vùng ảnh theo dải màu HSV (Hue-Saturation-Value).
        Trả về mask nhị phân: pixel nằm trong dải → trắng, ngoài → đen.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
        return mask

    def segment_road_region(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Phân vùng mặt đường bằng phương pháp kết hợp:
        1. Lọc màu xám (asphalt color range) trong HSV.
        2. Áp dụng hình thái học để làm mịn vùng đường.
        3. Tạo mask vùng quan tâm (ROI) để chỉ phân tích khu vực mặt đường.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Dải màu asphalt (mặt đường nhựa): xám tối → xám sáng
        lower_road = np.array([0, 0, 40])
        upper_road = np.array([180, 60, 180])
        road_mask = cv2.inRange(hsv, lower_road, upper_road)

        # Hình thái học: Đóng → loại lỗ hổng, Mở → loại nhiễu
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, self.kernel_large, iterations=3)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, self.kernel_large, iterations=2)

        # Tạo overlay trực quan hóa vùng đường
        road_overlay = image.copy()
        road_colored = np.zeros_like(image)
        road_colored[:] = (0, 180, 0)  # Phủ xanh lá lên vùng đường
        road_overlay[road_mask > 0] = cv2.addWeighted(
            image[road_mask > 0], 0.6,
            road_colored[road_mask > 0], 0.4, 0
        )

        return {
            "road_mask": road_mask,
            "road_overlay": road_overlay,
        }

    def detect_vehicle_colors(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Phát hiện phương tiện theo các dải màu phổ biến trong HSV.
        Trả về dictionary các mask phân vùng theo từng nhóm màu.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        color_ranges = {
            "Do (Red)": [
                ((0, 70, 50), (10, 255, 255)),
                ((170, 70, 50), (180, 255, 255))
            ],
            "Xanh duong (Blue)": [
                ((100, 70, 50), (130, 255, 255)),
            ],
            "Trang (White)": [
                ((0, 0, 180), (180, 30, 255)),
            ],
            "Den (Black)": [
                ((0, 0, 0), (180, 255, 50)),
            ],
            "Vang (Yellow)": [
                ((20, 80, 80), (35, 255, 255)),
            ],
        }

        results = {}
        for color_name, ranges in color_ranges.items():
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Loại nhiễu
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel_small)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel_medium)
            results[color_name] = combined_mask

        return results

    # =====================================================================
    # 7. BIẾN ĐỔI PHỐI CẢNH (PERSPECTIVE TRANSFORM)
    # =====================================================================

    def perspective_transform(self, image: np.ndarray,
                              src_points: Optional[np.ndarray] = None,
                              dst_points: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Biến đổi phối cảnh (Perspective Transform / Homography).
        Chuyển đổi góc nhìn camera thành Bird's Eye View (nhìn từ trên xuống)
        để đo khoảng cách và mật độ xe chính xác hơn.

        Nếu không cung cấp src_points, sử dụng hình thang mặc định
        bao quát vùng đường phía dưới ảnh.
        """
        h, w = image.shape[:2]

        if src_points is None:
            # Hình thang mặc định: vùng đường phía dưới ảnh
            src_points = np.array([
                [int(w * 0.15), int(h * 0.95)],    # Góc dưới trái
                [int(w * 0.85), int(h * 0.95)],    # Góc dưới phải
                [int(w * 0.65), int(h * 0.55)],    # Góc trên phải
                [int(w * 0.35), int(h * 0.55)],    # Góc trên trái
            ], dtype=np.float32)

        if dst_points is None:
            dst_points = np.array([
                [int(w * 0.2), h],
                [int(w * 0.8), h],
                [int(w * 0.8), 0],
                [int(w * 0.2), 0],
            ], dtype=np.float32)

        # Tính ma trận biến đổi phối cảnh
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Áp dụng biến đổi
        warped = cv2.warpPerspective(image, M, (w, h))

        # Vẽ vùng ROI lên ảnh gốc
        src_vis = image.copy()
        pts = src_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        for i, pt in enumerate(src_points.astype(int)):
            cv2.circle(src_vis, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(src_vis, f"P{i+1}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        return {
            "warped": warped,
            "source_annotated": src_vis,
            "transform_matrix": M,
        }

    # =====================================================================
    # 8. PIPELINE TỔNG HỢP (FULL CLASSICAL CV PIPELINE)
    # =====================================================================

    def full_pipeline_image(self, image: np.ndarray,
                            blur_ksize: int = 5,
                            canny_low: int = 50,
                            canny_high: int = 150,
                            min_contour_area: int = 800,
                            max_contour_area: int = 100000,
                            clahe_clip: float = 3.0
                            ) -> Dict[str, Any]:
        """
        Pipeline xử lý ảnh tĩnh hoàn chỉnh:
        Ảnh gốc → Grayscale → CLAHE → Gaussian Blur → Canny Edge
        → Morphology (Close) → Contour Detection → Bounding Boxes

        Trả về tất cả ảnh trung gian để trình bày từng bước xử lý.
        """
        # Bước 1: Chuyển xám
        gray = self.convert_grayscale(image)

        # Bước 2: Tăng cường tương phản bằng CLAHE
        enhanced = self.apply_clahe(gray, clip_limit=clahe_clip)

        # Bước 3: Lọc nhiễu Gaussian
        blurred = self.apply_gaussian_blur(enhanced, ksize=blur_ksize)

        # Bước 4: Phát hiện cạnh Canny
        edges = self.detect_edges_canny(blurred, canny_low, canny_high)

        # Bước 5: Phép đóng hình thái học (lấp lỗ hổng cạnh)
        closed = self.apply_morphology(edges, "close", "large", iterations=3)

        # Bước 5b: Giãn nở thêm để nối các cạnh rời rạc
        dilated = cv2.dilate(closed, self.kernel_medium, iterations=2)

        # Bước 6: Tìm contour và lọc
        detections = self.detect_contours(
            dilated,
            min_area=min_contour_area,
            max_area=max_contour_area
        )

        # Bước 7: Vẽ kết quả
        result_img = self.draw_contour_detections(image, detections)

        return {
            "steps": {
                "1_original": image,
                "2_grayscale": gray,
                "3_clahe": enhanced,
                "4_gaussian_blur": blurred,
                "5_canny_edges": edges,
                "6_morphology_close": closed,
                "7_dilated": dilated,
                "8_result": result_img,
            },
            "detections": detections,
            "result": result_img,
        }

    def full_pipeline_video_frame(self, frame: np.ndarray,
                                  method: str = "MOG2",
                                  min_contour_area: int = 1200,
                                  max_contour_area: int = 80000
                                  ) -> Dict[str, Any]:
        """
        Pipeline xử lý video (từng frame):
        Frame → Background Subtraction → Morphology → Contour Detection → Bounding Boxes

        Phương pháp cổ điển để phát hiện xe đang di chuyển
        mà KHÔNG cần mô hình Deep Learning.
        """
        # Bước 1: Trừ nền
        bg_result = self.apply_background_subtraction(frame, method=method)
        fg_mask = bg_result["fg_mask_clean"]

        # Bước 2: Tìm contour phương tiện
        detections = self.detect_contours(
            fg_mask,
            min_area=min_contour_area,
            max_area=max_contour_area,
            min_aspect_ratio=0.3,
            max_aspect_ratio=4.0
        )

        # Bước 3: Vẽ kết quả
        result_img = self.draw_contour_detections(frame, detections, color=(0, 200, 255))

        return {
            "fg_mask_raw": bg_result["fg_mask_raw"],
            "fg_mask_clean": fg_mask,
            "detections": detections,
            "result": result_img,
        }

    # =====================================================================
    # 9. SO SÁNH PHƯƠNG PHÁP (COMPARISON UTILITIES)
    # =====================================================================

    def create_comparison_grid(self, images: Dict[str, np.ndarray],
                               grid_cols: int = 4, cell_size: Tuple[int, int] = (320, 240)
                               ) -> np.ndarray:
        """
        Tạo lưới so sánh các ảnh trung gian trong pipeline.
        Mỗi ô hiển thị tên bước xử lý + ảnh kết quả.
        """
        cw, ch = cell_size
        n = len(images)
        rows = (n + grid_cols - 1) // grid_cols
        grid = np.zeros((rows * (ch + 30), grid_cols * cw, 3), dtype=np.uint8)

        for idx, (name, img) in enumerate(images.items()):
            row = idx // grid_cols
            col = idx % grid_cols
            x_off = col * cw
            y_off = row * (ch + 30)

            # Resize ảnh
            if len(img.shape) == 2:
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_color = img.copy()
            resized = cv2.resize(img_color, (cw, ch))

            # Đặt vào grid
            grid[y_off + 30:y_off + 30 + ch, x_off:x_off + cw] = resized

            # Vẽ tiêu đề
            cv2.putText(grid, name, (x_off + 5, y_off + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

        return grid

    # =====================================================================
    # 10. NÂNG CAO CHẤT LƯỢNG ẢNH (IMAGE ENHANCEMENT)
    # =====================================================================

    def apply_gamma_correction(self, image: np.ndarray,
                               gamma: float = 1.0) -> np.ndarray:
        """
        Biến đổi Gamma (Gamma Correction / Power-Law Transform).

        Công thức: s = c * r^gamma  (c = 1, r chuẩn hóa về [0, 1])
        - gamma < 1.0: Tăng sáng vùng tối (mở rộng dải tối, nén dải sáng).
          → Dùng cho ảnh ban đêm, thiếu sáng.
        - gamma > 1.0: Tăng tương phản vùng sáng (nén dải tối, mở rộng dải sáng).
          → Dùng khi ảnh bị ngược sáng, cháy sáng.
        - gamma = 1.0: Không thay đổi (hàm đồng nhất).

        Sử dụng Look-Up Table (LUT) để tính trước 256 giá trị → tốc độ O(1) mỗi pixel.
        """
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype("uint8")
        return cv2.LUT(image, table)

    def auto_adjust_brightness(self, image: np.ndarray,
                               target_mean: float = 120.0) -> np.ndarray:
        """
        Tự động điều chỉnh độ sáng dựa trên mean intensity hiện tại.
        Tính gamma cần thiết để đưa mean intensity về target_mean.
        Hữu ích cho camera giao thông hoạt động 24/7 (ngày/đêm).
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        current_mean = np.mean(gray)
        if current_mean < 1:
            return image
        gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
        gamma = np.clip(gamma, 0.3, 3.0)
        return self.apply_gamma_correction(image, gamma)

    def apply_log_transform(self, image: np.ndarray,
                            c: float = 46.0) -> np.ndarray:
        """
        Biến đổi Logarithm (Logarithmic Transform).

        Công thức: s = c * log(1 + r)
        - Mở rộng dải pixel tối (giá trị nhỏ), nén dải pixel sáng.
        - Hằng số c = 255 / log(1 + max_pixel) để chuẩn hóa output về [0, 255].
        - Phù hợp tăng chi tiết trong vùng tối (phổ Fourier, ảnh thiên văn).
        """
        img_float = image.astype(np.float64)
        # c tự động tính nếu dùng giá trị mặc định
        c_auto = 255.0 / np.log(1 + np.max(img_float))
        result = c_auto * np.log(1 + img_float)
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_power_law_transform(self, image: np.ndarray,
                                  gamma: float = 0.4,
                                  c: float = 1.0) -> np.ndarray:
        """
        Biến đổi Lũy thừa (Power-Law / Gamma Transform) tổng quát.

        Công thức: s = c * r^gamma  (r chuẩn hóa về [0, 1])
        Giống apply_gamma_correction nhưng cho phép tùy chỉnh hằng số c.
        Ứng dụng: Hiệu chỉnh gamma cho màn hình CRT, MRI, CT-scan.
        """
        img_normalized = image.astype(np.float64) / 255.0
        result = c * np.power(img_normalized, gamma) * 255.0
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_contrast_stretching(self, image: np.ndarray,
                                  low_percentile: float = 2.0,
                                  high_percentile: float = 98.0) -> np.ndarray:
        """
        Kéo giãn tương phản (Contrast Stretching / Histogram Stretching).

        Ánh xạ tuyến tính dải [p_low, p_high] về [0, 255]:
            s = 255 * (r - r_min) / (r_max - r_min)

        Sử dụng percentile thay vì min/max thuần để tránh ảnh hưởng
        bởi pixel nhiễu ngoại lai (outlier).
        """
        if len(image.shape) == 3:
            # Xử lý từng kênh màu
            channels = cv2.split(image)
            stretched = []
            for ch in channels:
                p_low = np.percentile(ch, low_percentile)
                p_high = np.percentile(ch, high_percentile)
                ch_stretched = np.clip((ch.astype(np.float64) - p_low) *
                                       255.0 / max(p_high - p_low, 1), 0, 255)
                stretched.append(ch_stretched.astype(np.uint8))
            return cv2.merge(stretched)
        else:
            p_low = np.percentile(image, low_percentile)
            p_high = np.percentile(image, high_percentile)
            result = np.clip((image.astype(np.float64) - p_low) *
                              255.0 / max(p_high - p_low, 1), 0, 255)
            return result.astype(np.uint8)

    def apply_unsharp_masking(self, image: np.ndarray,
                              kernel_size: int = 5,
                              sigma: float = 1.0,
                              amount: float = 1.5,
                              threshold: int = 0) -> np.ndarray:
        """
        Làm sắc nét bằng Unsharp Masking.

        Nguyên lý: Sharpened = Original + amount * (Original - Blurred)
        1. Tạo ảnh mờ (Gaussian Blur) → "unsharp mask".
        2. Trừ ảnh gốc cho ảnh mờ → lấy chi tiết cạnh.
        3. Cộng ngược chi tiết vào ảnh gốc với hệ số amount.

        Threshold: Chỉ làm sắc pixel có chênh lệch > threshold (tránh khuếch đại nhiễu).
        """
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        # Tính chi tiết cạnh
        detail = image.astype(np.float64) - blurred.astype(np.float64)

        if threshold > 0:
            # Chỉ áp dụng cho pixel có chênh lệch lớn
            mask = np.abs(detail) > threshold
            sharpened = image.astype(np.float64).copy()
            sharpened[mask] += amount * detail[mask]
        else:
            sharpened = image.astype(np.float64) + amount * detail

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def apply_denoising(self, image: np.ndarray,
                        method: str = "nlm",
                        h: float = 10.0,
                        ksize: int = 5) -> np.ndarray:
        """
        Khử nhiễu ảnh (Image Denoising).

        Phương pháp:
        - 'nlm': Non-Local Means (Buades et al., 2005)
          So sánh các patch (vùng nhỏ) trong toàn ảnh → trung bình có trọng số.
          Bảo toàn chi tiết tốt hơn Gaussian Blur.
          h: Cường độ lọc (h lớn → mịn hơn nhưng mất chi tiết).

        - 'bilateral': Bilateral Filter
          Kết hợp trọng số không gian (spatial) và cường độ (intensity).
          Bảo toàn cạnh viền tốt.

        - 'median': Median Filter
          Thay mỗi pixel bằng giá trị trung vị của láng giềng.
          Hiệu quả với nhiễu muối tiêu (salt-and-pepper noise).

        - 'mean': Box / Mean Filter
          Lọc trung bình láng giềng ksize x ksize.

        - 'gaussian': Gaussian Filter
          Làm mượt bằng phân phối chuẩn 2D.
        """
        k = ksize if ksize % 2 == 1 else ksize + 1
        if method == "nlm":
            if len(image.shape) == 3:
                return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
            else:
                return cv2.fastNlMeansDenoising(image, None, h, 7, 21)
        elif method == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == "median":
            return cv2.medianBlur(image, k)
        elif method == "mean":
            return cv2.blur(image, (k, k))
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (k, k), 0)
        else:
            return image

    def apply_white_balance_grayworld(self, image: np.ndarray) -> np.ndarray:
        """
        Cân bằng trắng bằng giả thuyết Gray World (Gray World Assumption).

        Giả thuyết: Trung bình tất cả pixel trong ảnh nên là xám trung tính.
        → Điều chỉnh từng kênh B, G, R sao cho mean ≈ mean tổng.
        Ứng dụng: Sửa ảnh bị ám vàng/xanh do ánh sáng đèn đường ban đêm.
        """
        result = image.copy().astype(np.float64)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_all = (avg_b + avg_g + avg_r) / 3.0

        result[:, :, 0] *= avg_all / max(avg_b, 1)
        result[:, :, 1] *= avg_all / max(avg_g, 1)
        result[:, :, 2] *= avg_all / max(avg_r, 1)

        return np.clip(result, 0, 255).astype(np.uint8)

    def demonstrate_enhancement_methods(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Trình diễn tất cả phương pháp nâng cao chất lượng ảnh.
        Trả về dictionary {tên_phương_pháp: ảnh_kết_quả}.
        """
        return {
            "Original": image.copy(),
            "Gamma 0.4 (Tang sang)": self.apply_gamma_correction(image, 0.4),
            "Gamma 2.0 (Giam sang)": self.apply_gamma_correction(image, 2.0),
            "Auto Brightness": self.auto_adjust_brightness(image),
            "Log Transform": self.apply_log_transform(image),
            "Power-Law (g=0.5)": self.apply_power_law_transform(image, gamma=0.5),
            "Contrast Stretching": self.apply_contrast_stretching(image),
            "CLAHE": cv2.cvtColor(
                self.apply_clahe(image), cv2.COLOR_GRAY2BGR
            ) if len(image.shape) == 3 else self.apply_clahe(image),
            "Unsharp Masking": self.apply_unsharp_masking(image),
            "Denoise NLM": self.apply_denoising(image, "nlm"),
            "Denoise Bilateral": self.apply_denoising(image, "bilateral"),
            "White Balance": self.apply_white_balance_grayworld(image),
        }

    # =====================================================================
    # 11. BIỂU ĐỒ TẦN SUẤT (HISTOGRAM ANALYSIS & VISUALIZATION)
    # =====================================================================

    def compute_histogram(self, image: np.ndarray,
                          channel: int = 0,
                          bins: int = 256,
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Tính biểu đồ tần suất (Histogram) cho một kênh ảnh.

        Histogram h(k) = số pixel có giá trị cường độ k, với k ∈ [0, 255].
        Biểu đồ cho biết phân bố cường độ pixel → đánh giá độ sáng,
        tương phản, và đặc tính tổng quan của ảnh.

        Args:
            image: Ảnh đầu vào (grayscale hoặc BGR).
            channel: Kênh cần tính (0=B/Gray, 1=G, 2=R).
            bins: Số bin (mặc định 256 = toàn bộ dải).
            mask: Mask tùy chọn để chỉ tính trên vùng quan tâm.
        Returns:
            Mảng 1D chứa giá trị histogram.
        """
        if len(image.shape) == 2:
            hist = cv2.calcHist([image], [0], mask, [bins], [0, 256])
        else:
            hist = cv2.calcHist([image], [channel], mask, [bins], [0, 256])
        return hist.flatten()

    def compute_normalized_histogram(self, image: np.ndarray,
                                     channel: int = 0,
                                     bins: int = 256) -> np.ndarray:
        """
        Histogram chuẩn hóa (Normalized Histogram / Probability Distribution).

        p(k) = h(k) / N,  với N = tổng số pixel.
        Tổng p(k) = 1.0 → có thể coi là xác suất xuất hiện của mức xám k.
        """
        hist = self.compute_histogram(image, channel, bins)
        total_pixels = image.shape[0] * image.shape[1]
        return hist / max(total_pixels, 1)

    def compute_cumulative_histogram(self, image: np.ndarray,
                                     channel: int = 0) -> np.ndarray:
        """
        Biểu đồ tần suất tích lũy (Cumulative Distribution Function - CDF).

        CDF(k) = Σ p(i) với i = 0..k
        Ứng dụng: Là cơ sở của thuật toán Histogram Equalization.
        CDF cho biết tỷ lệ pixel có giá trị ≤ k.
        """
        hist = self.compute_normalized_histogram(image, channel)
        return np.cumsum(hist)

    def draw_histogram_image(self, image: np.ndarray,
                             width: int = 512, height: int = 400,
                             draw_cdf: bool = True) -> np.ndarray:
        """
        Vẽ biểu đồ tần suất trực quan bằng OpenCV.

        Nếu ảnh màu: vẽ 3 histogram (B, G, R) chồng lên nhau.
        Nếu ảnh xám: vẽ 1 histogram xám.
        Tùy chọn vẽ thêm đường CDF (đường tích lũy).

        Returns:
            Ảnh BGR chứa biểu đồ histogram đã vẽ.
        """
        hist_img = np.zeros((height, width, 3), dtype=np.uint8)
        hist_img[:] = (30, 30, 30)  # Nền tối

        if len(image.shape) == 2:
            # Ảnh grayscale
            hist = self.compute_histogram(image, 0)
            hist_norm = hist / max(hist.max(), 1) * (height - 40)
            bin_w = width / 256.0

            for i in range(1, 256):
                pt1 = (int((i - 1) * bin_w), height - 20 - int(hist_norm[i - 1]))
                pt2 = (int(i * bin_w), height - 20 - int(hist_norm[i]))
                cv2.line(hist_img, pt1, pt2, (200, 200, 200), 1, cv2.LINE_AA)

            if draw_cdf:
                cdf = self.compute_cumulative_histogram(image, 0)
                cdf_scaled = cdf * (height - 40)
                for i in range(1, 256):
                    pt1 = (int((i - 1) * bin_w), height - 20 - int(cdf_scaled[i - 1]))
                    pt2 = (int(i * bin_w), height - 20 - int(cdf_scaled[i]))
                    cv2.line(hist_img, pt1, pt2, (0, 200, 255), 1, cv2.LINE_AA)
        else:
            # Ảnh màu BGR
            colors_bgr = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            colors_cdf = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
            channel_names = ["B", "G", "R"]

            for ch_idx in range(3):
                hist = self.compute_histogram(image, ch_idx)
                hist_norm = hist / max(hist.max(), 1) * (height - 40)
                bin_w = width / 256.0

                for i in range(1, 256):
                    pt1 = (int((i - 1) * bin_w), height - 20 - int(hist_norm[i - 1]))
                    pt2 = (int(i * bin_w), height - 20 - int(hist_norm[i]))
                    cv2.line(hist_img, pt1, pt2, colors_bgr[ch_idx], 1, cv2.LINE_AA)

                if draw_cdf:
                    cdf = self.compute_cumulative_histogram(image, ch_idx)
                    cdf_scaled = cdf * (height - 40)
                    for i in range(1, 256):
                        pt1 = (int((i - 1) * bin_w), height - 20 - int(cdf_scaled[i - 1]))
                        pt2 = (int(i * bin_w), height - 20 - int(cdf_scaled[i]))
                        cv2.line(hist_img, pt1, pt2, colors_cdf[ch_idx], 1, cv2.LINE_AA)

        # Vẽ trục và tiêu đề
        cv2.line(hist_img, (0, height - 20), (width, height - 20), (100, 100, 100), 1)
        cv2.putText(hist_img, "Bieu do tan suat (Histogram)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(hist_img, "0", (5, height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(hist_img, "255", (width - 30, height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        if draw_cdf:
            cv2.putText(hist_img, "--- CDF", (width - 80, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA)

        return hist_img

    def compare_histograms(self, image1: np.ndarray, image2: np.ndarray,
                           method: str = "correlation") -> Dict[str, Any]:
        """
        So sánh biểu đồ tần suất giữa 2 ảnh.

        Các phương pháp so sánh:
        - 'correlation':  Tương quan Pearson. Giá trị [-1, 1], 1 = giống hoàn toàn.
        - 'chi_square':   Khoảng cách Chi-Square. 0 = giống, lớn = khác.
        - 'intersection':  Giao của 2 histogram. Lớn = giống.
        - 'bhattacharyya': Khoảng cách Bhattacharyya. [0, 1], 0 = giống.
        """
        methods = {
            "correlation": cv2.HISTCMP_CORREL,
            "chi_square": cv2.HISTCMP_CHISQR,
            "intersection": cv2.HISTCMP_INTERSECT,
            "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
        }

        gray1 = self.convert_grayscale(image1) if len(image1.shape) == 3 else image1
        gray2 = self.convert_grayscale(image2) if len(image2.shape) == 3 else image2

        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        results = {}
        for name, method_id in methods.items():
            results[name] = round(cv2.compareHist(hist1, hist2, method_id), 4)

        return results

    def histogram_backprojection(self, image: np.ndarray,
                                 target_roi: np.ndarray) -> np.ndarray:
        """
        Chiếu ngược biểu đồ (Histogram Backprojection).

        Cho 1 vùng mẫu (target_roi), tìm tất cả pixel trong ảnh gốc
        có phân bố màu giống vùng mẫu.
        Ứng dụng: Theo dõi xe theo màu sắc, tìm đối tượng tương tự.

        Thuật toán:
        1. Tính histogram Hue-Saturation của vùng mẫu.
        2. Backproject: Mỗi pixel (h, s) → giá trị histogram tại (h, s).
        3. Lọc nhiễu kết quả.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_roi = cv2.cvtColor(target_roi, cv2.COLOR_BGR2HSV)

        # Tính histogram 2D (Hue x Saturation) của vùng mẫu
        roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Chiếu ngược
        backproj = cv2.calcBackProject([hsv_image], [0, 1], roi_hist, [0, 180, 0, 256], 1)

        # Lọc bằng convolution với kernel tròn
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(backproj, -1, disc, backproj)

        _, backproj = cv2.threshold(backproj, 50, 255, cv2.THRESH_BINARY)
        return backproj

    def demonstrate_histogram_effects(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Trình diễn hiệu quả các phép biến đổi lên histogram.
        Mỗi cặp gồm: ảnh sau biến đổi + histogram tương ứng.
        """
        results = {}
        transforms = {
            "Original": image.copy(),
            "Hist Equalization": cv2.cvtColor(
                self.apply_histogram_equalization(image), cv2.COLOR_GRAY2BGR
            ) if len(image.shape) == 3 else self.apply_histogram_equalization(image),
            "CLAHE": cv2.cvtColor(
                self.apply_clahe(image), cv2.COLOR_GRAY2BGR
            ) if len(image.shape) == 3 else self.apply_clahe(image),
            "Gamma 0.4": self.apply_gamma_correction(image, 0.4),
            "Gamma 2.5": self.apply_gamma_correction(image, 2.5),
            "Contrast Stretch": self.apply_contrast_stretching(image),
        }
        for name, img in transforms.items():
            results[f"{name}"] = img
            results[f"Hist: {name}"] = self.draw_histogram_image(img, draw_cdf=False)

        return results

    # =====================================================================
    # 12. CÁC PHƯƠNG PHÁP PHÁT HIỆN BIÊN NÂNG CAO
    #     (ADVANCED EDGE DETECTION METHODS)
    # =====================================================================

    def detect_edges_prewitt(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Toán tử Prewitt (Prewitt Operator, 1970).

        Kernel Prewitt X:          Kernel Prewitt Y:
        [-1  0  1]                 [-1 -1 -1]
        [-1  0  1]                 [ 0  0  0]
        [-1  0  1]                 [ 1  1  1]

        So với Sobel: Prewitt dùng trọng số đều (không nhấn mạnh pixel trung tâm),
        đơn giản hơn nhưng nhạy nhiễu hơn Sobel.
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)

        prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        magnitude = cv2.magnitude(prewitt_x, prewitt_y)

        return {
            "prewitt_x": cv2.convertScaleAbs(prewitt_x),
            "prewitt_y": cv2.convertScaleAbs(prewitt_y),
            "magnitude": cv2.convertScaleAbs(magnitude),
        }

    def detect_edges_roberts(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Toán tử Roberts Cross (Roberts, 1965).

        Kernel Roberts 1:     Kernel Roberts 2:
        [1   0]               [0   1]
        [0  -1]               [-1  0]

        Đặc điểm:
        - Kernel 2x2 → đơn giản nhất trong các toán tử biên.
        - Phát hiện cạnh chéo (diagonal edges) tốt.
        - Rất nhạy với nhiễu → nên kết hợp với Gaussian Blur.
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        kernel_1 = np.array([[1, 0], [0, -1]], dtype=np.float64)
        kernel_2 = np.array([[0, 1], [-1, 0]], dtype=np.float64)

        roberts_1 = cv2.filter2D(gray, cv2.CV_64F, kernel_1)
        roberts_2 = cv2.filter2D(gray, cv2.CV_64F, kernel_2)
        magnitude = cv2.magnitude(roberts_1, roberts_2)

        return {
            "roberts_cross_1": cv2.convertScaleAbs(roberts_1),
            "roberts_cross_2": cv2.convertScaleAbs(roberts_2),
            "magnitude": cv2.convertScaleAbs(magnitude),
        }

    def detect_edges_scharr(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Toán tử Scharr (Scharr Operator).

        Kernel Scharr X:          Kernel Scharr Y:
        [-3   0   3]              [-3  -10  -3]
        [-10  0  10]              [ 0    0   0]
        [-3   0   3]              [ 3   10   3]

        So với Sobel 3x3: Scharr cho xấp xỉ gradient chính xác hơn,
        đặc biệt với các cạnh có hướng nghiêng (rotational symmetry tốt hơn).
        OpenCV khuyến nghị dùng Scharr thay Sobel khi ksize=3.
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        magnitude = cv2.magnitude(scharr_x, scharr_y)

        return {
            "scharr_x": cv2.convertScaleAbs(scharr_x),
            "scharr_y": cv2.convertScaleAbs(scharr_y),
            "magnitude": cv2.convertScaleAbs(magnitude),
        }

    def detect_edges_log(self, image: np.ndarray,
                         sigma: float = 1.4,
                         ksize: int = 9) -> np.ndarray:
        """
        Laplacian of Gaussian (LoG) — Marr-Hildreth Edge Detector (1980).

        Pipeline:
        1. Gaussian Blur (σ) → loại nhiễu trước.
        2. Laplacian (đạo hàm bậc 2) → tìm điểm zero-crossing.

        LoG kết hợp khử nhiễu + phát hiện cạnh trong một bước.
        σ lớn → phát hiện cạnh thô, σ nhỏ → phát hiện cạnh mịn.
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        return cv2.convertScaleAbs(log)

    def detect_edges_auto_canny(self, image: np.ndarray,
                                sigma: float = 0.33) -> np.ndarray:
        """
        Canny tự động ngưỡng (Zero-parameter Automatic Canny).

        Tính ngưỡng dựa trên giá trị median của ảnh:
            low  = (1 - σ) × median
            high = (1 + σ) × median

        Ưu điểm: Tự thích nghi với điều kiện ánh sáng khác nhau,
        không cần chỉnh tay ngưỡng cho mỗi loại ảnh.

        Tham khảo: Adrian Rosebrock, "Zero-parameter automatic Canny edge detection"
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        v = np.median(blurred)
        low = int(max(0, (1.0 - sigma) * v))
        high = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(blurred, low, high)

    def detect_edges_zero_crossing(self, image: np.ndarray,
                                   sigma: float = 1.5) -> np.ndarray:
        """
        Phát hiện biên bằng Zero-Crossing trên LoG.

        Biên = vị trí đạo hàm bậc 2 (Laplacian) đổi dấu (qua 0).
        1. Tính LoG (Laplacian of Gaussian).
        2. Tìm các pixel mà LoG chuyển từ dương sang âm hoặc ngược lại.
        3. Đánh dấu đó là cạnh.
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        log = cv2.Laplacian(blurred, cv2.CV_64F)

        # Tìm zero-crossing
        rows, cols = log.shape
        zero_crossing = np.zeros_like(gray)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                neighbors = [
                    log[i - 1, j], log[i + 1, j],
                    log[i, j - 1], log[i, j + 1],
                    log[i - 1, j - 1], log[i - 1, j + 1],
                    log[i + 1, j - 1], log[i + 1, j + 1],
                ]
                pos = sum(1 for n in neighbors if n > 0)
                neg = sum(1 for n in neighbors if n < 0)
                if pos > 0 and neg > 0:
                    zero_crossing[i, j] = 255

        return zero_crossing

    def compare_edge_methods(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        So sánh tất cả phương pháp phát hiện biên trên cùng một ảnh.
        Trả về dictionary để dùng với create_comparison_grid().
        """
        return {
            "Original": self.convert_grayscale(image) if len(image.shape) == 3 else image,
            "Canny (50,150)": self.detect_edges_canny(image, 50, 150),
            "Auto Canny": self.detect_edges_auto_canny(image),
            "Sobel Magnitude": self.detect_edges_sobel(image)["magnitude"],
            "Scharr Magnitude": self.detect_edges_scharr(image)["magnitude"],
            "Prewitt Magnitude": self.detect_edges_prewitt(image)["magnitude"],
            "Roberts Magnitude": self.detect_edges_roberts(image)["magnitude"],
            "Laplacian": self.detect_edges_laplacian(image),
            "LoG (sigma=1.4)": self.detect_edges_log(image, sigma=1.4),
        }

    # =====================================================================
    # 13. PHÂN VÙNG ẢNH (IMAGE SEGMENTATION)
    # =====================================================================

    def segment_otsu(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Phân ngưỡng Otsu (Otsu's Thresholding, 1979).

        Tự động tìm ngưỡng tối ưu T* bằng cách tối đa hóa phương sai liên lớp
        (between-class variance σ²_B):
            σ²_B(T) = ω₀(T) × ω₁(T) × [μ₀(T) - μ₁(T)]²

        Trong đó ω₀, ω₁ là trọng số lớp, μ₀, μ₁ là mean lớp.

        Giả định: Ảnh có histogram hai đỉnh (bimodal distribution).
        Kết quả: Ảnh nhị phân — nền (0) và đối tượng (255).
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        threshold_val, binary = cv2.threshold(gray, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Otsu nghịch đảo
        _, binary_inv = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return {
            "grayscale": gray,
            "otsu_binary": binary,
            "otsu_binary_inv": binary_inv,
            "threshold_value": threshold_val,
        }

    def segment_adaptive_threshold(self, image: np.ndarray,
                                   method: str = "gaussian",
                                   block_size: int = 11,
                                   C: int = 2) -> Dict[str, np.ndarray]:
        """
        Phân ngưỡng thích nghi (Adaptive Thresholding).

        Thay vì 1 ngưỡng toàn cục, tính ngưỡng riêng cho từng vùng cục bộ:
            T(x, y) = mean/gaussian_weighted_mean của vùng lân cận (block_size) - C

        Phương pháp:
        - 'mean':     Trung bình cộng các pixel trong vùng lân cận.
        - 'gaussian': Trung bình có trọng số Gaussian (quan trọng hơn pixel trung tâm).

        Ưu điểm so với Otsu: Hoạt động tốt khi ánh sáng không đều (shadows, gradients).
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image

        if block_size % 2 == 0:
            block_size += 1

        if method == "mean":
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
        else:
            adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

        binary = cv2.adaptiveThreshold(
            gray, 255, adaptive_method,
            cv2.THRESH_BINARY, block_size, C
        )
        binary_inv = cv2.adaptiveThreshold(
            gray, 255, adaptive_method,
            cv2.THRESH_BINARY_INV, block_size, C
        )

        return {
            "grayscale": gray,
            "adaptive_binary": binary,
            "adaptive_binary_inv": binary_inv,
        }

    def segment_multi_threshold(self, image: np.ndarray,
                                thresholds: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Phân ngưỡng đa mức (Multi-level Thresholding).

        Chia ảnh thành nhiều vùng dựa trên danh sách ngưỡng.
        Ví dụ thresholds=[85, 170]: chia thành 3 vùng (0-84, 85-169, 170-255).
        Mỗi vùng được gán 1 mức xám đại diện.
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image

        if thresholds is None:
            thresholds = [85, 170]

        thresholds = sorted(thresholds)
        result = np.zeros_like(gray)

        # Tính giá trị đại diện cho mỗi vùng
        boundaries = [0] + thresholds + [256]
        n_regions = len(boundaries) - 1
        for i in range(n_regions):
            low = boundaries[i]
            high = boundaries[i + 1]
            value = int(255 * i / max(n_regions - 1, 1))
            mask = (gray >= low) & (gray < high)
            result[mask] = value

        return {
            "grayscale": gray,
            "multi_threshold": result,
            "thresholds": thresholds,
        }

    def segment_watershed(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Phân vùng Watershed (Phương pháp Đường phân thủy).

        Nguyên lý: Coi ảnh gradient như bề mặt địa hình (topographic surface).
        1. Tìm "hạt giống" (markers) = vùng chắc chắn foreground & background.
        2. "Ngập nước" từ các markers → ranh giới giữa các lưu vực = biên phân vùng.

        Pipeline:
        - Otsu → Morphology (Opening) → Distance Transform → Markers → Watershed.
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        img_color = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Bước 1: Otsu + Morphology để tách foreground thô
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel_medium, iterations=2)

        # Bước 2: Xác định vùng chắc chắn là background (dilation)
        sure_bg = cv2.dilate(opening, self.kernel_large, iterations=3)

        # Bước 3: Distance Transform → vùng chắc chắn foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        # Bước 4: Vùng không chắc chắn = background - foreground
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Bước 5: Gán markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Bước 6: Áp dụng Watershed
        markers_result = cv2.watershed(img_color, markers.copy())

        # Tạo ảnh kết quả: ranh giới = đỏ
        result = img_color.copy()
        result[markers_result == -1] = [0, 0, 255]

        # Tạo colormap cho các vùng phân biệt
        n_labels = markers_result.max() + 2
        colormap = np.random.randint(50, 255, size=(n_labels, 3), dtype=np.uint8)
        colormap[0] = [0, 0, 0]  # Background = đen
        colored_segments = colormap[markers_result + 1]

        return {
            "binary_otsu": binary,
            "distance_transform": cv2.normalize(dist_transform, None, 0, 255,
                                                cv2.NORM_MINMAX).astype(np.uint8),
            "sure_foreground": sure_fg,
            "sure_background": sure_bg,
            "watershed_boundaries": result,
            "colored_segments": colored_segments.astype(np.uint8),
        }

    def segment_kmeans_color(self, image: np.ndarray,
                             k: int = 4,
                             max_iterations: int = 100) -> Dict[str, np.ndarray]:
        """
        Phân vùng theo màu sắc bằng K-Means Clustering.

        Thuật toán K-Means:
        1. Chọn ngẫu nhiên K tâm cụm (centroids) trong không gian màu.
        2. Gán mỗi pixel vào cụm có tâm gần nhất (Euclidean distance).
        3. Cập nhật tâm cụm = trung bình các pixel trong cụm.
        4. Lặp lại bước 2-3 cho đến khi hội tụ.

        Kết quả: Ảnh được "nén" còn K màu đại diện.
        K nhỏ = ít chi tiết, K lớn = nhiều chi tiết.
        """
        # Reshape ảnh thành ma trận (N x 3) — mỗi hàng là 1 pixel BGR
        pixel_values = image.reshape((-1, 3)).astype(np.float32)

        # Điều kiện dừng: 100 iterations hoặc epsilon = 1.0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    max_iterations, 1.0)

        # K-Means
        _, labels, centers = cv2.kmeans(
            pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        # Tái tạo ảnh từ các tâm cụm
        centers = centers.astype(np.uint8)
        segmented = centers[labels.flatten()].reshape(image.shape)

        # Tạo mask cho từng cụm
        label_map = labels.reshape(image.shape[:2])
        cluster_masks = {}
        for i in range(k):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[label_map == i] = 255
            cluster_masks[f"Cluster_{i+1}"] = mask

        return {
            "segmented": segmented,
            "label_map": (label_map * (255 // max(k - 1, 1))).astype(np.uint8),
            "centers": centers,
            "cluster_masks": cluster_masks,
        }

    def segment_grabcut(self, image: np.ndarray,
                        rect: Optional[Tuple[int, int, int, int]] = None,
                        iterations: int = 5) -> Dict[str, np.ndarray]:
        """
        Phân vùng GrabCut (Rother et al., 2004).

        Kết hợp graph cuts + Gaussian Mixture Models (GMM):
        1. Khởi tạo: Người dùng chỉ định ROI chứa đối tượng (rectangle).
        2. Xây dựng mô hình GMM cho foreground và background.
        3. Graph Cut trên đồ thị pixel → phân loại FG/BG.
        4. Lặp lại để tinh chỉnh ranh giới.

        Nếu không cung cấp rect, sử dụng vùng trung tâm ảnh.
        """
        h, w = image.shape[:2]

        if rect is None:
            # Mặc định: vùng trung tâm 60% ảnh
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(image, mask, rect, bgd_model, fgd_model,
                    iterations, cv2.GC_INIT_WITH_RECT)

        # Mask kết quả: GC_FGD(1) và GC_PR_FGD(3) = foreground
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # Ảnh kết quả
        result = image.copy()
        result[fg_mask == 0] = 0  # Background = đen

        # Vẽ ROI rect lên ảnh
        roi_vis = image.copy()
        cv2.rectangle(roi_vis, (rect[0], rect[1]),
                      (rect[0] + rect[2], rect[1] + rect[3]),
                      (0, 255, 0), 2)

        return {
            "roi_visualization": roi_vis,
            "foreground_mask": fg_mask,
            "result": result,
        }

    def segment_mean_shift(self, image: np.ndarray,
                           sp: int = 20, sr: int = 40,
                           max_level: int = 1) -> np.ndarray:
        """
        Phân vùng Mean Shift (Lọc Mean Shift).

        Thuật toán dịch chuyển trung bình trong không gian (x, y, color):
        1. Với mỗi pixel, xét cửa sổ bán kính sp (spatial) và sr (color range).
        2. Tính trung bình có trọng số của tất cả pixel trong cửa sổ.
        3. Dịch chuyển tâm cửa sổ đến vị trí trung bình mới.
        4. Lặp lại cho đến khi hội tụ.

        sp: Bán kính không gian (spatial radius) — sp lớn → vùng mịn hơn.
        sr: Bán kính màu (color radius) — sr lớn → trộn nhiều màu hơn.
        """
        return cv2.pyrMeanShiftFiltering(image, sp, sr, maxLevel=max_level)

    def segment_region_growing(self, image: np.ndarray,
                               seed_point: Tuple[int, int],
                               threshold: int = 15) -> np.ndarray:
        """
        Phân vùng phát triển vùng (Region Growing).

        Thuật toán:
        1. Bắt đầu từ 1 pixel hạt giống (seed).
        2. Kiểm tra các pixel lân cận:
           Nếu |I(neighbor) - I(seed_region_mean)| < threshold → thêm vào vùng.
        3. Cập nhật mean của vùng và tiếp tục mở rộng.
        4. Dừng khi không còn pixel nào thỏa mãn.

        Đơn giản, trực quan, nhưng nhạy cảm với vị trí hạt giống
        và ngưỡng threshold.
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        h, w = gray.shape
        visited = np.zeros((h, w), dtype=bool)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Khởi tạo
        seed_y, seed_x = seed_point
        if not (0 <= seed_y < h and 0 <= seed_x < w):
            return mask

        seed_val = float(gray[seed_y, seed_x])
        region_sum = seed_val
        region_count = 1

        # BFS (Breadth-First Search) để mở rộng vùng
        from collections import deque
        queue = deque()
        queue.append((seed_y, seed_x))
        visited[seed_y, seed_x] = True
        mask[seed_y, seed_x] = 255

        neighbors_offset = [(-1, 0), (1, 0), (0, -1), (0, 1),
                            (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while queue:
            cy, cx = queue.popleft()
            region_mean = region_sum / region_count

            for dy, dx in neighbors_offset:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    pixel_val = float(gray[ny, nx])
                    if abs(pixel_val - region_mean) < threshold:
                        mask[ny, nx] = 255
                        queue.append((ny, nx))
                        region_sum += pixel_val
                        region_count += 1

        return mask

    def demonstrate_segmentation_methods(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Trình diễn và so sánh các phương pháp phân vùng trên cùng một ảnh.
        """
        results = {"Original": image.copy()}

        # Otsu
        otsu = self.segment_otsu(image)
        results["Otsu Binary"] = otsu["otsu_binary"]

        # Adaptive
        adaptive = self.segment_adaptive_threshold(image, "gaussian", 11, 2)
        results["Adaptive Gaussian"] = adaptive["adaptive_binary"]

        # Multi-threshold
        multi = self.segment_multi_threshold(image, [85, 170])
        results["Multi-Threshold"] = multi["multi_threshold"]

        # K-Means
        if len(image.shape) == 3:
            kmeans = self.segment_kmeans_color(image, k=4)
            results["K-Means (K=4)"] = kmeans["segmented"]

        # Mean Shift
        if len(image.shape) == 3:
            results["Mean Shift"] = self.segment_mean_shift(image, sp=20, sr=40)

        # Watershed
        watershed = self.segment_watershed(image)
        results["Watershed"] = watershed["watershed_boundaries"]

        return results

    # =====================================================================
    # 14. NÉN DỮ LIỆU ẢNH (IMAGE COMPRESSION)
    # =====================================================================

    def compress_jpeg(self, image: np.ndarray,
                      quality: int = 90) -> Dict[str, Any]:
        """
        Nén JPEG (Joint Photographic Experts Group).

        Nguyên lý: Nén có tổn hao (lossy compression).
        Pipeline JPEG:
        1. Chuyển RGB → YCbCr (tách luminance và chrominance).
        2. Chia ảnh thành các block 8×8.
        3. DCT (Discrete Cosine Transform) trên mỗi block.
        4. Lượng tử hóa (Quantization) — bước gây mất dữ liệu.
        5. Mã hóa Huffman (entropy coding).

        quality: Chất lượng nén (1-100). Thấp = file nhỏ, nhiều artifact.
        """
        # Nén vào bộ nhớ
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        compressed_bytes = encoded.tobytes()
        compressed_size = len(compressed_bytes)

        # Giải nén lại để xem chất lượng
        decoded = cv2.imdecode(np.frombuffer(compressed_bytes, np.uint8),
                               cv2.IMREAD_COLOR)

        # Tính kích thước gốc
        original_size = image.shape[0] * image.shape[1] * image.shape[2] if len(image.shape) == 3 \
            else image.shape[0] * image.shape[1]

        # Tính sai số (PSNR, MSE)
        mse = np.mean((image.astype(np.float64) - decoded.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-10)) if mse > 0 else float('inf')

        return {
            "compressed_image": decoded,
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": round(original_size / max(compressed_size, 1), 2),
            "quality": quality,
            "psnr_db": round(psnr, 2),
            "mse": round(mse, 4),
        }

    def compress_png(self, image: np.ndarray,
                     compression_level: int = 3) -> Dict[str, Any]:
        """
        Nén PNG (Portable Network Graphics).

        Nguyên lý: Nén không tổn hao (lossless compression).
        Pipeline PNG:
        1. Filtering: Dự đoán pixel dựa trên láng giềng (Sub, Up, Average, Paeth).
        2. DEFLATE compression (LZ77 + Huffman).

        compression_level: Mức nén (0-9).
        0 = không nén (nhanh nhất), 9 = nén tối đa (chậm nhất).
        Chất lượng ảnh không thay đổi (lossless).
        """
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), compression_level]
        _, encoded = cv2.imencode('.png', image, encode_param)
        compressed_bytes = encoded.tobytes()
        compressed_size = len(compressed_bytes)

        original_size = image.shape[0] * image.shape[1] * image.shape[2] if len(image.shape) == 3 \
            else image.shape[0] * image.shape[1]

        return {
            "compressed_image": image.copy(),  # Lossless → ảnh không đổi
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": round(original_size / max(compressed_size, 1), 2),
            "compression_level": compression_level,
            "psnr_db": float('inf'),  # Lossless → PSNR vô hạn
            "mse": 0.0,
        }

    def compress_dct_educational(self, image: np.ndarray,
                                 quality_factor: int = 50) -> Dict[str, np.ndarray]:
        """
        Nén dựa trên DCT (Discrete Cosine Transform) — Phiên bản giáo dục.

        Mô phỏng bước quan trọng nhất trong JPEG:
        1. Chia ảnh xám thành các block 8×8.
        2. Áp dụng DCT 2D trên mỗi block.
        3. Lượng tử hóa: Chia hệ số DCT cho ma trận lượng tử hóa Q.
           → Các hệ số tần số cao (chi tiết) bị làm tròn về 0.
        4. Giải lượng tử hóa + IDCT → tái tạo ảnh.

        quality_factor: Hệ số chất lượng (1-100).
        Thấp = nén mạnh, nhiều artifact block.
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image.copy()
        h, w = gray.shape

        # Padding để chia hết cho 8
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
        padded = np.pad(gray, ((0, h_pad), (0, w_pad)), mode='reflect')
        ph, pw = padded.shape

        # Ma trận lượng tử hóa chuẩn JPEG (luminance)
        Q_base = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ], dtype=np.float64)

        # Điều chỉnh Q theo quality_factor
        if quality_factor < 50:
            scale = 5000.0 / max(quality_factor, 1)
        else:
            scale = 200.0 - 2.0 * quality_factor
        Q = np.clip(np.floor((Q_base * scale + 50) / 100.0), 1, 255)

        # Nén: DCT → Quantize
        img_float = padded.astype(np.float64) - 128.0  # Dịch về [-128, 127]
        compressed = np.zeros_like(img_float)
        dct_coeffs_vis = np.zeros_like(img_float)

        for i in range(0, ph, 8):
            for j in range(0, pw, 8):
                block = img_float[i:i + 8, j:j + 8]
                # DCT 2D
                dct_block = cv2.dct(block)
                dct_coeffs_vis[i:i + 8, j:j + 8] = dct_block
                # Lượng tử hóa
                quantized = np.round(dct_block / Q)
                # Giải lượng tử hóa
                dequantized = quantized * Q
                # IDCT
                idct_block = cv2.idct(dequantized)
                compressed[i:i + 8, j:j + 8] = idct_block

        # Khôi phục
        result = np.clip(compressed + 128.0, 0, 255).astype(np.uint8)
        result = result[:h, :w]

        # Hiển thị hệ số DCT (log scale)
        dct_vis = np.log(np.abs(dct_coeffs_vis) + 1)
        dct_vis = cv2.normalize(dct_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dct_vis = dct_vis[:h, :w]

        # Tính sai số
        diff = cv2.absdiff(gray, result)

        return {
            "original_gray": gray,
            "dct_coefficients": dct_vis,
            "compressed_result": result,
            "error_map": diff,
            "quality_factor": quality_factor,
        }

    def compress_svd(self, image: np.ndarray,
                     k: int = 50) -> Dict[str, Any]:
        """
        Nén ảnh bằng SVD (Singular Value Decomposition).

        Phân tích: A = U × Σ × V^T
        - U (m×m): Ma trận trực giao trái.
        - Σ (m×n): Ma trận đường chéo chứa singular values.
        - V^T (n×n): Ma trận trực giao phải.

        Nén hạng k: Giữ lại k singular values lớn nhất.
            A_k = U[:, :k] × Σ[:k, :k] × V^T[:k, :]

        k nhỏ = nén mạnh (mất chi tiết), k lớn = ít mất mát.
        Tỷ lệ nén ≈ k(m + n + 1) / (m × n).
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        h, w = gray.shape
        img_float = gray.astype(np.float64)

        # SVD
        U, S, Vt = np.linalg.svd(img_float, full_matrices=False)

        # Giới hạn k
        k = min(k, min(h, w))

        # Tái tạo với hạng k
        reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        result = np.clip(reconstructed, 0, 255).astype(np.uint8)

        # Tính kích thước lưu trữ
        original_size = h * w
        compressed_size = k * (h + w + 1)
        compression_ratio = original_size / max(compressed_size, 1)

        # Sai số
        mse = np.mean((img_float - reconstructed) ** 2)
        psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-10)) if mse > 0 else float('inf')

        # Biểu đồ singular values (vẽ trên ảnh)
        sv_plot = np.zeros((200, 300, 3), dtype=np.uint8)
        sv_plot[:] = (30, 30, 30)
        n_plot = min(len(S), 100)
        s_norm = S[:n_plot] / max(S[0], 1) * 170
        for i in range(1, n_plot):
            pt1 = (int((i - 1) * 290 / n_plot) + 5, 190 - int(s_norm[i - 1]))
            pt2 = (int(i * 290 / n_plot) + 5, 190 - int(s_norm[i]))
            cv2.line(sv_plot, pt1, pt2, (0, 200, 255), 1, cv2.LINE_AA)
        # Đánh dấu k
        k_x = int(min(k, n_plot - 1) * 290 / n_plot) + 5
        cv2.line(sv_plot, (k_x, 10), (k_x, 190), (0, 0, 255), 1)
        cv2.putText(sv_plot, f"k={k}", (k_x + 3, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(sv_plot, "Singular Values", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1, cv2.LINE_AA)

        return {
            "original_gray": gray,
            "compressed_result": result,
            "error_map": cv2.absdiff(gray, result),
            "singular_values_plot": sv_plot,
            "k": k,
            "compression_ratio": round(compression_ratio, 2),
            "psnr_db": round(psnr, 2),
            "mse": round(mse, 4),
        }

    def compress_rle(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Nén Run-Length Encoding (RLE).

        Nguyên lý: Thay chuỗi pixel lặp liên tiếp bằng (giá_trị, số_lần_lặp).
        Ví dụ: [255, 255, 255, 0, 0] → [(255, 3), (0, 2)]

        Hiệu quả với ảnh nhị phân, ảnh có vùng lớn đồng nhất.
        Không hiệu quả với ảnh tự nhiên có nhiều biến đổi.
        Đây là nén không tổn hao (lossless).
        """
        gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
        flat = gray.flatten()
        original_size = len(flat)

        # Encode RLE
        runs = []
        current_val = flat[0]
        count = 1
        for i in range(1, len(flat)):
            if flat[i] == current_val and count < 255:
                count += 1
            else:
                runs.append((int(current_val), count))
                current_val = flat[i]
                count = 1
        runs.append((int(current_val), count))

        # Kích thước nén = 2 bytes mỗi run (value + count)
        compressed_size = len(runs) * 2

        # Decode để xác minh
        decoded = []
        for val, cnt in runs:
            decoded.extend([val] * cnt)
        decoded_img = np.array(decoded[:original_size], dtype=np.uint8).reshape(gray.shape)

        return {
            "original_gray": gray,
            "decoded_image": decoded_img,
            "num_runs": len(runs),
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": round(original_size / max(compressed_size, 1), 2),
            "is_lossless": np.array_equal(gray, decoded_img),
        }

    def analyze_compression_methods(self, image: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        So sánh tất cả phương pháp nén trên cùng một ảnh.

        Trả về bảng so sánh: Tỷ lệ nén, PSNR, MSE, loại nén (lossy/lossless).
        """
        results = {}

        # JPEG với các mức chất lượng
        for q in [10, 30, 50, 70, 90]:
            jpeg_result = self.compress_jpeg(image, quality=q)
            results[f"JPEG (Q={q})"] = {
                "type": "Lossy",
                "compression_ratio": jpeg_result["compression_ratio"],
                "psnr_db": jpeg_result["psnr_db"],
                "mse": jpeg_result["mse"],
                "compressed_image": jpeg_result["compressed_image"],
            }

        # PNG
        png_result = self.compress_png(image, compression_level=6)
        results["PNG (Level 6)"] = {
            "type": "Lossless",
            "compression_ratio": png_result["compression_ratio"],
            "psnr_db": png_result["psnr_db"],
            "mse": png_result["mse"],
            "compressed_image": png_result["compressed_image"],
        }

        # DCT (giáo dục)
        for qf in [10, 50, 90]:
            dct_result = self.compress_dct_educational(image, quality_factor=qf)
            gray = self.convert_grayscale(image) if len(image.shape) == 3 else image
            mse_val = np.mean((gray.astype(float) - dct_result["compressed_result"].astype(float)) ** 2)
            psnr_val = 10 * np.log10(255.0 ** 2 / max(mse_val, 1e-10)) if mse_val > 0 else float('inf')
            results[f"DCT (QF={qf})"] = {
                "type": "Lossy",
                "compression_ratio": f"~QF={qf}",
                "psnr_db": round(psnr_val, 2),
                "mse": round(mse_val, 4),
                "compressed_image": dct_result["compressed_result"],
            }

        # SVD
        for k in [10, 30, 50, 100]:
            svd_result = self.compress_svd(image, k=k)
            results[f"SVD (k={k})"] = {
                "type": "Lossy (rank approx)",
                "compression_ratio": svd_result["compression_ratio"],
                "psnr_db": svd_result["psnr_db"],
                "mse": svd_result["mse"],
                "compressed_image": svd_result["compressed_result"],
            }

        # RLE
        rle_result = self.compress_rle(image)
        results["RLE"] = {
            "type": "Lossless",
            "compression_ratio": rle_result["compression_ratio"],
            "psnr_db": float('inf'),
            "mse": 0.0,
            "compressed_image": rle_result["decoded_image"],
        }

        return results

    # =====================================================================
    # 15. XỬ LÝ ẢNH NHIỄU (NOISE PROCESSING)
    # =====================================================================

    def add_gaussian_noise(self, image: np.ndarray,
                           mean: float = 0.0,
                           sigma: float = 25.0) -> np.ndarray:
        """
        Thêm nhiễu Gaussian (Gaussian Noise) vào ảnh.

        Mô hình: I_noisy(x,y) = I(x,y) + N(mean, sigma²)
        N là biến ngẫu nhiên phân phối chuẩn.
        sigma lớn → nhiễu mạnh hơn.
        Đây là loại nhiễu phổ biến nhất trong ảnh số (sensor noise, electronic noise).
        """
        noise = np.random.normal(mean, sigma, image.shape)
        noisy = image.astype(np.float64) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_salt_pepper_noise(self, image: np.ndarray,
                              amount: float = 0.02,
                              salt_ratio: float = 0.5) -> np.ndarray:
        """
        Thêm nhiễu Muối-Tiêu (Salt-and-Pepper / Impulse Noise).

        Một tỷ lệ pixel ngẫu nhiên bị thay thế bằng giá trị cực đại (255 = muối)
        hoặc cực tiểu (0 = tiêu).
        amount: Tỷ lệ pixel bị nhiễu (0.0 – 1.0).
        salt_ratio: Tỷ lệ muối / tổng nhiễu.
        Nguyên nhân thực tế: Lỗi truyền dữ liệu, pixel chết trên sensor.
        """
        out = image.copy()
        total_pixels = image.size if len(image.shape) == 2 else image.shape[0] * image.shape[1]

        # Salt (trắng)
        n_salt = int(total_pixels * amount * salt_ratio)
        salt_coords = (
            np.random.randint(0, image.shape[0], n_salt),
            np.random.randint(0, image.shape[1], n_salt),
        )
        out[salt_coords] = 255

        # Pepper (đen)
        n_pepper = int(total_pixels * amount * (1.0 - salt_ratio))
        pepper_coords = (
            np.random.randint(0, image.shape[0], n_pepper),
            np.random.randint(0, image.shape[1], n_pepper),
        )
        out[pepper_coords] = 0

        return out

    def add_poisson_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Thêm nhiễu Poisson (Shot Noise / Photon Noise).

        Mô hình: Số photon đến mỗi pixel tuân theo phân phối Poisson.
        Nhiễu phụ thuộc vào tín hiệu (signal-dependent noise):
        vùng sáng → nhiễu lớn hơn, vùng tối → nhiễu nhỏ hơn.
        Phổ biến trong ảnh y tế (X-ray, CT), thiên văn, ảnh thiếu sáng.
        """
        img_float = image.astype(np.float64) / 255.0
        # Nhân lên để Poisson sampling có ý nghĩa, rồi chuẩn hóa lại
        noisy = np.random.poisson(img_float * 255.0) / 255.0
        return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)

    def add_speckle_noise(self, image: np.ndarray,
                          sigma: float = 0.15) -> np.ndarray:
        """
        Thêm nhiễu Speckle (Multiplicative Noise).

        Mô hình: I_noisy = I + I × N(0, sigma²)
        Nhiễu nhân tính: cường độ nhiễu tỷ lệ với cường độ pixel.
        Phổ biến trong ảnh SAR (radar), siêu âm (ultrasound), ảnh laser.
        """
        noise = np.random.normal(0, sigma, image.shape)
        noisy = image.astype(np.float64) + image.astype(np.float64) * noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_uniform_noise(self, image: np.ndarray,
                          low: float = -30.0,
                          high: float = 30.0) -> np.ndarray:
        """
        Thêm nhiễu Đều (Uniform Noise).

        Mô hình: I_noisy(x,y) = I(x,y) + U(low, high)
        U là biến ngẫu nhiên phân phối đều trên [low, high].
        Ít phổ biến hơn Gaussian trong thực tế, nhưng hữu ích
        cho mục đích giảng dạy và kiểm thử thuật toán lọc.
        """
        noise = np.random.uniform(low, high, image.shape)
        noisy = image.astype(np.float64) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def compute_psnr_mse(self, clean: np.ndarray, test_image: np.ndarray) -> Dict[str, float]:
        """
        Tính PSNR (dB) và MSE giữa ảnh gốc và ảnh so sánh.
        """
        c = clean.copy()
        t = test_image.copy()
        if c.shape != t.shape:
            if len(c.shape) == 2 and len(t.shape) == 3:
                c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
            elif len(c.shape) == 3 and len(t.shape) == 2:
                t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        mse = float(np.mean((c.astype(np.float64) - t.astype(np.float64)) ** 2))
        psnr = 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-10)) if mse > 0 else float('inf')
        return {"psnr": round(float(psnr), 2), "mse": round(mse, 2)}

    def demonstrate_noise_processing(self, image: np.ndarray,
                                     gaussian_sigma: float = 25.0,
                                     sp_amount: float = 0.02,
                                     speckle_sigma: float = 0.15,
                                     uniform_range: float = 30.0,
                                     denoise_h: float = 10.0
                                     ) -> Dict[str, Dict[str, Any]]:
        """
        Trình diễn thêm nhiễu + khử nhiễu cho từng loại nhiễu.

        Với mỗi loại nhiễu:
        1. Tạo ảnh nhiễu từ ảnh gốc.
        2. Áp dụng bộ lọc khử nhiễu phù hợp nhất.
        3. Tính toán PSNR, MSE để định lượng hiệu quả lọc.
        """
        configs = [
            ("Gaussian", lambda: self.add_gaussian_noise(image, sigma=gaussian_sigma),
             lambda n: self.apply_denoising(n, method="nlm", h=denoise_h), "NLM (Non-Local Means)"),
            ("Salt-and-Pepper", lambda: self.add_salt_pepper_noise(image, amount=sp_amount),
             lambda n: self.apply_denoising(n, method="median", ksize=5), "Median Filter"),
            ("Poisson", lambda: self.add_poisson_noise(image),
             lambda n: self.apply_denoising(n, method="nlm", h=denoise_h), "NLM (Non-Local Means)"),
            ("Speckle", lambda: self.add_speckle_noise(image, sigma=speckle_sigma),
             lambda n: self.apply_denoising(n, method="bilateral"), "Bilateral Filter"),
            ("Uniform", lambda: self.add_uniform_noise(image, low=-uniform_range, high=uniform_range),
             lambda n: self.apply_denoising(n, method="gaussian", ksize=5), "Gaussian Blur"),
        ]

        results = {}
        for name, noise_fn, denoise_fn, filter_name in configs:
            noisy = noise_fn()
            denoised = denoise_fn(noisy)
            m_noisy = self.compute_psnr_mse(image, noisy)
            m_denoised = self.compute_psnr_mse(image, denoised)
            delta_psnr = round(m_denoised["psnr"] - m_noisy["psnr"], 2) if m_noisy["psnr"] != float('inf') and m_denoised["psnr"] != float('inf') else 0.0
            results[name] = {
                "noisy": noisy,
                "denoised": denoised,
                "filter": filter_name,
                "psnr_noisy": m_noisy["psnr"],
                "psnr_denoised": m_denoised["psnr"],
                "mse_noisy": m_noisy["mse"],
                "mse_denoised": m_denoised["mse"],
                "delta_psnr": delta_psnr,
            }

        return results

