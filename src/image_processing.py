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
            src_points = np.float32([
                [int(w * 0.15), int(h * 0.95)],    # Góc dưới trái
                [int(w * 0.85), int(h * 0.95)],    # Góc dưới phải
                [int(w * 0.65), int(h * 0.55)],    # Góc trên phải
                [int(w * 0.35), int(h * 0.55)],    # Góc trên trái
            ])

        if dst_points is None:
            dst_points = np.float32([
                [int(w * 0.2), h],
                [int(w * 0.8), h],
                [int(w * 0.8), 0],
                [int(w * 0.2), 0],
            ])

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
