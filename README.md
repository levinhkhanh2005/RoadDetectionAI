# ĐỒ ÁN MÔN HỌC: TRÍ TUỆ NHÂN TẠO (ARTIFICIAL INTELLIGENCE)
## ĐỀ TÀI: XÂY DỰNG HỆ THỐNG NHẬN DIỆN VÀ ĐẾM PHƯƠNG TIỆN GIAO THÔNG ĐƯỜNG BỘ

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-orange)

---

## 📌 Giới thiệu đề tài
Hệ thống giải quyết bài toán thị giác máy tính trong hệ thống giao thông thông minh (Intelligent Transportation Systems - ITS):
- **Phát hiện & Phân loại phương tiện (Vehicle Detection & Classification)**: Nhận diện chính xác 5 nhóm phương tiện đường bộ: **Ô tô con (Car)**, **Xe máy (Motorcycle)**, **Xe buýt (Bus)**, **Xe tải (Truck)**, **Xe đạp (Bicycle)**.
- **Theo dõi đối tượng (Object Tracking)**: Sử dụng giải thuật **ByteTrack** để duy trì định danh (ID) cho từng phương tiện qua chuỗi khung hình video.
- **Đếm xe tự động qua vạch kẻ ảo (Virtual Line Crossing)**: Thống kê lưu lượng phương tiện theo loại và hướng di chuyển mà không bị đếm lặp.
- **Giao diện Demo trực quan**:
  - Giao diện OpenCV thời gian thực (CLI / Webcam / Video).
  - Giao diện Web tương tác hiện đại bằng **Streamlit** phục vụ thuyết trình và chấm điểm đồ án.

---

## 📂 Cấu trúc thư mục dự án

```text
RoadDetectionAI/
├── data/
│   ├── samples/                # Thư mục chứa ảnh và video mẫu giao thông
│   └── output/                 # Thư mục lưu kết quả ảnh/video sau xử lý
├── models/
│   └── yolov8n.pt              # Trọng số mô hình YOLOv8 Nano (tự động tải)
├── src/
│   ├── __init__.py             # Khởi tạo package
│   ├── detector.py             # Lớp VehicleDetector: nạp model, lọc class phương tiện
│   ├── tracker.py              # Lớp VehicleCounter: theo dõi quỹ đạo, thuật toán cắt vạch
│   └── visualizer.py           # Lớp Visualizer: vẽ HUD, Bounding box, vạch đếm, quỹ đạo
├── app_streamlit.py            # Giao diện Web tương tác Dashboard Streamlit
├── main.py                     # File thực thi dòng lệnh / OpenCV Window
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md                   # Tài liệu hướng dẫn & Đề cương báo cáo
```

---

## 🚀 Hướng dẫn cài đặt & Chạy chương trình

### 1. Kích hoạt môi trường ảo & Cài đặt thư viện
Mở Terminal trong PyCharm hoặc PowerShell tại thư mục dự án:
```powershell
# Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Chạy giao diện Web Streamlit (Khuyên dùng để thuyết trình)
```powershell
streamlit run app_streamlit.py
```
Trình duyệt sẽ tự động mở trang web dashboard tại địa chỉ `http://localhost:8501`. Tại đây bạn có thể:
- Kéo thả video hoặc chọn video mẫu có sẵn.
- Tùy chỉnh thanh trượt độ tin cậy (Confidence) và vị trí vạch đếm ảo.
- Xem biểu đồ thống kê lưu lượng xe thời gian thực.

### 3. Chạy bằng mã nguồn chính (OpenCV Window)

- **Nhận diện trên Webcam thời gian thực:**
  ```powershell
  python main.py --source 0
  ```
- **Nhận diện trên Video file:**
  ```powershell
  python main.py --source data/samples/traffic.mp4
  ```
- **Nhận diện trên Video và Lưu file kết quả xuất ra:**
  ```powershell
  python main.py --source data/samples/traffic.mp4 --save
  ```
- **Nhận diện trên Ảnh tĩnh:**
  ```powershell
  python main.py --source data/samples/traffic.jpg --save
  ```

### ⌨️ Các phím tắt khi chạy trên OpenCV:
- `q` hoặc `ESC`: Thoát chương trình.
- `p`: Tạm dừng (Pause) / Tiếp tục video.
- `s`: Chụp ảnh màn hình (Snapshot) và lưu vào thư mục `data/output/`.
- `r`: Khởi tạo lại (Reset) bộ đếm xe về 0.

---

## 📑 ĐỀ CƯƠNG BÁO CÁO ĐỒ ÁN MÔN HỌC (CHUẨN 5 CHƯƠNG)

Dưới đây là khung sườn chi tiết để bạn đưa vào file Word/LaTeX báo cáo nộp giảng viên:

### CHƯƠNG 1: GIỚI THIỆU & ĐẶT VẤN ĐỀ
1.1. Bối cảnh bài toán: Ách tắc giao thông, nhu cầu giám sát giao thông tự động trong đô thị thông minh.  
1.2. Mục tiêu của đề tài: Ứng dụng mô hình học sâu để tự động phát hiện, phân loại và đếm lưu lượng phương tiện.  
1.3. Phạm vi nghiên cứu & Đối tượng thực nghiệm: Các phương tiện đường bộ phổ biến (ô tô con, xe máy, xe buýt, xe tải, xe đạp).  
1.4. Bố cục của đồ án.

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & CÔNG NGHỆ LIÊN QUAN
2.1. Tổng quan về Thị giác máy tính (Computer Vision) và Bài toán Phát hiện đối tượng (Object Detection).  
2.2. Mạng nơ-ron tích chập (Convolutional Neural Networks - CNN).  
2.3. Kiến trúc mô hình YOLO (You Only Look Once):
  - Sự tiến hóa từ YOLOv1 đến YOLOv8.
  - Cơ chế Anchor-free, kiến trúc C2f Backbone và Head tách biệt (Decoupled Head).
  - Hàm mất mát: Task-Aligned Loss, CIoU Loss, Distribution Focal Loss (DFL).  
2.4. Thuật toán theo dõi đối tượng (Multi-Object Tracking - MOT):
  - Giới thiệu giải thuật ByteTrack: Nguyên lý tận dụng low-score detection và bộ lọc Kalman Filter.  
2.5. Thuật toán kiểm tra giao điểm đường thẳng (Line Segment Intersection Algorithm).

### CHƯƠNG 3: THIẾT KẾ VÀ HIỆN THỰC HỆ THỐNG
3.1. Sơ đồ kiến trúc tổng thể (System Architecture Pipeline):
  - Input Video $\to$ Frame Preprocessing $\to$ YOLOv8 Inference $\to$ Class Filtering $\to$ ByteTrack Tracking $\to$ Line Crossing Checking $\to$ Visualizer/HUD Output.  
3.2. Hiện thực module phát hiện (`VehicleDetector`): Lọc 5 lớp phương tiện chuẩn COCO dataset.  
3.3. Hiện thực module đếm (`VehicleCounter`): Quản lý vector quỹ đạo tâm điểm xe, tính toán hướng di chuyển (In/Out) bằng tích có hướng.  
3.4. Hiện thực module hiển thị (`Visualizer`): Vẽ bounding box đa màu, bảng điều khiển HUD trong suốt (Glassmorphism), hiển thị FPS.  
3.5. Xây dựng giao diện tương tác người dùng bằng Streamlit.

### CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ
4.1. Môi trường thử nghiệm phần cứng và phần mềm (CPU, GPU, RAM, Python version).  
4.2. Dữ liệu thử nghiệm: Video giao thông thực tế tại các tuyến đường đô thị.  
4.3. Kết quả đánh giá:
  - Độ chính xác phát hiện và phân loại các dòng xe.
  - Tốc độ xử lý khung hình (FPS) theo thời gian thực.
  - Độ tin cậy của bộ đếm qua vạch (tỷ lệ đếm đúng so với đếm thủ công).  
4.4. Phân tích các trường hợp sai lệch (xe bị che khuất một phần, điều kiện ánh sáng yếu, xe di chuyển quá sát nhau).

### CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
5.1. Các kết quả đã đạt được của đồ án.  
5.2. Hạn chế của hệ thống.  
5.3. Hướng phát triển trong tương lai:
  - Tích hợp mô hình nhận diện biển số xe (ANPR).
  - Cảnh báo phương tiện vi phạm đi sai làn, vượt đèn đỏ, không đội mũ bảo hiểm.
  - Tối ưu hóa mô hình sang định dạng TensorRT / ONNX để nhúng lên thiết bị biên (Edge AI: Jetson Nano, Raspberry Pi).
