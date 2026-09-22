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
