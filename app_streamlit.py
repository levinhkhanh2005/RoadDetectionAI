import os
import glob
import time
import threading
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from src.detector import VehicleDetector, VEHICLE_COLORS, CLASS_NAMES_VI
from src.tracker import VehicleCounter
from src.visualizer import Visualizer
from src.image_processing import ImageProcessor
from src.video_processing import cleanup_temp_video, get_video_metadata, iter_video_frames, save_uploaded_video


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_model_path(model_name: str) -> str:
    """Resolve model weights relative to the project, regardless of cwd."""
    candidates = [
        PROJECT_ROOT / "models" / model_name,
        PROJECT_ROOT / "runs" / "detect" / "traffic_model-2" / "weights" / model_name,
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return str(candidates[0])


# Cấu hình trang Streamlit
st.set_page_config(
    page_title="AI Road Vehicle Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling cho phần nhận diện thương hiệu; màu sắc và widget dùng theme Streamlit.
st.markdown("""
<style>
    .app-hero {
        padding: 0.35rem 0 1.1rem;
    }
    .app-hero h1 {
        margin: 0;
        font-size: clamp(1.7rem, 3vw, 2.45rem);
        line-height: 1.15;
        letter-spacing: -0.03em;
        background: linear-gradient(100deg, #67e8f9 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-hero p {
        margin: 0.45rem 0 0;
        color: #94a9c2;
        font-size: 0.98rem;
    }
    .sidebar-label {
        color: #67e8f9;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 0.15rem 0 0.65rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector(model_name: str):
    """Cache model YOLO để không phải tải lại nhiều lần."""
    model_path = resolve_model_path(model_name)
    return VehicleDetector(model_path=model_path)


@st.cache_data
def load_training_results(run_dir: str):
    """Đọc file results.csv từ thư mục kết quả huấn luyện và trả về DataFrame."""
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data
def load_all_experiments(base_dir: str):
    """Quét tất cả các thư mục thí nghiệm trong runs/detect/ và tổng hợp kết quả."""
    experiments = []
    if not os.path.exists(base_dir):
        return pd.DataFrame()

    for run_name in sorted(os.listdir(base_dir)):
        run_path = os.path.join(base_dir, run_name)
        if not os.path.isdir(run_path):
            continue

        args_path = os.path.join(run_path, "args.yaml")
        csv_path = os.path.join(run_path, "results.csv")
        if not os.path.exists(csv_path):
            continue

        args_data = {}
        if os.path.exists(args_path):
            with open(args_path, "r", encoding="utf-8") as f:
                args_data = yaml.safe_load(f) or {}

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if df.empty:
            continue

        last_row = df.iloc[-1]
        num_epochs = len(df)

        experiments.append({
            "Tên thí nghiệm": run_name,
            "Mô hình gốc": args_data.get("model", "N/A"),
            "Số epoch": num_epochs,
            "Batch size": args_data.get("batch", "N/A"),
            "Image size": args_data.get("imgsz", "N/A"),
            "mAP@50 (cuối)": round(float(last_row.get("metrics/mAP50(B)", 0)), 4),
            "mAP@50-95 (cuối)": round(float(last_row.get("metrics/mAP50-95(B)", 0)), 4),
            "Precision (cuối)": round(float(last_row.get("metrics/precision(B)", 0)), 4),
            "Recall (cuối)": round(float(last_row.get("metrics/recall(B)", 0)), 4),
            "Thời gian (phút)": round(float(last_row.get("time", 0)) / 60, 1),
        })

    return pd.DataFrame(experiments)


def main():
    # Header
    with st.container(border=True):
        st.markdown(
            '<div class="app-hero"><h1>Hệ thống nhận diện & đếm phương tiện đường bộ</h1>'
            '<p>Giám sát giao thông thời gian thực với YOLOv8, ByteTrack và xử lý ảnh OpenCV.</p></div>',
            unsafe_allow_html=True,
        )

    # Sidebar điều khiển
    with st.sidebar:
        with st.container(border=True):
            st.markdown('<div class="sidebar-label">AI model</div>', unsafe_allow_html=True)
            st.subheader("Cấu hình mô hình")

            model_choice = st.selectbox(
                "Chọn mô hình YOLO:",
                [
                    "best.pt (Fine-tuned, Khuyên dùng ✅)",
                    "yolov8n.pt (Pretrained gốc, Nhanh)",
                    "yolov8s.pt (Pretrained gốc, Chính xác hơn)"
                ],
                index=0
            )
            model_file = model_choice.split(" ")[0]
            pixels_per_meter = st.number_input("Pixels per meter (calibration):", 1.0, 500.0, 10.0, 1.0,
                                               help="Số pixel tương đương 1 mét trên mặt đường.")
            speed_warning_threshold = st.number_input("Speed warning (km/h):", 1.0, 200.0, 50.0, 1.0)

            conf_thresh = st.slider("Ngưỡng tự tin (Confidence Threshold):", 0.1, 1.0, 0.35, 0.05)
            iou_thresh = st.slider("Ngưỡng IoU (Non-Max Suppression):", 0.1, 1.0, 0.45, 0.05)

        with st.container(border=True):
            st.markdown('<div class="sidebar-label">Counting line</div>', unsafe_allow_html=True)
            st.subheader("Cấu hình vạch đếm")
            line_pos = st.slider("Vị trí vạch kẻ ảo (% chiều cao video):", 10, 90, 65, 5) / 100.0

        st.info("Các lớp phương tiện hỗ trợ:\n\n- Ô tô (Car)\n- Xe máy (Motorcycle)\n- Xe buýt (Bus)\n- Xe tải (Truck)\n- Xe đạp (Bicycle)")

    # Khởi tạo detector
    with st.spinner("Đang tải mô hình AI..."):
        detector = load_detector(model_file)
    visualizer = Visualizer()

    # Khởi tạo bộ xử lý ảnh cổ điển
    img_processor = ImageProcessor()

    # Tabs chức năng
    tab_webcam, tab_video, tab_image, tab_cv, tab_analytics, tab_theory = st.tabs([
        "📷 Webcam trực tiếp",
        "📹 Nhận diện & Đếm Video",
        "🖼️ Nhận diện Ảnh tĩnh",
        "🔬 Xử lý Ảnh Cổ điển",
        "📈 Đánh giá & Huấn luyện",
        "📖 Cơ sở lý thuyết & Báo cáo"
    ])

    # ----------------------------------------------------
    # TAB 0: WEBCAM TRỰC TIẾP (REAL-TIME)
    # ----------------------------------------------------
    with tab_webcam:
        st.subheader("Nhận diện phương tiện trực tiếp qua Webcam")
        st.info(
            "💡 **Hướng dẫn:** Nhấn **START** để bật webcam. "
            "Mô hình AI sẽ nhận diện phương tiện giao thông trong thời gian thực. "
            "Nhấn **STOP** để tắt."
        )

        # Cấu hình webcam
        col_wc1, col_wc2 = st.columns(2)
        with col_wc1:
            wc_conf = st.slider(
                "Ngưỡng tự tin (Webcam):",
                0.1, 1.0, 0.35, 0.05,
                key="wc_conf"
            )
        with col_wc2:
            wc_iou = st.slider(
                "Ngưỡng IoU (Webcam):",
                0.1, 1.0, 0.45, 0.05,
                key="wc_iou"
            )

        # Class để xử lý video frame từ webcam
        class VehicleVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.conf_threshold = 0.35
                self.iou_threshold = 0.45
                self.result_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0}
                self._lock = threading.Lock()

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")

                # Chạy nhận diện YOLO
                detections = detector.detect(
                    img,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold
                )

                # Đếm theo loại xe
                counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0}
                for d in detections:
                    cls_en = d.get("class_en", "car")
                    if cls_en in counts:
                        counts[cls_en] += 1

                with self._lock:
                    self.result_counts = counts.copy()

                # Vẽ kết quả lên frame
                out_img = visualizer.draw_detections(img, detections, VEHICLE_COLORS)
                out_img = visualizer.draw_dashboard_hud(
                    out_img, counts, len(detections), fps=0
                )

                return av.VideoFrame.from_ndarray(out_img, format="bgr24")

        # WebRTC Streamer
        ctx = webrtc_streamer(
            key="vehicle-detection-webcam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VehicleVideoProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Cập nhật tham số confidence/iou theo slider
        if ctx.video_processor:
            ctx.video_processor.conf_threshold = wc_conf
            ctx.video_processor.iou_threshold = wc_iou

        st.markdown("---")
        st.markdown(
            "**📌 Ghi chú kỹ thuật:**\n"
            "- Webcam sử dụng giao thức **WebRTC** để truyền video trực tiếp trong trình duyệt.\n"
            "- Mỗi frame được xử lý qua mô hình **YOLOv8** để nhận diện phương tiện.\n"
            "- Tốc độ xử lý phụ thuộc vào phần cứng (GPU/CPU) của máy chủ."
        )

    # ----------------------------------------------------
    # TAB 1: NHẬN DIỆN & ĐẾM TRÊN VIDEO
    # ----------------------------------------------------
    with tab_video:
        st.subheader("Nhận diện, Theo dõi (Tracking) & Đếm xe qua vạch")
        
        input_source = st.radio("Chọn nguồn video:", ["Tải lên video của bạn (.mp4, .avi, .mov)", "Sử dụng video mẫu có sẵn"], horizontal=True)

        video_path = None
        if input_source == "Tải lên video của bạn (.mp4, .avi, .mov)":
            uploaded_file = st.file_uploader("Kéo thả hoặc chọn file video:", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                try:
                    video_path = save_uploaded_video(uploaded_file)
                except ValueError as exc:
                    st.error(str(exc))
        else:
            # Video mẫu trong thư mục data/samples/
            sample_dir = "data/samples"
            os.makedirs(sample_dir, exist_ok=True)
            sample_videos = [f for f in sorted(os.listdir(sample_dir)) if f.endswith((".mp4", ".avi", ".mov"))]
            video_labels = {
                "traffic.mp4": "🚗 traffic.mp4 (Cao tốc thực tế - 720p, tối ưu chạy mượt)",
                "highway.mp4": "🛣️ highway.mp4 (Cao tốc Full HD 1080p - Lưu lượng xe lớn)",
                "motorcycles.mp4": "🛵 motorcycles.mp4 (Đường phố đô thị - Nhận diện xe máy & ô tô)"
            }
            if sample_videos:
                selected_sample = st.selectbox(
                    "Chọn video mẫu:",
                    sample_videos,
                    format_func=lambda x: video_labels.get(x, f"📹 {x}")
                )
                video_path = os.path.join(sample_dir, selected_sample)
            else:
                st.warning(f"Chưa có video mẫu trong thư mục `{sample_dir}`. Bạn có thể tải lên video ở tùy chọn bên trên!")

        if video_path and os.path.exists(video_path):
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
            start_btn = col_ctrl1.button("▶️ Bắt đầu xử lý Video", type="primary", width="stretch")
            enable_heatmap = col_ctrl2.checkbox("🔥 Hiển thị Heatmap mật độ", value=False)
            enable_roi = col_ctrl3.checkbox("🔷 Bật vùng kiểm soát ROI", value=False)

            if start_btn:
                try:
                    width, height, total_frames, _source_fps = get_video_metadata(video_path)
                except OSError as exc:
                    cleanup_temp_video(video_path)
                    st.error(str(exc))
                    st.stop()

                # Thiết lập vạch đếm & ROI
                line_y = int(height * line_pos)
                roi_coords = None
                if enable_roi:
                    roi_coords = [
                        (int(width * 0.15), int(height * 0.45)),
                        (int(width * 0.85), int(height * 0.45)),
                        (int(width * 0.85), int(height * 0.85)),
                        (int(width * 0.15), int(height * 0.85))
                    ]
                counter = VehicleCounter(
                    line_coords=((0, line_y), (width, line_y)),
                    polygon_coords=roi_coords,
                    pixels_per_meter=pixels_per_meter
                )

                # Khu vực hiển thị kết quả
                col_view, col_stats = st.columns([3, 1])
                video_display = col_view.empty()
                progress_bar = st.progress(0)
                
                with col_stats:
                    st.markdown("### 📊 Thống kê trực tiếp")
                    m_total = st.empty()
                    m_car = st.empty()
                    m_moto = st.empty()
                    m_bus = st.empty()
                    m_truck = st.empty()
                    m_fps = st.empty()

                prev_time = time.time()
                fps = 0.0
                next_display_time = 0.0

                for frame_idx, frame, total_frames, frame_timestamp in iter_video_frames(video_path, cleanup=True):
                    curr_time = time.time()
                    dt = curr_time - prev_time
                    if dt > 0:
                        fps = 0.9 * fps + 0.1 * (1.0 / dt)
                    prev_time = curr_time

                    # Nhận diện & Track
                    tracked_objs = detector.track(frame, conf_threshold=conf_thresh, iou_threshold=iou_thresh, persist=True)
                    count_info = counter.update(tracked_objs, timestamp=frame_timestamp)

                    # Vẽ trực quan
                    disp = visualizer.draw_detections(
                        frame, tracked_objs, VEHICLE_COLORS,
                        speed_warning_threshold=speed_warning_threshold
                    )
                    disp = visualizer.draw_trajectories(disp, counter.tracks_history)
                    disp = visualizer.draw_counting_line(disp, counter.line_coords)
                    if enable_roi and counter.polygon_coords:
                        disp = visualizer.draw_polygon_roi(disp, counter.polygon_coords)
                    if enable_heatmap:
                        disp = visualizer.draw_heatmap(disp, counter.tracks_history)
                    disp = visualizer.draw_dashboard_hud(
                        disp, count_info["counts"], count_info["total"], fps=fps,
                        directions=count_info["directions"],
                        speed_summary=counter.get_speed_summary(speed_warning_threshold),
                        speed_warning_threshold=speed_warning_threshold,
                    )

                    # Chuyển BGR sang RGB cho Streamlit
                    if curr_time >= next_display_time:
                        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                        video_display.image(disp_rgb, channels="RGB", width="stretch")
                        progress_bar.progress(min(1.0, frame_idx / total_frames))
                        next_display_time = curr_time + (1 / 12)

                    # Cập nhật metrics
                    m_total.metric("🚗 Tổng số xe", count_info["total"])
                    m_car.metric("Ô tô con", count_info["counts"].get("car", 0))
                    m_moto.metric("Xe máy", count_info["counts"].get("motorcycle", 0))
                    m_bus.metric("Xe buýt", count_info["counts"].get("bus", 0))
                    m_truck.metric("Xe tải", count_info["counts"].get("truck", 0))
                    m_fps.metric("Tốc độ (FPS)", f"{fps:.1f}")


                cleanup_temp_video(video_path)
                st.success("🎉 Đã hoàn thành phân tích video!")

                # Tải xuống báo cáo CSV
                if counter.events_log:
                    st.markdown("#### 📥 Báo cáo chi tiết các lượt xe qua vạch")
                    df_events = pd.DataFrame(counter.events_log)
                    st.dataframe(df_events, width="stretch", hide_index=True)
                    csv_data = df_events.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Tải xuống file CSV báo cáo",
                        data=csv_data,
                        file_name=f"traffic_report_{int(time.time())}.csv",
                        mime="text/csv",
                        type="primary"
                    )

    # ----------------------------------------------------
    # TAB 2: NHẬN DIỆN TRÊN ẢNH TĨNH
    # ----------------------------------------------------
    with tab_image:
        st.subheader("Nhận diện và Phân loại Phương tiện trên Ảnh")
        uploaded_img = st.file_uploader("Chọn ảnh giao thông (.jpg, .png, .jpeg):", type=["jpg", "png", "jpeg", "webp"])

        if uploaded_img is not None:
            image_pil = Image.open(uploaded_img)
            image_np = np.array(image_pil)
            # Chuyển RGB sang BGR để đưa qua OpenCV/YOLO
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)

            t_start = time.time()
            detections = detector.detect(image_bgr, conf_threshold=conf_thresh, iou_threshold=iou_thresh)
            t_detect = (time.time() - t_start) * 1000

            # Thống kê loại xe
            summary = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0}
            table_data = []
            for idx, d in enumerate(detections, 1):
                cls_en = d["class_en"]
                if cls_en in summary:
                    summary[cls_en] += 1
                x1, y1, x2, y2 = d["bbox"]
                table_data.append({
                    "STT": idx,
                    "Loại phương tiện": d["class_vi"],
                    "Tên tiếng Anh": cls_en,
                    "Độ tin cậy (Conf)": f"{d['confidence'] * 100:.1f}%",
                    "Tọa độ BBox [x1, y1, x2, y2]": f"[{x1}, {y1}, {x2}, {y2}]"
                })

            # Vẽ bounding boxes
            res_img = visualizer.draw_detections(image_bgr, detections, VEHICLE_COLORS)
            res_img = visualizer.draw_dashboard_hud(res_img, summary, len(detections), fps=1000.0 / max(t_detect, 1))
            res_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

            col_img1, col_img2 = st.columns([3, 2])
            with col_img1:
                st.image(res_rgb, caption=f"Kết quả nhận diện ({len(detections)} phương tiện)", width="stretch")
            with col_img2:
                st.markdown(f"**⏱️ Thời gian suy luận:** `{t_detect:.2f} ms`")
                st.markdown(f"**🔢 Tổng phương tiện tìm thấy:** `{len(detections)}`")
                for k, v in summary.items():
                    if v > 0:
                        st.markdown(f"- **{k.capitalize()}**: `{v}` xe")

                if table_data:
                    st.markdown("#### Chi tiết danh sách phát hiện:")
                    df_dets = pd.DataFrame(table_data)
                    st.dataframe(df_dets, width="stretch", hide_index=True)
                    csv_dets = df_dets.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Tải danh sách phát hiện (CSV)",
                        data=csv_dets,
                        file_name=f"detections_{int(time.time())}.csv",
                        mime="text/csv"
                    )

    # ----------------------------------------------------
    # TAB 3: XỬ LÝ ẢNH CỔ ĐIỂN (CLASSICAL COMPUTER VISION)
    # ----------------------------------------------------
    with tab_cv:
        st.subheader("🔬 Xử lý Ảnh & Thị giác Máy tính — Phương pháp Cổ điển")
        st.caption("Triển khai các kỹ thuật xử lý ảnh nền tảng (không dùng Deep Learning) để phát hiện phương tiện. "
                   "Đồng thời so sánh trực tiếp với kết quả từ mô hình YOLOv8.")

        (cv_sub_tab1, cv_sub_tab2, cv_sub_tab3, cv_sub_tab4, cv_sub_tab5,
         cv_sub_tab6, cv_sub_tab7, cv_sub_tab8, cv_sub_tab9, cv_sub_tab10, cv_sub_tab11) = st.tabs([
            "🖼️ Pipeline Xử lý Ảnh",
            "🔍 Phát hiện Cạnh",
            "⚙️ Phép Hình thái học",
            "📹 Trừ nền Video",
            "⚔️ So sánh AI vs Cổ điển",
            "✨ Nâng cao Chất lượng",
            "📊 Biểu đồ Tần suất",
            "🔬 Biên Nâng cao",
            "🧩 Phân vùng Ảnh",
            "📦 Nén Dữ liệu Ảnh",
            "📡 Xử lý Ảnh Nhiễu"
        ])

        # ----- SUB TAB 1: PIPELINE XỬ LÝ ẢNH TĨNH -----
        with cv_sub_tab1:
            st.markdown("#### Pipeline xử lý ảnh tĩnh hoàn chỉnh")
            st.info(
                "**Quy trình:** Ảnh gốc → Grayscale → CLAHE → Gaussian Blur → Canny Edge Detection "
                "→ Morphology (Closing + Dilation) → Contour Detection → Bounding Box"
            )

            cv_uploaded_img = st.file_uploader(
                "Tải lên ảnh giao thông (.jpg, .png):",
                type=["jpg", "png", "jpeg", "webp"],
                key="cv_pipeline_upload"
            )

            col_param1, col_param2, col_param3 = st.columns(3)
            with col_param1:
                cv_blur_k = st.slider("Kích thước Gaussian Kernel:", 3, 15, 5, 2, key="cv_blur")
            with col_param2:
                cv_canny_low = st.slider("Canny ngưỡng thấp:", 10, 150, 50, 5, key="cv_canny_lo")
                cv_canny_high = st.slider("Canny ngưỡng cao:", 50, 300, 150, 5, key="cv_canny_hi")
            with col_param3:
                cv_min_area = st.slider("Diện tích contour tối thiểu:", 100, 5000, 800, 100, key="cv_min_area")
                cv_clahe_clip = st.slider("CLAHE Clip Limit:", 1.0, 10.0, 3.0, 0.5, key="cv_clahe")

            if cv_uploaded_img is not None:
                img_pil = Image.open(cv_uploaded_img)
                img_np = np.array(img_pil)
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

                with st.spinner("Đang xử lý pipeline ảnh cổ điển..."):
                    pipeline_result = img_processor.full_pipeline_image(
                        img_bgr,
                        blur_ksize=cv_blur_k,
                        canny_low=cv_canny_low,
                        canny_high=cv_canny_high,
                        min_contour_area=cv_min_area,
                        clahe_clip=cv_clahe_clip
                    )

                steps = pipeline_result["steps"]
                dets = pipeline_result["detections"]

                st.success(f"✅ Pipeline hoàn tất! Phát hiện **{len(dets)} vùng phương tiện** bằng phương pháp cổ điển.")

                # Hiển thị từng bước
                st.markdown("##### Các bước xử lý trung gian:")
                step_cols = st.columns(4)
                step_names = list(steps.keys())
                for i, name in enumerate(step_names):
                    with step_cols[i % 4]:
                        img_step = steps[name]
                        if len(img_step.shape) == 2:
                            st.image(img_step, caption=name, width="stretch", clamp=True)
                        else:
                            st.image(cv2.cvtColor(img_step, cv2.COLOR_BGR2RGB), caption=name, width="stretch")

                # Bảng chi tiết contours
                if dets:
                    st.markdown("##### Chi tiết các phương tiện phát hiện được:")
                    table_cv = []
                    for idx, d in enumerate(dets, 1):
                        x1, y1, x2, y2 = d["bbox"]
                        table_cv.append({
                            "STT": idx,
                            "Diện tích (px²)": int(d["area"]),
                            "Tỷ lệ W/H": d["aspect_ratio"],
                            "Tâm (cx, cy)": f"({d['center'][0]}, {d['center'][1]})",
                            "BBox [x1,y1,x2,y2]": f"[{x1},{y1},{x2},{y2}]"
                        })
                    st.dataframe(pd.DataFrame(table_cv), hide_index=True, width="stretch")

        # ----- SUB TAB 2: PHÁT HIỆN CẠNH -----
        with cv_sub_tab2:
            st.markdown("#### So sánh các phương pháp Phát hiện Cạnh (Edge Detection)")
            st.info(
                "Ba phương pháp phổ biến: **Sobel** (gradient bậc 1 theo X/Y), "
                "**Canny** (multi-stage, chuẩn công nghiệp), **Laplacian** (đạo hàm bậc 2, đẳng hướng)."
            )

            edge_uploaded = st.file_uploader(
                "Tải ảnh để so sánh edge detection:",
                type=["jpg", "png", "jpeg"],
                key="edge_upload"
            )

            if edge_uploaded is not None:
                img_pil = Image.open(edge_uploaded)
                img_np = np.array(img_pil)
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    edge_canny_lo = st.slider("Canny Low:", 10, 200, 50, 5, key="edge_canny_lo")
                    edge_canny_hi = st.slider("Canny High:", 50, 400, 150, 10, key="edge_canny_hi")
                with col_e2:
                    edge_sobel_k = st.slider("Sobel Kernel Size:", 3, 7, 3, 2, key="edge_sobel_k")

                # Tính toán
                canny_result = img_processor.detect_edges_canny(img_bgr, edge_canny_lo, edge_canny_hi)
                sobel_result = img_processor.detect_edges_sobel(img_bgr, ksize=edge_sobel_k)
                laplacian_result = img_processor.detect_edges_laplacian(img_bgr)

                # Hiển thị
                st.markdown("##### Kết quả so sánh:")
                ce1, ce2, ce3, ce4 = st.columns(4)
                with ce1:
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                with ce2:
                    st.image(canny_result, caption="Canny Edge", width="stretch", clamp=True)
                with ce3:
                    st.image(sobel_result["magnitude"], caption="Sobel Magnitude", width="stretch", clamp=True)
                with ce4:
                    st.image(laplacian_result, caption="Laplacian", width="stretch", clamp=True)

                # Chi tiết Sobel X/Y
                with st.expander("🔎 Xem chi tiết Sobel X và Sobel Y"):
                    cs1, cs2 = st.columns(2)
                    with cs1:
                        st.image(sobel_result["sobel_x"], caption="Sobel X (cạnh dọc)", width="stretch", clamp=True)
                    with cs2:
                        st.image(sobel_result["sobel_y"], caption="Sobel Y (cạnh ngang)", width="stretch", clamp=True)

                    st.markdown("""
                    **Giải thích:**
                    - **Sobel X** tính gradient theo hướng ngang → phát hiện **cạnh dọc** (viền trái/phải xe).
                    - **Sobel Y** tính gradient theo hướng dọc → phát hiện **cạnh ngang** (nóc, gầm xe).
                    - **Magnitude** = √(Gx² + Gy²): kết hợp cả 2 hướng để có bản đồ cạnh hoàn chỉnh.
                    """)

        # ----- SUB TAB 3: PHÉP HÌNH THÁI HỌC -----
        with cv_sub_tab3:
            st.markdown("#### Minh họa các Phép toán Hình thái học (Morphological Operations)")
            st.info(
                "Các phép toán trên ảnh nhị phân: **Erosion** (co), **Dilation** (giãn), "
                "**Opening** (loại nhiễu), **Closing** (lấp lỗ), **Gradient** (trích viền)."
            )

            morph_uploaded = st.file_uploader(
                "Tải ảnh để demo hình thái học:",
                type=["jpg", "png", "jpeg"],
                key="morph_upload"
            )

            if morph_uploaded is not None:
                img_pil = Image.open(morph_uploaded)
                img_np = np.array(img_pil)
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

                morph_canny_lo = st.slider("Ngưỡng Canny (tạo mask nhị phân):", 20, 200, 80, 10, key="morph_canny")

                # Tạo mask nhị phân từ Canny edge
                binary_mask = img_processor.detect_edges_canny(img_bgr, morph_canny_lo, morph_canny_lo * 2)

                # Demo tất cả phép toán
                morph_steps = img_processor.demonstrate_morphology_steps(binary_mask)

                st.markdown("##### Kết quả từng phép toán:")
                mc = st.columns(3)
                morph_names = list(morph_steps.keys())
                morph_labels = {
                    "original": "Mask gốc (Canny)",
                    "erosion": "Erosion (Co nhỏ)",
                    "dilation": "Dilation (Giãn nở)",
                    "opening": "Opening (Mở = E→D)",
                    "closing": "Closing (Đóng = D→E)",
                    "gradient": "Gradient (D - E)"
                }
                for i, name in enumerate(morph_names):
                    with mc[i % 3]:
                        st.image(morph_steps[name], caption=morph_labels.get(name, name), width="stretch", clamp=True)

                st.markdown("""
                **Ý nghĩa trong bài toán phát hiện xe:**
                - **Erosion**: Loại bỏ nhiễu nhỏ (hạt muối) trên mask phân vùng.
                - **Dilation**: Lấp đầy các lỗ hổng nhỏ bên trong vùng xe.
                - **Opening** (Erosion → Dilation): Loại nhiễu mà không thay đổi kích thước đối tượng lớn.
                - **Closing** (Dilation → Erosion): Nối các vùng rời rạc của cùng 1 xe thành vùng liền mạch.
                - **Morphological Gradient**: Trích xuất đường biên hình thái, tương tự edge detection.
                """)

        # ----- SUB TAB 4: BACKGROUND SUBTRACTION (VIDEO) -----
        with cv_sub_tab4:
            st.markdown("#### Phát hiện phương tiện bằng Trừ nền (Background Subtraction)")
            st.info(
                "Phương pháp cổ điển để phát hiện xe đang di chuyển trong video: "
                "So sánh frame hiện tại với mô hình nền thích nghi (MOG2/KNN) "
                "để tách foreground (phương tiện) ra khỏi background (đường, cây, ...).")

            bg_method = st.radio("Chọn thuật toán trừ nền:", ["MOG2", "KNN"], horizontal=True, key="bg_method")

            bg_source = st.radio("Chọn nguồn video:",
                                 ["Tải lên video", "Dùng video mẫu có sẵn"],
                                 horizontal=True, key="bg_source")

            bg_video_path = None
            if bg_source == "Tải lên video":
                bg_uploaded = st.file_uploader("Tải video:", type=["mp4", "avi", "mov"], key="bg_video_upload")
                if bg_uploaded is not None:
                    try:
                        bg_video_path = save_uploaded_video(bg_uploaded)
                    except ValueError as exc:
                        st.error(str(exc))
            else:
                sample_dir = "data/samples"
                os.makedirs(sample_dir, exist_ok=True)
                sample_videos = [f for f in sorted(os.listdir(sample_dir)) if f.endswith((".mp4", ".avi"))]
                video_labels = {
                    "traffic.mp4": "🚗 traffic.mp4 (Cao tốc thực tế - 720p, tối ưu chạy mượt)",
                    "highway.mp4": "🛣️ highway.mp4 (Cao tốc Full HD 1080p - Lưu lượng xe lớn)",
                    "motorcycles.mp4": "🛵 motorcycles.mp4 (Đường phố đô thị - Nhận diện xe máy & ô tô)"
                }
                if sample_videos:
                    bg_selected = st.selectbox(
                        "Chọn video mẫu:",
                        sample_videos,
                        format_func=lambda x: video_labels.get(x, f"📹 {x}"),
                        key="bg_sample_select"
                    )
                    bg_video_path = os.path.join(sample_dir, bg_selected)
                else:
                    st.warning("Chưa có video mẫu trong thư mục `data/samples/`.")

            col_bg_param1, col_bg_param2 = st.columns(2)
            with col_bg_param1:
                bg_min_area = st.slider("Diện tích contour tối thiểu:", 200, 5000, 1200, 100, key="bg_min_area")
            with col_bg_param2:
                bg_max_area = st.slider("Diện tích contour tối đa:", 5000, 200000, 80000, 5000, key="bg_max_area")

            if bg_video_path and os.path.exists(bg_video_path):
                bg_start_btn = st.button("▶️ Bắt đầu xử lý Background Subtraction", type="primary", key="bg_start")

                if bg_start_btn:
                    img_processor.reset_background_subtractor(bg_method)

                    try:
                        _, _, total_frames, _ = get_video_metadata(bg_video_path)
                    except OSError as exc:
                        cleanup_temp_video(bg_video_path)
                        st.error(str(exc))
                        st.stop()

                    col_bg_view1, col_bg_view2 = st.columns(2)
                    bg_display_orig = col_bg_view1.empty()
                    bg_display_mask = col_bg_view2.empty()
                    col_bg_view3, col_bg_view4 = st.columns(2)
                    bg_display_clean = col_bg_view3.empty()
                    bg_display_result = col_bg_view4.empty()
                    bg_progress = st.progress(0)
                    bg_status = st.empty()

                    next_display_time = 0.0
                    for frame_idx, frame, total_frames, _ in iter_video_frames(bg_video_path, cleanup=True):

                        bg_pipeline = img_processor.full_pipeline_video_frame(
                            frame, method=bg_method,
                            min_contour_area=bg_min_area,
                            max_contour_area=bg_max_area
                        )

                        # Hiển thị 4 cửa sổ
                        bg_display_orig.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                             caption="Frame gốc", width="stretch")
                        bg_display_mask.image(bg_pipeline["fg_mask_raw"],
                                             caption=f"Foreground Mask ({bg_method} thô)", width="stretch", clamp=True)
                        bg_display_clean.image(bg_pipeline["fg_mask_clean"],
                                              caption="Mask sau Morphology", width="stretch", clamp=True)
                        bg_display_result.image(cv2.cvtColor(bg_pipeline["result"], cv2.COLOR_BGR2RGB),
                                               caption=f"Kết quả: {len(bg_pipeline['detections'])} xe", width="stretch")

                        bg_progress.progress(min(frame_idx / total_frames, 1.0))
                        bg_status.markdown(f"Frame **{frame_idx}/{total_frames}** — Phát hiện: **{len(bg_pipeline['detections'])}** phương tiện")

                    st.success("✅ Xử lý video hoàn tất!")

        # ----- SUB TAB 5: SO SÁNH YOLOv8 vs CỔ ĐIỂN -----
        with cv_sub_tab5:
            st.markdown("#### ⚔️ So sánh: YOLOv8 (Deep Learning) vs Xử lý Ảnh Cổ điển")
            st.info(
                "Tải lên một ảnh giao thông để chạy đồng thời 2 phương pháp và so sánh kết quả "
                "về số lượng phát hiện, thời gian xử lý, và chất lượng nhận diện."
            )

            cmp_uploaded = st.file_uploader(
                "Tải ảnh giao thông để so sánh:",
                type=["jpg", "png", "jpeg", "webp"],
                key="cmp_upload"
            )

            if cmp_uploaded is not None:
                img_pil = Image.open(cmp_uploaded)
                img_np = np.array(img_pil)
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

                st.markdown("---")

                # --- PHƯƠNG PHÁP 1: YOLOv8 ---
                with st.spinner("🤖 Đang chạy YOLOv8..."):
                    t1 = time.time()
                    yolo_dets = detector.detect(img_bgr, conf_threshold=conf_thresh, iou_threshold=iou_thresh)
                    t_yolo = (time.time() - t1) * 1000

                yolo_img = visualizer.draw_detections(img_bgr, yolo_dets, VEHICLE_COLORS)
                yolo_summary = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0}
                for d in yolo_dets:
                    cls = d["class_en"]
                    if cls in yolo_summary:
                        yolo_summary[cls] += 1

                # --- PHƯƠNG PHÁP 2: Classical CV ---
                with st.spinner("🔬 Đang chạy pipeline xử lý ảnh cổ điển..."):
                    t2 = time.time()
                    cv_result = img_processor.full_pipeline_image(img_bgr)
                    t_cv = (time.time() - t2) * 1000

                cv_img = cv_result["result"]
                cv_dets = cv_result["detections"]

                # --- HIỂN THỊ SO SÁNH SONG SONG ---
                col_yolo, col_cv = st.columns(2)
                with col_yolo:
                    st.markdown("##### 🤖 YOLOv8 (Deep Learning)")
                    st.image(cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB),
                             caption=f"{len(yolo_dets)} phương tiện — {t_yolo:.1f}ms", width="stretch")
                with col_cv:
                    st.markdown("##### 🔬 Xử lý Ảnh Cổ điển")
                    st.image(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB),
                             caption=f"{len(cv_dets)} vùng phương tiện — {t_cv:.1f}ms", width="stretch")

                # --- BẢNG SO SÁNH CHI TIẾT ---
                st.markdown("##### 📊 Bảng so sánh tổng hợp:")

                comparison_data = {
                    "Tiêu chí": [
                        "Số phương tiện phát hiện",
                        "Thời gian xử lý",
                        "Phân loại loại xe",
                        "Độ chính xác ước lượng",
                        "Cần GPU",
                        "Cần huấn luyện dữ liệu",
                        "Hoạt động trong mọi điều kiện"
                    ],
                    "🤖 YOLOv8 (AI)": [
                        f"{len(yolo_dets)} phương tiện",
                        f"{t_yolo:.1f} ms",
                        "✅ Có (Car, Motorcycle, Bus, Truck, Bicycle)",
                        "⭐⭐⭐⭐⭐ Rất cao",
                        "⚡ Nhanh hơn nhiều nếu có GPU",
                        "✅ Cần dataset + huấn luyện trước",
                        "✅ Ổn định cao"
                    ],
                    "🔬 Cổ điển (CV)": [
                        f"{len(cv_dets)} vùng",
                        f"{t_cv:.1f} ms",
                        "❌ Không (chỉ phát hiện vùng chuyển động/contour)",
                        "⭐⭐ Trung bình (nhiều false positive)",
                        "❌ Chỉ cần CPU",
                        "❌ Không cần huấn luyện",
                        "⚠️ Nhạy với ánh sáng, thời tiết"
                    ]
                }
                st.table(pd.DataFrame(comparison_data))

                # --- NHẬN XÉT TỰ ĐỘNG ---
                with st.expander("📝 Xem nhận xét phân tích so sánh", expanded=True):
                    st.markdown(f"""
**Phân tích so sánh kết quả:**

| Phương pháp | Số phát hiện | Thời gian |
|---|---|---|
| YOLOv8 (Deep Learning) | **{len(yolo_dets)}** phương tiện | **{t_yolo:.1f}** ms |
| Xử lý Ảnh Cổ điển | **{len(cv_dets)}** vùng | **{t_cv:.1f}** ms |

**Ưu điểm của YOLOv8:**
- Phân loại chính xác từng loại xe (ô tô, xe máy, xe buýt, xe tải, xe đạp).
- Robust trước nhiễu, thay đổi ánh sáng và góc chụp.
- Mỗi detection có confidence score tin cậy.

**Ưu điểm của Xử lý Ảnh Cổ điển:**
- Không cần GPU hay dữ liệu huấn luyện — phù hợp hệ thống nhúng đơn giản.
- Giải thích được từng bước (explainable/interpretable) — phù hợp cho nghiên cứu/giảng dạy.
- Phát hiện mọi vật thể chuyển động, không giới hạn loại xe cố định.

**Kết luận:** Trong thực tế, phương pháp **Deep Learning (YOLOv8)** vượt trội hơn về độ chính xác.
Tuy nhiên, nền tảng **xử lý ảnh cổ điển** là kiến thức bắt buộc vì nó là cơ sở lý thuyết
để hiểu cách hoạt động bên trong các mạng nơ-ron tích chập (CNN).
                    """)

        # ----- SUB TAB 6: NÂNG CAO CHẤT LƯỢNG ẢNH -----
        with cv_sub_tab6:
            st.markdown("#### ✨ Nâng cao Chất lượng Ảnh (Image Enhancement)")
            st.info(
                "Các phương pháp tăng cường ảnh: **Gamma Correction**, **Log Transform**, "
                "**Contrast Stretching**, **Unsharp Masking**, **Denoising (NLM/Bilateral/Median)**, **White Balance**."
            )

            enh_uploaded = st.file_uploader(
                "Tải ảnh để demo nâng cao chất lượng:",
                type=["jpg", "png", "jpeg", "webp"],
                key="enh_upload"
            )

            if enh_uploaded is not None:
                img_pil = Image.open(enh_uploaded)
                img_np = np.array(img_pil)
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                else:
                    img_bgr = img_np

                enh_method = st.selectbox("Chọn phương pháp nâng cao:", [
                    "So sánh tất cả",
                    "Gamma Correction",
                    "Auto Brightness",
                    "Log Transform",
                    "Power-Law Transform",
                    "Contrast Stretching",
                    "Unsharp Masking (Làm sắc nét)",
                    "Denoising (Khử nhiễu)",
                    "White Balance (Cân bằng trắng)",
                ], key="enh_method")

                if enh_method == "So sánh tất cả":
                    with st.spinner("Đang xử lý tất cả phương pháp..."):
                        demo = img_processor.demonstrate_enhancement_methods(img_bgr)
                    st.markdown("##### Kết quả so sánh:")
                    cols = st.columns(4)
                    for i, (name, img_out) in enumerate(demo.items()):
                        with cols[i % 4]:
                            if len(img_out.shape) == 2:
                                st.image(img_out, caption=name, width="stretch", clamp=True)
                            else:
                                st.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), caption=name, width="stretch")

                elif enh_method == "Gamma Correction":
                    gamma_val = st.slider("Giá trị Gamma:", 0.1, 5.0, 1.0, 0.1, key="gamma_val")
                    result = img_processor.apply_gamma_correction(img_bgr, gamma_val)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                    with c2:
                        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Gamma = {gamma_val}", width="stretch")
                    st.markdown(f"""**Giải thích:** Gamma = {gamma_val:.1f} → {'Tăng sáng vùng tối' if gamma_val < 1 else 'Giảm sáng / tăng contrast vùng sáng' if gamma_val > 1 else 'Không thay đổi'}""")

                elif enh_method == "Auto Brightness":
                    target = st.slider("Mean intensity mục tiêu:", 50.0, 200.0, 120.0, 5.0, key="auto_bright_target")
                    result = img_processor.auto_adjust_brightness(img_bgr, target)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                    with c2:
                        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Auto Brightness (target={target})", width="stretch")

                elif enh_method == "Log Transform":
                    result = img_processor.apply_log_transform(img_bgr)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                    with c2:
                        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Log Transform", width="stretch")
                    st.markdown("**Giải thích:** s = c × log(1 + r). Mở rộng dải pixel tối, nén dải pixel sáng.")

                elif enh_method == "Power-Law Transform":
                    pl_gamma = st.slider("Gamma:", 0.1, 5.0, 0.4, 0.1, key="pl_gamma")
                    pl_c = st.slider("Hằng số c:", 0.5, 3.0, 1.0, 0.1, key="pl_c")
                    result = img_processor.apply_power_law_transform(img_bgr, pl_gamma, pl_c)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                    with c2:
                        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Power-Law (γ={pl_gamma}, c={pl_c})", width="stretch")

                elif enh_method == "Contrast Stretching":
                    cs_low = st.slider("Percentile thấp (%):", 0.0, 10.0, 2.0, 0.5, key="cs_low")
                    cs_high = st.slider("Percentile cao (%):", 90.0, 100.0, 98.0, 0.5, key="cs_high")
                    result = img_processor.apply_contrast_stretching(img_bgr, cs_low, cs_high)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                    with c2:
                        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Contrast Stretching [{cs_low}%, {cs_high}%]", width="stretch")

                elif enh_method == "Unsharp Masking (Làm sắc nét)":
                    us_amount = st.slider("Cường độ (amount):", 0.5, 5.0, 1.5, 0.1, key="us_amount")
                    us_sigma = st.slider("Sigma Gaussian:", 0.5, 5.0, 1.0, 0.5, key="us_sigma")
                    result = img_processor.apply_unsharp_masking(img_bgr, sigma=us_sigma, amount=us_amount)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                    with c2:
                        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Unsharp Masking (amount={us_amount})", width="stretch")

                elif enh_method == "Denoising (Khử nhiễu)":
                    dn_method = st.selectbox("Phương pháp khử nhiễu:", ["nlm", "bilateral", "median"], key="dn_method")
                    dn_h = st.slider("Cường độ lọc h (NLM):", 3.0, 30.0, 10.0, 1.0, key="dn_h")
                    result = img_processor.apply_denoising(img_bgr, method=dn_method, h=dn_h)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                    with c2:
                        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Denoising ({dn_method})", width="stretch")

                elif enh_method == "White Balance (Cân bằng trắng)":
                    if len(img_bgr.shape) == 3:
                        result = img_processor.apply_white_balance_grayworld(img_bgr)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                        with c2:
                            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="White Balance (Gray World)", width="stretch")
                    else:
                        st.warning("White Balance chỉ áp dụng cho ảnh màu.")

        # ----- SUB TAB 7: BIỂU ĐỒ TẦN SUẤT -----
        with cv_sub_tab7:
            st.markdown("#### 📊 Biểu đồ Tần suất (Histogram Analysis)")
            st.info(
                "Phân tích phân bố cường độ pixel: **Histogram**, **Normalized Histogram**, "
                "**CDF (Cumulative Distribution Function)**, **So sánh Histogram**, **Histogram Backprojection**."
            )

            hist_uploaded = st.file_uploader(
                "Tải ảnh để phân tích histogram:",
                type=["jpg", "png", "jpeg", "webp"],
                key="hist_upload"
            )

            if hist_uploaded is not None:
                img_pil = Image.open(hist_uploaded)
                img_np = np.array(img_pil)
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                else:
                    img_bgr = img_np

                hist_mode = st.selectbox("Chế độ phân tích:", [
                    "Histogram + CDF",
                    "So sánh hiệu quả các phép biến đổi",
                    "So sánh Histogram 2 ảnh",
                ], key="hist_mode")

                if hist_mode == "Histogram + CDF":
                    draw_cdf = st.checkbox("Hiển thị đường CDF (tích lũy)", value=True, key="hist_cdf")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                    with c2:
                        hist_img = img_processor.draw_histogram_image(img_bgr, draw_cdf=draw_cdf)
                        st.image(cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB), caption="Biểu đồ Histogram", width="stretch")

                    # Thống kê cơ bản
                    gray = img_processor.convert_grayscale(img_bgr)
                    st.markdown("##### Thống kê cường độ pixel:")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Mean", f"{np.mean(gray):.1f}")
                    mc2.metric("Std Dev", f"{np.std(gray):.1f}")
                    mc3.metric("Min", f"{np.min(gray)}")
                    mc4.metric("Max", f"{np.max(gray)}")

                elif hist_mode == "So sánh hiệu quả các phép biến đổi":
                    with st.spinner("Đang xử lý..."):
                        demo = img_processor.demonstrate_histogram_effects(img_bgr)
                    st.markdown("##### Ảnh sau biến đổi và Histogram tương ứng:")
                    cols = st.columns(4)
                    for i, (name, img_out) in enumerate(demo.items()):
                        with cols[i % 4]:
                            if len(img_out.shape) == 2:
                                st.image(img_out, caption=name, width="stretch", clamp=True)
                            else:
                                st.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), caption=name, width="stretch")

                elif hist_mode == "So sánh Histogram 2 ảnh":
                    hist_uploaded2 = st.file_uploader(
                        "Tải ảnh thứ 2 để so sánh:",
                        type=["jpg", "png", "jpeg", "webp"],
                        key="hist_upload2"
                    )
                    if hist_uploaded2 is not None:
                        img_pil2 = Image.open(hist_uploaded2)
                        img_np2 = np.array(img_pil2)
                        if len(img_np2.shape) == 3 and img_np2.shape[2] == 3:
                            img_bgr2 = cv2.cvtColor(img_np2, cv2.COLOR_RGB2BGR)
                        elif len(img_np2.shape) == 3 and img_np2.shape[2] == 4:
                            img_bgr2 = cv2.cvtColor(img_np2, cv2.COLOR_RGBA2BGR)
                        else:
                            img_bgr2 = img_np2

                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh 1", width="stretch")
                            hist1_img = img_processor.draw_histogram_image(img_bgr, draw_cdf=False)
                            st.image(cv2.cvtColor(hist1_img, cv2.COLOR_BGR2RGB), caption="Histogram Ảnh 1", width="stretch")
                        with c2:
                            st.image(cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2RGB), caption="Ảnh 2", width="stretch")
                            hist2_img = img_processor.draw_histogram_image(img_bgr2, draw_cdf=False)
                            st.image(cv2.cvtColor(hist2_img, cv2.COLOR_BGR2RGB), caption="Histogram Ảnh 2", width="stretch")

                        # Kết quả so sánh
                        comparison = img_processor.compare_histograms(img_bgr, img_bgr2)
                        st.markdown("##### Kết quả so sánh Histogram:")
                        cmp_df = pd.DataFrame([
                            {"Phương pháp": k, "Giá trị": v,
                             "Ý nghĩa": "Giống" if (k == "correlation" and v > 0.7) or (k == "bhattacharyya" and v < 0.3) or (k == "chi_square" and v < 100) else "Khác biệt"}
                            for k, v in comparison.items()
                        ])
                        st.dataframe(cmp_df, hide_index=True, width="stretch")

        # ----- SUB TAB 8: PHÁT HIỆN BIÊN NÂNG CAO -----
        with cv_sub_tab8:
            st.markdown("#### 🔬 Các Phương pháp Phát hiện Biên Nâng cao")
            st.info(
                "So sánh các toán tử: **Prewitt**, **Roberts Cross**, **Scharr**, "
                "**Laplacian of Gaussian (LoG)**, **Auto Canny**, **Zero-Crossing**."
            )

            adv_edge_uploaded = st.file_uploader(
                "Tải ảnh để so sánh các phương pháp biên:",
                type=["jpg", "png", "jpeg", "webp"],
                key="adv_edge_upload"
            )

            if adv_edge_uploaded is not None:
                img_pil = Image.open(adv_edge_uploaded)
                img_np = np.array(img_pil)
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                else:
                    img_bgr = img_np

                adv_edge_mode = st.selectbox("Chế độ:", [
                    "So sánh tất cả phương pháp",
                    "Prewitt (chi tiết X/Y)",
                    "Roberts Cross (chi tiết)",
                    "Scharr (chi tiết X/Y)",
                    "LoG — Laplacian of Gaussian",
                    "Auto Canny (tự động ngưỡng)",
                    "Zero-Crossing",
                ], key="adv_edge_mode")

                if adv_edge_mode == "So sánh tất cả phương pháp":
                    with st.spinner("Đang tính toán tất cả phương pháp biên..."):
                        edge_results = img_processor.compare_edge_methods(img_bgr)
                    st.markdown("##### Kết quả so sánh:")
                    cols = st.columns(3)
                    for i, (name, img_out) in enumerate(edge_results.items()):
                        with cols[i % 3]:
                            st.image(img_out, caption=name, width="stretch", clamp=True)

                elif adv_edge_mode == "Prewitt (chi tiết X/Y)":
                    result = img_processor.detect_edges_prewitt(img_bgr)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(result["prewitt_x"], caption="Prewitt X (cạnh dọc)", width="stretch", clamp=True)
                    with c2:
                        st.image(result["prewitt_y"], caption="Prewitt Y (cạnh ngang)", width="stretch", clamp=True)
                    with c3:
                        st.image(result["magnitude"], caption="Prewitt Magnitude", width="stretch", clamp=True)
                    st.markdown("**Prewitt** dùng kernel 3×3 với trọng số đều. Đơn giản hơn Sobel nhưng nhạy nhiễu hơn.")

                elif adv_edge_mode == "Roberts Cross (chi tiết)":
                    result = img_processor.detect_edges_roberts(img_bgr)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(result["roberts_cross_1"], caption="Roberts Kernel 1", width="stretch", clamp=True)
                    with c2:
                        st.image(result["roberts_cross_2"], caption="Roberts Kernel 2", width="stretch", clamp=True)
                    with c3:
                        st.image(result["magnitude"], caption="Roberts Magnitude", width="stretch", clamp=True)
                    st.markdown("**Roberts Cross** dùng kernel 2×2 — toán tử biên đơn giản nhất, phát hiện tốt cạnh chéo.")

                elif adv_edge_mode == "Scharr (chi tiết X/Y)":
                    result = img_processor.detect_edges_scharr(img_bgr)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(result["scharr_x"], caption="Scharr X", width="stretch", clamp=True)
                    with c2:
                        st.image(result["scharr_y"], caption="Scharr Y", width="stretch", clamp=True)
                    with c3:
                        st.image(result["magnitude"], caption="Scharr Magnitude", width="stretch", clamp=True)
                    st.markdown("**Scharr** cho xấp xỉ gradient chính xác hơn Sobel 3×3, đặc biệt với cạnh nghiêng.")

                elif adv_edge_mode == "LoG — Laplacian of Gaussian":
                    log_sigma = st.slider("Sigma (σ):", 0.5, 5.0, 1.4, 0.1, key="log_sigma")
                    result = img_processor.detect_edges_log(img_bgr, sigma=log_sigma)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                    with c2:
                        st.image(result, caption=f"LoG (σ={log_sigma})", width="stretch", clamp=True)
                    st.markdown("**LoG** kết hợp Gaussian Blur + Laplacian. σ lớn → cạnh thô, σ nhỏ → cạnh mịn.")

                elif adv_edge_mode == "Auto Canny (tự động ngưỡng)":
                    ac_sigma = st.slider("Sigma (điều chỉnh ngưỡng):", 0.1, 0.8, 0.33, 0.01, key="ac_sigma")
                    auto_canny = img_processor.detect_edges_auto_canny(img_bgr, sigma=ac_sigma)
                    manual_canny = img_processor.detect_edges_canny(img_bgr, 50, 150)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                    with c2:
                        st.image(manual_canny, caption="Canny thủ công (50,150)", width="stretch", clamp=True)
                    with c3:
                        st.image(auto_canny, caption=f"Auto Canny (σ={ac_sigma})", width="stretch", clamp=True)
                    gray_tmp = img_processor.convert_grayscale(img_bgr)
                    v = int(np.median(gray_tmp))
                    st.markdown(f"**Median intensity** = {v} → Low = {int(max(0,(1-ac_sigma)*v))}, High = {int(min(255,(1+ac_sigma)*v))}")

                elif adv_edge_mode == "Zero-Crossing":
                    zc_sigma = st.slider("Sigma LoG:", 0.5, 5.0, 1.5, 0.1, key="zc_sigma")
                    st.warning("⚠️ Zero-Crossing có thể mất vài giây trên ảnh lớn (tính pixel-by-pixel).")
                    with st.spinner("Đang tính Zero-Crossing..."):
                        result = img_processor.detect_edges_zero_crossing(img_bgr, sigma=zc_sigma)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                    with c2:
                        st.image(result, caption=f"Zero-Crossing (σ={zc_sigma})", width="stretch", clamp=True)

        # ----- SUB TAB 9: PHÂN VÙNG ẢNH -----
        with cv_sub_tab9:
            st.markdown("#### 🧩 Phân vùng Ảnh (Image Segmentation)")
            st.info(
                "Các phương pháp: **Otsu**, **Adaptive Threshold**, **Multi-level Threshold**, "
                "**Watershed**, **K-Means**, **GrabCut**, **Mean Shift**, **Region Growing**."
            )

            seg_uploaded = st.file_uploader(
                "Tải ảnh để phân vùng:",
                type=["jpg", "png", "jpeg", "webp"],
                key="seg_upload"
            )

            if seg_uploaded is not None:
                img_pil = Image.open(seg_uploaded)
                img_np = np.array(img_pil)
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                else:
                    img_bgr = img_np

                seg_method = st.selectbox("Chọn phương pháp phân vùng:", [
                    "So sánh tất cả",
                    "Otsu Thresholding",
                    "Adaptive Threshold",
                    "Multi-level Threshold",
                    "Watershed",
                    "K-Means Color",
                    "GrabCut",
                    "Mean Shift",
                    "Region Growing",
                ], key="seg_method")

                if seg_method == "So sánh tất cả":
                    with st.spinner("Đang xử lý tất cả phương pháp phân vùng..."):
                        demo = img_processor.demonstrate_segmentation_methods(img_bgr)
                    cols = st.columns(4)
                    for i, (name, img_out) in enumerate(demo.items()):
                        with cols[i % 4]:
                            if len(img_out.shape) == 2:
                                st.image(img_out, caption=name, width="stretch", clamp=True)
                            else:
                                st.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), caption=name, width="stretch")

                elif seg_method == "Otsu Thresholding":
                    result = img_processor.segment_otsu(img_bgr)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(result["grayscale"], caption="Grayscale", width="stretch", clamp=True)
                    with c2:
                        st.image(result["otsu_binary"], caption=f"Otsu Binary (T={result['threshold_value']:.0f})", width="stretch", clamp=True)
                    with c3:
                        st.image(result["otsu_binary_inv"], caption="Otsu Inverse", width="stretch", clamp=True)
                    st.markdown(f"**Ngưỡng tối ưu tìm được:** T* = {result['threshold_value']:.0f} (tối đa hóa phương sai liên lớp)")

                elif seg_method == "Adaptive Threshold":
                    at_method = st.selectbox("Phương pháp:", ["gaussian", "mean"], key="at_method")
                    at_block = st.slider("Block size:", 3, 31, 11, 2, key="at_block")
                    at_c = st.slider("Hằng số C:", -10, 20, 2, 1, key="at_c")
                    result = img_processor.segment_adaptive_threshold(img_bgr, at_method, at_block, at_c)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(result["grayscale"], caption="Grayscale", width="stretch", clamp=True)
                    with c2:
                        st.image(result["adaptive_binary"], caption="Adaptive Binary", width="stretch", clamp=True)
                    with c3:
                        st.image(result["adaptive_binary_inv"], caption="Adaptive Inverse", width="stretch", clamp=True)

                elif seg_method == "Multi-level Threshold":
                    n_levels = st.slider("Số mức ngưỡng:", 1, 5, 2, 1, key="mt_levels")
                    thresholds = [int(255 * (i + 1) / (n_levels + 1)) for i in range(n_levels)]
                    st.write(f"Các ngưỡng: {thresholds}")
                    result = img_processor.segment_multi_threshold(img_bgr, thresholds)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                    with c2:
                        st.image(result["multi_threshold"], caption=f"Multi-Threshold ({n_levels} mức)", width="stretch", clamp=True)

                elif seg_method == "Watershed":
                    with st.spinner("Đang xử lý Watershed..."):
                        result = img_processor.segment_watershed(img_bgr)
                    st.markdown("##### Các bước Watershed:")
                    cols = st.columns(3)
                    ws_items = [("binary_otsu", "Otsu Binary"), ("distance_transform", "Distance Transform"),
                                ("sure_foreground", "Sure Foreground"), ("sure_background", "Sure Background"),
                                ("watershed_boundaries", "Watershed Boundaries"), ("colored_segments", "Colored Segments")]
                    for i, (key, label) in enumerate(ws_items):
                        with cols[i % 3]:
                            img_out = result[key]
                            if len(img_out.shape) == 2:
                                st.image(img_out, caption=label, width="stretch", clamp=True)
                            else:
                                st.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), caption=label, width="stretch")

                elif seg_method == "K-Means Color":
                    if len(img_bgr.shape) == 3:
                        km_k = st.slider("Số cụm K:", 2, 10, 4, 1, key="km_k")
                        with st.spinner(f"Đang chạy K-Means (K={km_k})..."):
                            result = img_processor.segment_kmeans_color(img_bgr, k=km_k)
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                        with c2:
                            st.image(cv2.cvtColor(result["segmented"], cv2.COLOR_BGR2RGB), caption=f"K-Means (K={km_k})", width="stretch")
                        with c3:
                            st.image(result["label_map"], caption="Label Map", width="stretch", clamp=True)

                        with st.expander("Xem mask từng cụm"):
                            mask_cols = st.columns(min(km_k, 4))
                            for i, (cname, cmask) in enumerate(result["cluster_masks"].items()):
                                with mask_cols[i % min(km_k, 4)]:
                                    st.image(cmask, caption=cname, width="stretch", clamp=True)
                    else:
                        st.warning("K-Means Color chỉ áp dụng cho ảnh màu.")

                elif seg_method == "GrabCut":
                    if len(img_bgr.shape) == 3:
                        gc_iter = st.slider("Số iteration:", 1, 15, 5, 1, key="gc_iter")
                        with st.spinner("Đang chạy GrabCut..."):
                            result = img_processor.segment_grabcut(img_bgr, iterations=gc_iter)
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.image(cv2.cvtColor(result["roi_visualization"], cv2.COLOR_BGR2RGB), caption="ROI (vùng khởi tạo)", width="stretch")
                        with c2:
                            st.image(result["foreground_mask"], caption="Foreground Mask", width="stretch", clamp=True)
                        with c3:
                            st.image(cv2.cvtColor(result["result"], cv2.COLOR_BGR2RGB), caption="GrabCut Result", width="stretch")
                    else:
                        st.warning("GrabCut chỉ áp dụng cho ảnh màu.")

                elif seg_method == "Mean Shift":
                    if len(img_bgr.shape) == 3:
                        ms_sp = st.slider("Spatial radius:", 5, 50, 20, 5, key="ms_sp")
                        ms_sr = st.slider("Color radius:", 10, 100, 40, 5, key="ms_sr")
                        with st.spinner("Đang chạy Mean Shift..."):
                            result = img_processor.segment_mean_shift(img_bgr, sp=ms_sp, sr=ms_sr)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                        with c2:
                            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Mean Shift (sp={ms_sp}, sr={ms_sr})", width="stretch")
                    else:
                        st.warning("Mean Shift chỉ áp dụng cho ảnh màu.")

                elif seg_method == "Region Growing":
                    h_img, w_img = img_bgr.shape[:2]
                    rg_y = st.slider("Seed Y (hàng):", 0, h_img - 1, h_img // 2, 1, key="rg_y")
                    rg_x = st.slider("Seed X (cột):", 0, w_img - 1, w_img // 2, 1, key="rg_x")
                    rg_thresh = st.slider("Threshold:", 1, 50, 15, 1, key="rg_thresh")
                    with st.spinner("Đang chạy Region Growing..."):
                        mask_rg = img_processor.segment_region_growing(img_bgr, (rg_y, rg_x), rg_thresh)
                    c1, c2 = st.columns(2)
                    img_seed = img_bgr.copy()
                    cv2.circle(img_seed, (rg_x, rg_y), 6, (0, 0, 255), -1)
                    cv2.putText(img_seed, "Seed", (rg_x + 10, rg_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    with c1:
                        st.image(cv2.cvtColor(img_seed, cv2.COLOR_BGR2RGB), caption="Ảnh gốc + Seed", width="stretch")
                    with c2:
                        st.image(mask_rg, caption=f"Region Growing (T={rg_thresh})", width="stretch", clamp=True)
                    n_pixels = np.sum(mask_rg > 0)
                    st.markdown(f"**Vùng phát triển:** {n_pixels:,} pixels ({n_pixels / (h_img * w_img) * 100:.1f}% ảnh)")

        # ----- SUB TAB 10: NÉN DỮ LIỆU ẢNH -----
        with cv_sub_tab10:
            st.markdown("#### 📦 Nén Dữ liệu Ảnh (Image Compression)")
            st.info(
                "Các phương pháp: **JPEG** (lossy), **PNG** (lossless), "
                "**DCT** (giáo dục), **SVD** (phân tích giá trị kỳ dị), **RLE** (mã hóa chiều dài chạy)."
            )

            compress_uploaded = st.file_uploader(
                "Tải ảnh để nén:",
                type=["jpg", "png", "jpeg", "webp"],
                key="compress_upload"
            )

            if compress_uploaded is not None:
                img_pil = Image.open(compress_uploaded)
                img_np = np.array(img_pil)
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                else:
                    img_bgr = img_np

                compress_method = st.selectbox("Chọn phương pháp nén:", [
                    "JPEG (Lossy)",
                    "PNG (Lossless)",
                    "DCT — Giáo dục (Discrete Cosine Transform)",
                    "SVD (Singular Value Decomposition)",
                    "RLE (Run-Length Encoding)",
                    "So sánh tất cả",
                ], key="compress_method")

                if compress_method == "JPEG (Lossy)":
                    jpeg_q = st.slider("Chất lượng JPEG (1-100):", 1, 100, 50, 1, key="jpeg_q")
                    result = img_processor.compress_jpeg(img_bgr, quality=jpeg_q)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
                    with c2:
                        st.image(cv2.cvtColor(result["compressed_image"], cv2.COLOR_BGR2RGB), caption=f"JPEG Q={jpeg_q}", width="stretch")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Tỷ lệ nén", f"{result['compression_ratio']}x")
                    mc2.metric("PSNR", f"{result['psnr_db']} dB")
                    mc3.metric("MSE", f"{result['mse']:.2f}")
                    mc4.metric("Kích thước nén", f"{result['compressed_size_bytes']:,} bytes")

                elif compress_method == "PNG (Lossless)":
                    png_level = st.slider("Mức nén PNG (0-9):", 0, 9, 3, 1, key="png_level")
                    result = img_processor.compress_png(img_bgr, compression_level=png_level)
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc (không thay đổi — lossless)", width="stretch")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Tỷ lệ nén", f"{result['compression_ratio']}x")
                    mc2.metric("Kích thước gốc", f"{result['original_size_bytes']:,} bytes")
                    mc3.metric("Kích thước nén", f"{result['compressed_size_bytes']:,} bytes")
                    st.success("PNG là nén **không tổn hao** → PSNR = ∞, MSE = 0")

                elif compress_method == "DCT — Giáo dục (Discrete Cosine Transform)":
                    dct_qf = st.slider("Quality Factor (1-100):", 1, 100, 50, 1, key="dct_qf")
                    with st.spinner("Đang xử lý DCT block 8×8..."):
                        result = img_processor.compress_dct_educational(img_bgr, quality_factor=dct_qf)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(result["original_gray"], caption="Ảnh xám gốc", width="stretch", clamp=True)
                    with c2:
                        st.image(result["compressed_result"], caption=f"DCT QF={dct_qf}", width="stretch", clamp=True)
                    c3, c4 = st.columns(2)
                    with c3:
                        st.image(result["dct_coefficients"], caption="Hệ số DCT (log scale)", width="stretch", clamp=True)
                    with c4:
                        st.image(result["error_map"], caption="Error Map |Original - Compressed|", width="stretch", clamp=True)
                    st.markdown(f"""**Giải thích:** Quality Factor = {dct_qf}.
                    {'Nén mạnh → nhiều artifact block, mất chi tiết' if dct_qf < 30 else 'Cân bằng chất lượng/nén' if dct_qf < 70 else 'Nén nhẹ → giữ gần như toàn bộ chi tiết'}.""")

                elif compress_method == "SVD (Singular Value Decomposition)":
                    gray = img_processor.convert_grayscale(img_bgr)
                    max_k = min(gray.shape)
                    svd_k = st.slider("Số singular values (k):", 1, min(max_k, 200), 50, 1, key="svd_k")
                    with st.spinner(f"Đang phân tích SVD (k={svd_k})..."):
                        result = img_processor.compress_svd(img_bgr, k=svd_k)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(result["original_gray"], caption="Ảnh xám gốc", width="stretch", clamp=True)
                    with c2:
                        st.image(result["compressed_result"], caption=f"SVD k={svd_k}", width="stretch", clamp=True)
                    c3, c4 = st.columns(2)
                    with c3:
                        st.image(result["error_map"], caption="Error Map", width="stretch", clamp=True)
                    with c4:
                        sv_plot = result["singular_values_plot"]
                        st.image(cv2.cvtColor(sv_plot, cv2.COLOR_BGR2RGB), caption="Singular Values", width="stretch")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Tỷ lệ nén", f"{result['compression_ratio']}x")
                    mc2.metric("PSNR", f"{result['psnr_db']} dB")
                    mc3.metric("MSE", f"{result['mse']:.2f}")

                elif compress_method == "RLE (Run-Length Encoding)":
                    result = img_processor.compress_rle(img_bgr)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(result["original_gray"], caption="Ảnh xám gốc", width="stretch", clamp=True)
                    with c2:
                        st.image(result["decoded_image"], caption="Giải nén RLE (lossless)", width="stretch", clamp=True)
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Tỷ lệ nén", f"{result['compression_ratio']}x")
                    mc2.metric("Số runs", f"{result['num_runs']:,}")
                    mc3.metric("Kích thước gốc", f"{result['original_size_bytes']:,} B")
                    mc4.metric("Kích thước nén", f"{result['compressed_size_bytes']:,} B")
                    st.info(f"**Lossless:** {result['is_lossless']} — RLE hiệu quả khi ảnh có nhiều vùng đồng nhất.")

                elif compress_method == "So sánh tất cả":
                    with st.spinner("Đang chạy so sánh tất cả phương pháp nén..."):
                        all_results = img_processor.analyze_compression_methods(img_bgr)
                    table_data = []
                    for name, info in all_results.items():
                        table_data.append({
                            "Phương pháp": name,
                            "Loại nén": info["type"],
                            "Tỷ lệ nén": str(info["compression_ratio"]),
                            "PSNR (dB)": f"{info['psnr_db']:.1f}" if info['psnr_db'] != float('inf') else "∞",
                            "MSE": f"{info['mse']:.2f}",
                        })
                    st.markdown("##### Bảng so sánh các phương pháp nén:")
                    st.dataframe(pd.DataFrame(table_data), hide_index=True, width="stretch")

                    st.markdown("##### Ảnh JPEG ở các mức chất lượng:")
                    jpeg_cols = st.columns(5)
                    for i, q in enumerate([10, 30, 50, 70, 90]):
                        key_name = f"JPEG (Q={q})"
                        if key_name in all_results:
                            with jpeg_cols[i]:
                                comp_img = all_results[key_name]["compressed_image"]
                                if len(comp_img.shape) == 2:
                                    st.image(comp_img, caption=key_name, width="stretch", clamp=True)
                                else:
                                    st.image(cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB), caption=key_name, width="stretch")

        # ----- SUB TAB 11: XỬ LÝ ẢNH NHIỄU -----
        with cv_sub_tab11:
            st.markdown("#### 📡 Xử lý Ảnh Nhiễu & Khôi phục Ảnh (Image Noise & Restoration)")
            st.info(
                "Mô phỏng 5 mô hình nhiễu thực tế (**Gaussian**, **Muối-Tiêu**, **Poisson**, **Speckle**, **Đều**) "
                "và áp dụng các bộ lọc phục hồi chuyên sâu (**NLM**, **Bilateral**, **Median**, **Gaussian**, **Mean**). "
                "Đánh giá định lượng hiệu quả lọc bằng chỉ số **PSNR (dB)** và **MSE**."
            )

            noise_uploaded = st.file_uploader(
                "Tải ảnh để thử nghiệm xử lý nhiễu:",
                type=["jpg", "png", "jpeg", "webp"],
                key="noise_upload"
            )

            use_sample_noise = False
            sample_noise_path = "data/samples/traffic.jpg"
            if noise_uploaded is None and os.path.exists(sample_noise_path):
                use_sample_noise = st.checkbox("Sử dụng ảnh mẫu giao thông (`data/samples/traffic.jpg`)", value=True, key="noise_use_sample")

            img_bgr = None
            if noise_uploaded is not None:
                img_pil = Image.open(noise_uploaded)
                img_np = np.array(img_pil)
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                else:
                    img_bgr = img_np
            elif use_sample_noise:
                img_bgr = cv2.imread(sample_noise_path)

            if img_bgr is not None:
                noise_mode = st.segmented_control(
                    "Chế độ hiển thị:",
                    options=["🔬 Thử nghiệm Tương tác", "📊 So sánh Toàn diện (Benchmark)"],
                    default="🔬 Thử nghiệm Tương tác",
                    key="noise_mode"
                )

                if noise_mode == "🔬 Thử nghiệm Tương tác":
                    col_noise_cfg, col_filter_cfg = st.columns(2)
                    with col_noise_cfg:
                        with st.container(border=True):
                            st.markdown("##### 1. Mô hình hóa & Tạo Nhiễu")
                            noise_type = st.selectbox(
                                "Chọn loại nhiễu:",
                                [
                                    "Gaussian Noise (Nhiễu chuẩn Gauss)",
                                    "Salt-and-Pepper Noise (Nhiễu Muối-Tiêu)",
                                    "Poisson Noise (Nhiễu Poisson / Hạt ánh sáng)",
                                    "Speckle Noise (Nhiễu Đốm / Nhân tính)",
                                    "Uniform Noise (Nhiễu Đều)",
                                ],
                                key="noise_type_select"
                            )

                            if noise_type.startswith("Gaussian"):
                                g_mean = st.slider("Giá trị kỳ vọng (mean μ):", -20.0, 20.0, 0.0, 1.0, key="n_g_mean")
                                g_sigma = st.slider("Độ lệch chuẩn (sigma σ):", 5.0, 100.0, 25.0, 1.0, key="n_g_sigma")
                                noisy_img = img_processor.add_gaussian_noise(img_bgr, mean=g_mean, sigma=g_sigma)
                                noise_desc = f"Nhiễu Gaussian: $I_{{noisy}} = I + \\mathcal{{N}}({g_mean}, {g_sigma}^2)$ — mô phỏng nhiễu cảm biến nhiệt/điện tử."

                            elif noise_type.startswith("Salt-and-Pepper"):
                                sp_amt = st.slider("Tỷ lệ điểm ảnh bị nhiễu (amount):", 0.01, 0.30, 0.05, 0.01, key="n_sp_amt")
                                sp_ratio = st.slider("Tỷ lệ Muối (trắng) / Tiêu (đen):", 0.1, 0.9, 0.5, 0.05, key="n_sp_ratio")
                                noisy_img = img_processor.add_salt_pepper_noise(img_bgr, amount=sp_amt, salt_ratio=sp_ratio)
                                noise_desc = f"Nhiễu Muối-Tiêu (Impulse): {sp_amt*100:.1f}% pixel bị thay bằng 0 hoặc 255 do lỗi truyền tín hiệu."

                            elif noise_type.startswith("Poisson"):
                                st.caption("Nhiễu Poisson phụ thuộc vào mức sáng: vùng sáng có phương sai nhiễu lớn hơn theo thống kê photon.")
                                noisy_img = img_processor.add_poisson_noise(img_bgr)
                                noise_desc = "Nhiễu Poisson (Shot Noise): Phụ thuộc tín hiệu, phổ biến khi chụp đêm hoặc tốc độ màn trập cao."

                            elif noise_type.startswith("Speckle"):
                                sk_sigma = st.slider("Độ lệch chuẩn nhiễu nhân (σ):", 0.05, 0.60, 0.20, 0.05, key="n_sk_sigma")
                                noisy_img = img_processor.add_speckle_noise(img_bgr, sigma=sk_sigma)
                                noise_desc = f"Nhiễu Speckle: $I_{{noisy}} = I + I \\times \\mathcal{{N}}(0, {sk_sigma}^2)$ — nhiễu nhân trong radar, siêu âm, thời tiết xấu."

                            else:
                                u_range = st.slider("Biên độ nhiễu đều (±A):", 5.0, 80.0, 30.0, 5.0, key="n_u_range")
                                noisy_img = img_processor.add_uniform_noise(img_bgr, low=-u_range, high=u_range)
                                noise_desc = f"Nhiễu Đều: $I_{{noisy}} = I + \\mathcal{{U}}(-{u_range}, +{u_range})$."

                    with col_filter_cfg:
                        with st.container(border=True):
                            st.markdown("##### 2. Lựa chọn Bộ lọc Khôi phục")
                            default_filter_idx = 0
                            if noise_type.startswith("Salt-and-Pepper"):
                                default_filter_idx = 1
                            elif noise_type.startswith("Speckle"):
                                default_filter_idx = 2

                            denoise_method = st.selectbox(
                                "Phương pháp lọc khử nhiễu:",
                                [
                                    "Non-Local Means (NLM - Chuẩn cao cấp)",
                                    "Median Filter (Lọc Trung vị - Tối ưu cho Muối-Tiêu)",
                                    "Bilateral Filter (Lọc Song phương - Bảo toàn cạnh)",
                                    "Gaussian Blur (Lọc Gauss làm mịn)",
                                    "Mean Filter (Lọc Trung bình)",
                                ],
                                index=default_filter_idx,
                                key="denoise_method_select"
                            )

                            if denoise_method.startswith("Non-Local Means"):
                                dn_h = st.slider("Cường độ làm mịn (h):", 3.0, 30.0, 10.0, 1.0, key="dn_h_slider")
                                denoised_img = img_processor.apply_denoising(noisy_img, method="nlm", h=dn_h)
                                filter_name = f"NLM (h={dn_h})"
                            elif denoise_method.startswith("Median"):
                                k_med = st.selectbox("Kích thước kernel (ksize):", [3, 5, 7, 9], index=1, key="k_med_select")
                                denoised_img = img_processor.apply_denoising(noisy_img, method="median", ksize=k_med)
                                filter_name = f"Median ({k_med}×{k_med})"
                            elif denoise_method.startswith("Bilateral"):
                                b_d = st.selectbox("Bán kính láng giềng (d):", [5, 9, 15], index=1, key="b_d_select")
                                denoised_img = img_processor.apply_bilateral_filter(noisy_img, d=b_d, sigma_color=75, sigma_space=75)
                                filter_name = f"Bilateral (d={b_d})"
                            elif denoise_method.startswith("Gaussian"):
                                k_gauss = st.selectbox("Kích thước kernel (ksize):", [3, 5, 7, 9], index=1, key="k_gauss_select")
                                denoised_img = img_processor.apply_gaussian_blur(noisy_img, ksize=k_gauss)
                                filter_name = f"Gaussian Blur ({k_gauss}×{k_gauss})"
                            else:
                                k_mean = st.selectbox("Kích thước kernel (ksize):", [3, 5, 7, 9], index=1, key="k_mean_select")
                                denoised_img = img_processor.apply_denoising(noisy_img, method="mean", ksize=k_mean)
                                filter_name = f"Mean Filter ({k_mean}×{k_mean})"

                    m_noisy = img_processor.compute_psnr_mse(img_bgr, noisy_img)
                    m_denoised = img_processor.compute_psnr_mse(img_bgr, denoised_img)
                    delta_psnr = round(m_denoised["psnr"] - m_noisy["psnr"], 2) if m_noisy["psnr"] != float('inf') and m_denoised["psnr"] != float('inf') else 0.0

                    c_orig, c_noisy, c_clean = st.columns(3)
                    with c_orig:
                        st.markdown("**1. Ảnh gốc ban đầu**")
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc (Ground Truth)", width="stretch")
                        st.metric("PSNR / MSE chuẩn", "Gốc (Chuẩn)", "0.0 MSE")

                    with c_noisy:
                        st.markdown("**2. Ảnh sau khi thêm nhiễu**")
                        st.image(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB), caption=f"Ảnh nhiễu ({noise_type.split(' (')[0]})", width="stretch")
                        st.metric("PSNR ảnh nhiễu", f"{m_noisy['psnr']} dB", f"MSE: {m_noisy['mse']:.1f}", delta_color="inverse")

                    with c_clean:
                        st.markdown(f"**3. Sau khi khử nhiễu: `{filter_name}`**")
                        st.image(cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB), caption=f"Đã khử nhiễu ({filter_name})", width="stretch")
                        st.metric(
                            "PSNR sau phục hồi",
                            f"{m_denoised['psnr']} dB",
                            f"{'+' if delta_psnr >= 0 else ''}{delta_psnr} dB (MSE: {m_denoised['mse']:.1f})",
                            delta_color="normal"
                        )

                    st.caption(f"ℹ️ **Ghi chú mô hình:** {noise_desc}")

                    with st.expander("🔍 Xem bản đồ sai số (Error Maps) & Phân tích chi tiết"):
                        em1, em2 = st.columns(2)
                        with em1:
                            err_noisy = cv2.absdiff(img_bgr, noisy_img)
                            err_noisy_gray = cv2.cvtColor(err_noisy, cv2.COLOR_BGR2GRAY) if len(err_noisy.shape) == 3 else err_noisy
                            st.image(cv2.applyColorMap(err_noisy_gray, cv2.COLORMAP_JET), caption="Bản đồ nhiễu |Ảnh nhiễu - Gốc| (JET Colormap)", width="stretch")
                        with em2:
                            err_denoised = cv2.absdiff(img_bgr, denoised_img)
                            err_denoised_gray = cv2.cvtColor(err_denoised, cv2.COLOR_BGR2GRAY) if len(err_denoised.shape) == 3 else err_denoised
                            st.image(cv2.applyColorMap(err_denoised_gray, cv2.COLORMAP_JET), caption="Bản đồ sai số còn lại |Ảnh sau lọc - Gốc| (JET Colormap)", width="stretch")

                else:
                    st.markdown("##### 📊 So sánh Hiệu năng Khử nhiễu Toàn diện trên 5 Loại Nhiễu")
                    st.caption("Chạy tự động mô phỏng 5 loại nhiễu và áp dụng bộ lọc tối ưu nhất theo lý thuyết xử lý ảnh số.")

                    with st.spinner("Đang xử lý mô phỏng và khử nhiễu toàn diện..."):
                        bench_res = img_processor.demonstrate_noise_processing(img_bgr)

                    table_rows = []
                    filter_reasons = {
                        "Gaussian": "NLM so sánh cấu trúc patch toàn ảnh → giữ chi tiết bề mặt tốt nhất.",
                        "Salt-and-Pepper": "Median loại bỏ hoàn toàn các giá trị cực trị (0 và 255) mà không làm mờ viền.",
                        "Poisson": "NLM thích ứng tốt với nhiễu phụ thuộc tín hiệu photon.",
                        "Speckle": "Bilateral kết hợp lọc không gian và cường độ → bảo toàn cạnh đối tượng giao thông.",
                        "Uniform": "Gaussian blur làm mượt phân phối đều hiệu quả.",
                    }

                    for n_name, data in bench_res.items():
                        table_rows.append({
                            "Loại Nhiễu": n_name,
                            "Bộ lọc Khuyên dùng": data["filter"],
                            "PSNR Nhiễu (dB)": f"{data['psnr_noisy']:.2f}",
                            "PSNR Sau Lọc (dB)": f"{data['psnr_denoised']:.2f}",
                            "Cải thiện ΔPSNR": f"+{data['delta_psnr']:.2f} dB" if data['delta_psnr'] >= 0 else f"{data['delta_psnr']:.2f} dB",
                            "MSE Nhiễu → Lọc": f"{data['mse_noisy']:.1f} → {data['mse_denoised']:.1f}",
                            "Lý do Khuyến nghị": filter_reasons.get(n_name, "")
                        })

                    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

                    st.markdown("##### Lưới đối chiếu trực quan (Ảnh nhiễu vs Ảnh sau lọc):")
                    for n_name, data in bench_res.items():
                        with st.container(border=True):
                            c_info, c_n, c_d = st.columns([1, 2, 2])
                            with c_info:
                                st.markdown(f"### {n_name}")
                                st.markdown(f"**Bộ lọc:** `{data['filter']}`")
                                st.metric("Cải thiện PSNR", f"+{data['delta_psnr']:.2f} dB")
                                st.caption(filter_reasons.get(n_name, ""))
                            with c_n:
                                st.image(cv2.cvtColor(data["noisy"], cv2.COLOR_BGR2RGB), caption=f"Ảnh Nhiễu {n_name} (PSNR: {data['psnr_noisy']} dB)", width="stretch")
                            with c_d:
                                st.image(cv2.cvtColor(data["denoised"], cv2.COLOR_BGR2RGB), caption=f"Đã Lọc: {data['filter']} (PSNR: {data['psnr_denoised']} dB)", width="stretch")
            else:
                st.info("👆 Vui lòng tải lên một bức ảnh giao thông hoặc chọn sử dụng ảnh mẫu để bắt đầu thử nghiệm xử lý ảnh nhiễu.")



    # ----------------------------------------------------
    # TAB 4: ĐÁNH GIÁ & LỊCH SỬ HUẤN LUYỆN
    # ----------------------------------------------------
    with tab_analytics:
        st.subheader("📈 Dashboard Đánh giá Mô hình & Lịch sử Huấn luyện")
        st.caption("Phân tích chi tiết quá trình huấn luyện, đánh giá hiệu suất và so sánh giữa các thử nghiệm — thể hiện đóng góp nghiên cứu thực nghiệm cá nhân.")

        # --- Chọn thí nghiệm để phân tích ---
        runs_base = "runs/detect"
        available_runs = []
        if os.path.exists(runs_base):
            for d in sorted(os.listdir(runs_base)):
                run_csv = os.path.join(runs_base, d, "results.csv")
                if os.path.isdir(os.path.join(runs_base, d)) and os.path.exists(run_csv):
                    df_check = pd.read_csv(run_csv)
                    if len(df_check) > 1:
                        available_runs.append(d)

        if not available_runs:
            st.warning("Chưa tìm thấy kết quả huấn luyện nào trong `runs/detect/`. Hãy chạy `python train.py` trước.")
        else:
            selected_run = st.selectbox(
                "Chọn phiên bản thí nghiệm để phân tích:",
                available_runs,
                index=len(available_runs) - 1,
                format_func=lambda x: f"🧪 {x}"
            )
            assert selected_run is not None
            selected_run_dir = os.path.join(runs_base, selected_run)
            df_results = load_training_results(selected_run_dir)

            if df_results is not None and not df_results.empty:
                # ====================================================
                # PHẦN 1: KPI TỔNG QUAN
                # ====================================================
                st.markdown("---")
                st.markdown("#### 📊 Chỉ số hiệu suất tổng quan")

                last = df_results.iloc[-1]
                first = df_results.iloc[0]

                map50_val = float(last.get("metrics/mAP50(B)", 0))
                map50_first = float(first.get("metrics/mAP50(B)", 0))
                map95_val = float(last.get("metrics/mAP50-95(B)", 0))
                map95_first = float(first.get("metrics/mAP50-95(B)", 0))
                prec_val = float(last.get("metrics/precision(B)", 0))
                prec_first = float(first.get("metrics/precision(B)", 0))
                recall_val = float(last.get("metrics/recall(B)", 0))
                recall_first = float(first.get("metrics/recall(B)", 0))

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("mAP@50", f"{map50_val:.1%}", delta=f"{map50_val - map50_first:+.1%}")
                kpi2.metric("mAP@50-95", f"{map95_val:.1%}", delta=f"{map95_val - map95_first:+.1%}")
                kpi3.metric("Precision", f"{prec_val:.1%}", delta=f"{prec_val - prec_first:+.1%}")
                kpi4.metric("Recall", f"{recall_val:.1%}", delta=f"{recall_val - recall_first:+.1%}")

                # ====================================================
                # PHẦN 2: BIỂU ĐỒ LOSS & METRICS
                # ====================================================
                st.markdown("---")
                st.markdown("#### 📉 Biểu đồ quá trình huấn luyện")

                chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Hàm mất mát (Loss)", "Độ chính xác (Accuracy)", "Tốc độ học (LR)"])

                with chart_tab1:
                    col_loss1, col_loss2 = st.columns(2)
                    with col_loss1:
                        st.markdown("**Train Loss** (Sai số trên tập huấn luyện)")
                        df_train_loss = df_results[["epoch"]].copy()
                        df_train_loss["Box Loss"] = df_results["train/box_loss"]
                        df_train_loss["Class Loss"] = df_results["train/cls_loss"]
                        df_train_loss["DFL Loss"] = df_results["train/dfl_loss"]
                        st.line_chart(df_train_loss, x="epoch", y=["Box Loss", "Class Loss", "DFL Loss"], color=["#FF6B6B", "#4ECDC4", "#45B7D1"])

                    with col_loss2:
                        st.markdown("**Validation Loss** (Sai số trên tập kiểm tra)")
                        df_val_loss = df_results[["epoch"]].copy()
                        df_val_loss["Box Loss"] = df_results["val/box_loss"]
                        df_val_loss["Class Loss"] = df_results["val/cls_loss"]
                        df_val_loss["DFL Loss"] = df_results["val/dfl_loss"]
                        st.line_chart(df_val_loss, x="epoch", y=["Box Loss", "Class Loss", "DFL Loss"], color=["#FF6B6B", "#4ECDC4", "#45B7D1"])

                    train_cls_last = float(last.get("train/cls_loss", 0))
                    val_cls_last = float(last.get("val/cls_loss", 0))
                    gap = val_cls_last - train_cls_last
                    if gap < 0.15:
                        st.success(f"✅ **Không có dấu hiệu Overfitting.** Khoảng cách Train-Val cls_loss = {gap:.3f} (< 0.15). Mô hình tổng quát hóa tốt.")
                    elif gap < 0.3:
                        st.warning(f"⚠️ **Overfitting nhẹ.** Khoảng cách Train-Val cls_loss = {gap:.3f}. Có thể cần thêm Data Augmentation hoặc giảm epochs.")
                    else:
                        st.error(f"🚨 **Overfitting nặng!** Khoảng cách Train-Val cls_loss = {gap:.3f}. Cần Dropout, tăng dữ liệu hoặc Early Stopping.")

                with chart_tab2:
                    st.markdown("**Sự tăng trưởng các chỉ số đánh giá qua từng epoch**")
                    df_metrics = df_results[["epoch"]].copy()
                    df_metrics["mAP@50"] = df_results["metrics/mAP50(B)"]
                    df_metrics["mAP@50-95"] = df_results["metrics/mAP50-95(B)"]
                    df_metrics["Precision"] = df_results["metrics/precision(B)"]
                    df_metrics["Recall"] = df_results["metrics/recall(B)"]
                    st.line_chart(df_metrics, x="epoch", y=["mAP@50", "mAP@50-95", "Precision", "Recall"], color=["#FFD93D", "#6BCB77", "#4D96FF", "#FF6B6B"])

                    st.info(
                        f"📌 **Phân tích:** mAP@50 tăng từ **{map50_first:.1%}** lên **{map50_val:.1%}** "
                        f"(+{map50_val - map50_first:.1%}) sau {len(df_results)} epochs. "
                        f"Recall cải thiện từ {recall_first:.1%} → {recall_val:.1%}, cho thấy mô hình ngày càng ít bỏ sót phương tiện."
                    )

                with chart_tab3:
                    st.markdown("**Learning Rate Schedule** (Chiến lược điều chỉnh tốc độ học)")
                    df_lr = df_results[["epoch"]].copy()
                    df_lr["Learning Rate (pg0)"] = df_results["lr/pg0"]
                    st.line_chart(df_lr, x="epoch", y="Learning Rate (pg0)", color=["#A855F7"])

                    st.info(
                        "📌 **Giải thích:** Learning Rate tăng dần trong 3 epoch đầu (Warm-up), "
                        "sau đó giảm dần theo lịch trình Cosine Annealing để mô hình hội tụ ổn định vào vùng tối ưu."
                    )

                # ====================================================
                # PHẦN 3: CONFUSION MATRIX & PR CURVE
                # ====================================================
                st.markdown("---")
                st.markdown("#### 🔍 Ma trận nhầm lẫn & Đường cong PR")

                col_cm, col_pr = st.columns(2)

                cm_norm_path = os.path.join(selected_run_dir, "confusion_matrix_normalized.png")
                cm_path = os.path.join(selected_run_dir, "confusion_matrix.png")
                with col_cm:
                    st.markdown("**Confusion Matrix (Ma trận nhầm lẫn)**")
                    if os.path.exists(cm_norm_path):
                        st.image(cm_norm_path, caption="Ma trận nhầm lẫn chuẩn hóa (Normalized)", width="stretch")
                    elif os.path.exists(cm_path):
                        st.image(cm_path, caption="Ma trận nhầm lẫn (Số lượng tuyệt đối)", width="stretch")
                    else:
                        st.info("Chưa có dữ liệu Confusion Matrix cho thí nghiệm này.")

                pr_path = os.path.join(selected_run_dir, "BoxPR_curve.png")
                with col_pr:
                    st.markdown("**Precision-Recall Curve (Đường cong PR)**")
                    if os.path.exists(pr_path):
                        st.image(pr_path, caption="Đường cong Precision-Recall theo từng lớp phương tiện", width="stretch")
                    else:
                        st.info("Chưa có dữ liệu PR Curve cho thí nghiệm này.")

                f1_path = os.path.join(selected_run_dir, "BoxF1_curve.png")
                if os.path.exists(f1_path):
                    with st.expander("🔎 Xem thêm: F1-Score Curve"):
                        st.image(f1_path, caption="F1-Score Curve — Điểm cân bằng giữa Precision và Recall", width="stretch")

                # Phân tích tự động
                with st.expander("🔬 Xem phân tích nhận xét tự động", expanded=False):
                    st.markdown(f"""
**Nhận xét về kết quả huấn luyện `{selected_run}`:**

| Chỉ số | Epoch đầu | Epoch cuối | Biến thiên |
| :--- | :---: | :---: | :---: |
| **mAP@50** | {map50_first:.1%} | {map50_val:.1%} | {map50_val - map50_first:+.1%} |
| **mAP@50-95** | {map95_first:.1%} | {map95_val:.1%} | {map95_val - map95_first:+.1%} |
| **Precision** | {prec_first:.1%} | {prec_val:.1%} | {prec_val - prec_first:+.1%} |
| **Recall** | {recall_first:.1%} | {recall_val:.1%} | {recall_val - recall_first:+.1%} |

**Kết luận:**
- Mô hình đạt mAP@50 = **{map50_val:.1%}**, cho thấy khả năng nhận diện phương tiện {'**rất tốt**' if map50_val > 0.85 else '**khá tốt**' if map50_val > 0.7 else '**cần cải thiện thêm**'}.
- {'Không phát hiện dấu hiệu Overfitting — mô hình tổng quát hóa tốt trên tập validation.' if gap < 0.15 else 'Có dấu hiệu Overfitting nhẹ — nên xem xét Early Stopping hoặc thêm Data Augmentation.'}
- Recall đạt **{recall_val:.1%}**, nghĩa là mô hình phát hiện được {recall_val:.0%} số phương tiện thực tế có trong ảnh.
                    """)

                # ====================================================
                # PHẦN 4: BẢNG SO SÁNH (ABLATION STUDY)
                # ====================================================
                st.markdown("---")
                st.markdown("#### 🔄 So sánh các phiên bản thí nghiệm (Ablation Study)")
                st.caption("Bảng dưới đây tổng hợp tất cả các lần huấn luyện bạn đã thực hiện, giúp thể hiện quá trình thử nghiệm và tối ưu hóa mô hình.")

                df_experiments = load_all_experiments(runs_base)

                if not df_experiments.empty:
                    best_idx = df_experiments["mAP@50 (cuối)"].idxmax()

                    st.dataframe(
                        df_experiments.style.highlight_max(
                            subset=["mAP@50 (cuối)", "mAP@50-95 (cuối)", "Precision (cuối)", "Recall (cuối)"],
                            color="#2D6A4F"
                        ).format({
                            "mAP@50 (cuối)": "{:.2%}",
                            "mAP@50-95 (cuối)": "{:.2%}",
                            "Precision (cuối)": "{:.2%}",
                            "Recall (cuối)": "{:.2%}",
                        }),
                        hide_index=True,
                        width="stretch",
                    )

                    best_run = df_experiments.iloc[best_idx]
                    st.success(
                        f"🏆 **Thí nghiệm tốt nhất: `{best_run['Tên thí nghiệm']}`** — "
                        f"mAP@50 = {best_run['mAP@50 (cuối)']:.2%}, "
                        f"mAP@50-95 = {best_run['mAP@50-95 (cuối)']:.2%}, "
                        f"sau {int(best_run['Số epoch'])} epochs "
                        f"({best_run['Thời gian (phút)']:.0f} phút)."
                    )

                    with st.expander("👤 Xem phân tích đóng góp cá nhân"):
                        st.markdown(f"""
**Đóng góp nghiên cứu thực nghiệm cá nhân:**

1. **Thu thập & chuẩn hóa dữ liệu:** Xây dựng tập dữ liệu gồm **4,783 ảnh** giao thông Việt Nam với hơn **32,460 nhãn** phương tiện, được chia theo tỷ lệ Train/Val/Test = 87.4%/8.4%/4.2%.

2. **Thử nghiệm nhiều cấu hình (Hyperparameter Tuning):** Đã thực hiện **{len(df_experiments)} lần thí nghiệm** với các biến thể:
   - Thay đổi mô hình gốc (YOLOv8n vs YOLOv8s)
   - Điều chỉnh số epoch, batch size, image size
   - Tối ưu Data Augmentation (Mixup, Copy-Paste, Rotation)
   - Áp dụng Cosine Annealing Learning Rate Schedule

3. **Phân tích lỗi (Error Analysis):** Nhận diện vấn đề mất cân bằng dữ liệu (Class Imbalance) giữa xe máy (17,046 mẫu) và xe tải (361 mẫu), giải thích sự chênh lệch mAP giữa các lớp.

4. **Đánh giá toàn diện:** Xây dựng dashboard phân tích tự động với biểu đồ Loss, Metrics, Confusion Matrix, PR Curve và bảng so sánh Ablation Study.
                        """)
                else:
                    st.info("Không tìm thấy đủ dữ liệu để so sánh các thí nghiệm.")

            else:
                st.warning(f"Không có dữ liệu kết quả huấn luyện trong thư mục `{selected_run_dir}`.")


    # ----------------------------------------------------
    # TAB 4: CƠ SỞ LÝ THUYẾT & TÀI LIỆU BÁO CÁO
    # ----------------------------------------------------
    with tab_theory:
        st.markdown("""
        ### 📚 Cơ sở Lý thuyết & Kiến trúc Hệ thống
        
        ---
        
        #### PHẦN A — Trí Tuệ Nhân Tạo (Artificial Intelligence)

        ##### 1. Mô hình YOLOv8 (You Only Look Once - Version 8)
        - **Kiến trúc Backbone**: Sử dụng mạng tích chập sâu (CSPDarknet cải tiến) để trích xuất đặc trưng hình ảnh ở nhiều tỉ lệ khác nhau.
        - **Head không dùng Anchor (Anchor-Free)**: Dự đoán trực tiếp tâm và kích thước hộp giới hạn (Bounding Box) thay vì phụ thuộc vào các anchor boxes định sẵn, giúp tăng tốc độ xử lý và độ chính xác đối với vật thể nhiều kích cỡ.
        - **Hàm mất mát (Loss Function)**:
          - *CIoU / DFL (Distribution Focal Loss)*: Tối ưu vị trí hộp bounding box.
          - *BCE Loss (Binary Cross-Entropy)*: Phân loại đối tượng đa nhãn.

        ##### 2. Thuật toán Theo dõi Đối tượng (ByteTrack)
        - Khác với SORT truyền thống chỉ ghép các hộp có độ tin cậy cao, **ByteTrack** tận dụng cả những phát hiện có độ tin cậy thấp (low confidence detections) để duy trì dấu vết (track) ngay cả khi phương tiện bị che khuất một phần (occlusion).
        - Sử dụng bộ lọc **Kalman Filter** để ước lượng trạng thái vận tốc và dự đoán vị trí kế tiếp của từng xe.

        ##### 3. Thuật toán Đếm cắt vạch (Line Crossing Intersection)
        - Dựa trên tích có hướng vector (Cross Product) giữa đoạn di chuyển của tâm xe $P_{prev} \\to P_{curr}$ và đoạn thẳng vạch đếm ảo $A \\to B$.
        - Ngăn chặn hiện tượng đếm trùng lặp bằng cấu trúc dữ liệu tập hợp `set()` lưu trữ các `track_id` đã từng cắt qua vạch.

        ---

        #### PHẦN B — Xử lý Ảnh & Thị giác Máy tính (Image Processing & Computer Vision)

        ##### 4. Tiền xử lý ảnh (Image Preprocessing)
        - **Chuyển đổi không gian màu:** RGB → Grayscale (giảm chiều dữ liệu), RGB → HSV (tách biệt sắc độ và cường độ sáng để phân vùng theo màu).
        - **Lọc nhiễu Gaussian:** Sử dụng nhân tích chập (kernel) Gaussian 2D để làm mịn ảnh, loại bỏ nhiễu tần số cao trước khi phát hiện cạnh.
        - **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Cân bằng lược đồ xám cục bộ, tăng tương phản thích nghi theo từng vùng ảnh.

        ##### 5. Phát hiện cạnh (Edge Detection)
        - **Toán tử Sobel:** Tính xấp xỉ gradient bậc 1 theo hướng X (cạnh dọc) và Y (cạnh ngang) bằng tích chập với nhân 3×3. Magnitude = √(Gx² + Gy²).
        - **Thuật toán Canny (1986):** Pipeline đa bước — Gaussian Blur → Gradient (Sobel) → Non-Maximum Suppression → Double Thresholding → Hysteresis Tracking.
        - **Laplacian:** Đạo hàm bậc 2, phát hiện cạnh đẳng hướng nhưng nhạy với nhiễu.

        ##### 6. Phép toán hình thái học (Morphological Operations)
        - **Erosion (Co):** Thu nhỏ vùng trắng, loại nhiễu hạt nhỏ.
        - **Dilation (Giãn nở):** Mở rộng vùng trắng, lấp lỗ hổng.
        - **Opening (Mở = Erosion → Dilation):** Loại nhiễu muối tiêu mà không thay đổi kích thước đối tượng.
        - **Closing (Đóng = Dilation → Erosion):** Nối các vùng rời rạc thành vùng liền mạch.

        ##### 7. Trừ nền chuyển động (Background Subtraction)
        - **MOG2 (Mixture of Gaussians v2):** Mô hình hóa mỗi pixel bằng hỗn hợp K phân phối Gaussian, tự động cập nhật tham số để thích nghi với nền thay đổi.
        - **KNN:** Phân loại pixel foreground/background dựa trên K-láng giềng gần nhất trong lịch sử.

        ##### 8. Phát hiện Contour & Bounding Box
        - Tìm đường viền khép kín trên ảnh nhị phân bằng `cv2.findContours()`.
        - Lọc theo diện tích (area) và tỷ lệ khung hình (aspect ratio) để loại nhiễu và chỉ giữ vùng có hình dạng xe hợp lệ.
        - Tính Bounding Box bao quanh từng contour bằng `cv2.boundingRect()`.

        ##### 9. Biến đổi phối cảnh (Perspective Transform)
        - Sử dụng ma trận Homography (3×3) để chuyển đổi góc nhìn camera xiên thành Bird's Eye View (nhìn từ trên xuống).
        - Ứng dụng: đo mật độ giao thông, khoảng cách giữa các xe chính xác hơn.

        ##### 10. Xử lý Ảnh Nhiễu & Khôi phục Ảnh (Noise Models & Restoration)
        - **Mô hình Nhiễu Gaussian**: $I_{noisy} = I + \mathcal{N}(\mu, \sigma^2)$, mô phỏng nhiễu cảm biến nhiệt độ cao hoặc khuếch đại điện tử. Khử hiệu quả nhất bằng Non-Local Means (NLM) hoặc Gaussian Filter.
        - **Mô hình Nhiễu Muối-Tiêu (Impulse Noise)**: Thay thế ngẫu nhiên điểm ảnh bằng giá trị cực trị (0 hoặc 255). Bộ lọc Trung vị (Median Filter) là tối ưu tuyệt đối vì loại bỏ hoàn toàn các giá trị ngoại lai (outliers) mà không làm mờ biên cạnh.
        - **Mô hình Nhiễu Poisson (Shot Noise)**: Xuất hiện do bản chất lượng tử rời rạc của photon trong ảnh chụp tốc độ cao / thiếu sáng. Phục hồi tối ưu bằng NLM thích ứng.
        - **Mô hình Nhiễu Speckle (Multiplicative Noise)**: $I_{noisy} = I + I \cdot \mathcal{N}(0, \sigma^2)$, thường gặp trong radar, siêu âm, camera giao thông trong mưa/sương mù. Phục hồi bằng Bilateral Filter để giữ sắc nét vạch đường và thân xe.
        - **Định lượng chất lượng phục hồi**: Sử dụng PSNR (Peak Signal-to-Noise Ratio, dB) và MSE (Mean Squared Error). Mức tăng $\Delta\text{PSNR} > 0$ chứng minh bộ lọc cải thiện chất lượng ảnh so với ảnh nhiễu.
        """)


if __name__ == "__main__":
    main()

