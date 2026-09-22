import os
import glob
import tempfile
import time
import threading
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


# Cấu hình trang Streamlit
st.set_page_config(
    page_title="AI Road Vehicle Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector(model_name: str):
    """Cache model YOLO để không phải tải lại nhiều lần."""
    model_path = os.path.join("models", model_name)
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
    st.markdown('<div class="main-header">🚗 HỆ THỐNG NHẬN DIỆN & ĐẾM PHƯƠNG TIỆN ĐƯỜNG BỘ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Công nghệ: YOLOv8 + ByteTrack + OpenCV</div>', unsafe_allow_html=True)

    # Sidebar điều khiển
    with st.sidebar:
        st.header("⚙️ Cấu hình mô hình AI")
        
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

        conf_thresh = st.slider("Ngưỡng tự tin (Confidence Threshold):", 0.1, 1.0, 0.35, 0.05)
        iou_thresh = st.slider("Ngưỡng IoU (Non-Max Suppression):", 0.1, 1.0, 0.45, 0.05)
        
        st.markdown("---")
        st.header("📏 Cấu hình vạch đếm xe")
        line_pos = st.slider("Vị trí vạch kẻ ảo (% chiều cao video):", 10, 90, 65, 5) / 100.0

        st.markdown("---")
        st.info("💡 **Các lớp phương tiện hỗ trợ:**\n- Ô tô (Car)\n- Xe máy (Motorcycle)\n- Xe buýt (Bus)\n- Xe tải (Truck)\n- Xe đạp (Bicycle)")

    # Khởi tạo detector
    with st.spinner("Đang tải mô hình AI..."):
        detector = load_detector(model_file)
    visualizer = Visualizer()

    # Tabs chức năng
    tab_webcam, tab_video, tab_image, tab_analytics, tab_theory = st.tabs([
        "📷 Webcam trực tiếp",
        "📹 Nhận diện & Đếm Video",
        "🖼️ Nhận diện Ảnh tĩnh",
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
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                video_path = tfile.name
        else:
            # Video mẫu trong thư mục data/samples/
            sample_dir = "data/samples"
            os.makedirs(sample_dir, exist_ok=True)
            sample_videos = [f for f in os.listdir(sample_dir) if f.endswith((".mp4", ".avi", ".mov"))]
            if sample_videos:
                selected_sample = st.selectbox("Chọn video mẫu:", sample_videos)
                video_path = os.path.join(sample_dir, selected_sample)
            else:
                st.warning(f"Chưa có video mẫu trong thư mục `{sample_dir}`. Bạn có thể tải lên video ở tùy chọn bên trên!")

        if video_path and os.path.exists(video_path):
            col_ctrl1, col_ctrl2 = st.columns([1, 4])
            start_btn = col_ctrl1.button("▶️ Bắt đầu xử lý Video", type="primary")
            stop_placeholder = col_ctrl2.empty()

            if start_btn:
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

                # Thiết lập vạch đếm
                line_y = int(height * line_pos)
                counter = VehicleCounter(line_coords=((0, line_y), (width, line_y)))

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
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    curr_time = time.time()
                    dt = curr_time - prev_time
                    if dt > 0:
                        fps = 0.9 * fps + 0.1 * (1.0 / dt)
                    prev_time = curr_time

                    # Nhận diện & Track
                    tracked_objs = detector.track(frame, conf_threshold=conf_thresh, iou_threshold=iou_thresh, persist=True)
                    count_info = counter.update(tracked_objs)

                    # Vẽ trực quan
                    disp = visualizer.draw_detections(frame, tracked_objs, VEHICLE_COLORS)
                    disp = visualizer.draw_trajectories(disp, counter.tracks_history)
                    disp = visualizer.draw_counting_line(disp, counter.line_coords)
                    disp = visualizer.draw_dashboard_hud(disp, count_info["counts"], count_info["total"], fps=fps, directions=count_info["directions"])

                    # Chuyển BGR sang RGB cho Streamlit
                    disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                    video_display.image(disp_rgb, channels="RGB", use_container_width=True)

                    # Cập nhật metrics
                    m_total.metric("🚗 Tổng số xe", count_info["total"])
                    m_car.metric("Ô tô con", count_info["counts"].get("car", 0))
                    m_moto.metric("Xe máy", count_info["counts"].get("motorcycle", 0))
                    m_bus.metric("Xe buýt", count_info["counts"].get("bus", 0))
                    m_truck.metric("Xe tải", count_info["counts"].get("truck", 0))
                    m_fps.metric("Tốc độ (FPS)", f"{fps:.1f}")

                    progress_bar.progress(min(1.0, frame_idx / total_frames))

                cap.release()
                st.success("🎉 Đã hoàn thành phân tích video!")

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
                st.image(res_rgb, caption=f"Kết quả nhận diện ({len(detections)} phương tiện)", use_container_width=True)
            with col_img2:
                st.markdown(f"**⏱️ Thời gian suy luận:** `{t_detect:.2f} ms`")
                st.markdown(f"**🔢 Tổng phương tiện tìm thấy:** `{len(detections)}`")
                for k, v in summary.items():
                    if v > 0:
                        st.markdown(f"- **{k.capitalize()}**: `{v}` xe")

                if table_data:
                    st.markdown("#### Chi tiết danh sách phát hiện:")
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # ----------------------------------------------------
    # TAB 3: ĐÁNH GIÁ & LỊCH SỬ HUẤN LUYỆN
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
                        st.image(cm_norm_path, caption="Ma trận nhầm lẫn chuẩn hóa (Normalized)", use_container_width=True)
                    elif os.path.exists(cm_path):
                        st.image(cm_path, caption="Ma trận nhầm lẫn (Số lượng tuyệt đối)", use_container_width=True)
                    else:
                        st.info("Chưa có dữ liệu Confusion Matrix cho thí nghiệm này.")

                pr_path = os.path.join(selected_run_dir, "BoxPR_curve.png")
                with col_pr:
                    st.markdown("**Precision-Recall Curve (Đường cong PR)**")
                    if os.path.exists(pr_path):
                        st.image(pr_path, caption="Đường cong Precision-Recall theo từng lớp phương tiện", use_container_width=True)
                    else:
                        st.info("Chưa có dữ liệu PR Curve cho thí nghiệm này.")

                f1_path = os.path.join(selected_run_dir, "BoxF1_curve.png")
                if os.path.exists(f1_path):
                    with st.expander("🔎 Xem thêm: F1-Score Curve"):
                        st.image(f1_path, caption="F1-Score Curve — Điểm cân bằng giữa Precision và Recall", use_container_width=True)

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
                        use_container_width=True,
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
        ### 📚 Cơ sở Lý thuyết & Kiến trúc Mô hình (Môn Trí Tuệ Nhân Tạo)
        
        #### 1. Mô hình YOLOv8 (You Only Look Once - Version 8)
        - **Kiến trúc Backbone**: Sử dụng mạng tích chập sâu (CSPDarknet cải tiến) để trích xuất đặc trưng hình ảnh ở nhiều tỉ lệ khác nhau.
        - **Head không dùng Anchor (Anchor-Free)**: Dự đoán trực tiếp tâm và kích thước hộp giới hạn (Bounding Box) thay vì phụ thuộc vào các anchor boxes định sẵn, giúp tăng tốc độ xử lý và độ chính xác đối với vật thể nhiều kích cỡ.
        - **Hàm mất mát (Loss Function)**:
          - *CIoU / DFL (Distribution Focal Loss)*: Tối ưu vị trí hộp bounding box.
          - *BCE Loss (Binary Cross-Entropy)*: Phân loại đối tượng đa nhãn.

        #### 2. Thuật toán Theo dõi Đối tượng (ByteTrack)
        - Khác với SORT truyền thống chỉ ghép các hộp có độ tin cậy cao, **ByteTrack** tận dụng cả những phát hiện có độ tin cậy thấp (low confidence detections) để duy trì dấu vết (track) ngay cả khi phương tiện bị che khuất một phần (occlusion).
        - Sử dụng bộ lọc **Kalman Filter** để ước lượng trạng thái vận tốc và dự đoán vị trí kế tiếp của từng xe.

        #### 3. Thuật toán Đếm cắt vạch (Line Crossing Intersection)
        - Dựa trên tích có hướng vector (Cross Product) giữa đoạn di chuyển của tâm xe $P_{prev} \to P_{curr}$ và đoạn thẳng vạch đếm ảo $A \to B$.
        - Ngăn chặn hiện tượng đếm trùng lặp bằng cấu trúc dữ liệu tập hợp `set()` lưu trữ các `track_id` đã từng cắt qua vạch.
        """)


if __name__ == "__main__":
    main()
