"""Shared video upload and resource-lifecycle helpers for the Streamlit app."""

from contextlib import contextmanager
from pathlib import Path
import tempfile


MAX_VIDEO_UPLOAD_BYTES = 500 * 1024 * 1024


def save_uploaded_video(uploaded_file) -> str:
    """Persist an upload for OpenCV after applying a basic size limit."""
    if uploaded_file.size > MAX_VIDEO_UPLOAD_BYTES:
        limit_mb = MAX_VIDEO_UPLOAD_BYTES // (1024 * 1024)
        raise ValueError(f"Video quá lớn. Giới hạn hiện tại là {limit_mb} MB.")

    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.getbuffer())
        return tfile.name


def cleanup_temp_video(video_path: str | None) -> None:
    """Delete only temporary files created by this app."""
    if not video_path:
        return
    path = Path(video_path).resolve()
    temp_dir = Path(tempfile.gettempdir()).resolve()
    if path.parent == temp_dir and path.is_file():
        path.unlink()


@contextmanager
def open_video(video_path: str):
    """Open a video and always release its OpenCV handle."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise OSError("Không thể mở video. Hãy kiểm tra codec hoặc tải lại file khác.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        yield cap, total_frames
    finally:
        cap.release()


def get_video_metadata(video_path: str) -> tuple[int, int, int, float]:
    """Return width, height, frame count and source FPS."""
    import cv2

    with open_video(video_path) as (cap, total_frames):
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    return width, height, total_frames, fps


def iter_video_frames(video_path: str, cleanup: bool = False):
    """Yield frames while guaranteeing capture release and optional temp cleanup."""
    import cv2

    try:
        with open_video(video_path) as (cap, total_frames):
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
                yield frame_idx, frame, total_frames, (frame_idx - 1) / fps
    finally:
        if cleanup:
            cleanup_temp_video(video_path)
