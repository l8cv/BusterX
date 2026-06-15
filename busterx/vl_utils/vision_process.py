import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import torchvision
from packaging import version
from PIL import Image
from torchvision import io


def _get_frame_indices(
    total_frames: int,
    video_fps: float,
    sample_fps: float,
    video_start: float | None = None,
    video_end: float | None = None,
) -> Tuple[List[int], float]:
    if total_frames <= 0:
        return [], video_fps
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")

    max_duration = total_frames / video_fps
    start_sec = max(0.0, min(video_start if video_start is not None else 0.0, max_duration))
    end_sec = max(start_sec, min(video_end if video_end is not None else max_duration, max_duration))

    target_ts: Any = np.arange(start_sec, end_sec, 1.0 / sample_fps)
    if target_ts.size == 0:
        target_ts = np.array([start_sec], dtype=float)

    frame_indices = np.round(target_ts * video_fps).astype(np.int64)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices.tolist(), video_fps


def _read_video_torchvision(ele: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], float]:
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn(
                "torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.", stacklevel=2
            )
        if "file://" in video_path:
            video_path = video_path[7:]

    video, _, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), float(info["video_fps"])
    frame_indices, _ = _get_frame_indices(total_frames, video_fps, float(ele["fps"]))

    if frame_indices:
        video = video[frame_indices]
    else:
        video = video[:0]

    video = video.permute(0, 2, 3, 1).cpu().numpy()

    video_metadata = {
        "fps": video_fps,
        "frames_indices": frame_indices,
        "total_num_frames": total_frames,
        "video_backend": "torchvision",
    }
    return video, video_metadata, float(ele["fps"])


def _read_video_decord(ele: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], float]:
    import decord

    video_path = ele["video"]
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    video_fps = float(vr.get_avg_fps())
    frame_indices, _ = _get_frame_indices(total_frames, video_fps, float(ele["fps"]))

    if frame_indices:
        video = vr.get_batch(frame_indices).asnumpy()
    else:
        video = np.empty((0, 0, 0, 3), dtype=np.uint8)

    video_metadata = {
        "fps": video_fps,
        "frames_indices": frame_indices,
        "total_num_frames": total_frames,
        "video_backend": "decord",
    }
    return video, video_metadata, float(ele["fps"])


def fetch_video_numpy(
    ele: Dict[str, Any],
    video_backend: str = "decord",
) -> Tuple[np.ndarray, Dict[str, Any], float]:
    """Fetch video frames as a numpy array, along with video metadata."""
    if video_backend == "decord":
        video, video_metadata, fps = _read_video_decord(ele)
    elif video_backend == "torchvision":
        video, video_metadata, fps = _read_video_torchvision(ele)
    elif video_backend == "auto":
        try:
            video, video_metadata, fps = _read_video_decord(ele)
        except Exception:
            warnings.warn(f"Decord failed to read video {ele['video']}, falling back to torchvision.", stacklevel=2)
            video, video_metadata, fps = _read_video_torchvision(ele)
    else:
        raise ValueError(f"Unsupported video backend: {video_backend!r}")

    return video, video_metadata, fps


def fetch_video_frames(
    ele: Dict[str, Any],
    video_backend: str = "decord",
) -> Tuple[List[Image.Image], Dict[str, Any], float]:
    """Fetch video frames as PIL Images, along with video metadata."""
    video, video_metadata, fps = fetch_video_numpy(ele, video_backend=video_backend)
    image_list = [Image.fromarray(frame) for frame in video]
    return image_list, video_metadata, fps
