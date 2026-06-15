from __future__ import annotations

import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from loguru import logger
from PIL import Image
from tqdm import tqdm

from busterx.util import make_image_cache_param_key
from busterx.vl_utils.vision_process import fetch_video_frames

SUPPORTED_VIDEO_READERS = {"decord", "torchvision", "auto"}


def _extract_video_frames(
    video_path: str,
    sample_fps: float,
    resize: Tuple[int, int] | None,
    video_backend: str = "auto",
) -> Tuple[List[Image.Image], List[float]]:
    """Extract frames via vision_process, then optionally resize locally."""
    assert sample_fps > 0, "[BusterX] sample_fps must be > 0"

    if video_backend not in SUPPORTED_VIDEO_READERS:
        raise ValueError(
            f"[BusterX] Unsupported video reader backend: {video_backend!r}, must be one of {SUPPORTED_VIDEO_READERS}"
        )

    frames, metadata, _ = fetch_video_frames(
        {"video": video_path, "fps": sample_fps},
        video_backend=video_backend,
    )
    if not frames:
        return [], []

    fps = metadata.get("fps", 0.0)
    if fps <= 0:
        return [], []

    if resize is not None:
        frames = [frame.resize(resize, Image.BICUBIC) for frame in frames]

    frame_indices = metadata.get("frames_indices", [])
    timestamps = [round(float(frame_idx) / float(fps), 1) for frame_idx in frame_indices]
    return frames, timestamps


# ---------------------------------------------------------------------------
# Cache metadata
# ---------------------------------------------------------------------------

_META_FILENAME = "cache_meta.json"


def _read_cache_dir(frame_cache: str, video_path: str) -> Tuple[List[str], List[float]]:
    """Read cached frame JPEGs + timestamps from a cache directory."""
    meta_path = os.path.join(frame_cache, _META_FILENAME)
    if not os.path.exists(meta_path):
        return [], []

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [], []

    # Validate video_path (different videos can share the same stem)
    if meta.get("video_path") != os.path.abspath(video_path):
        return [], []

    timestamps: List[float] = meta.get("timestamps", [])
    if not timestamps:
        return [], []

    entries = sorted(e for e in os.listdir(frame_cache) if e.startswith("frame_") and e.endswith(".jpg"))
    paths = [os.path.join(frame_cache, e) for e in entries]

    n = min(len(paths), len(timestamps))
    return paths[:n], timestamps[:n]


# ---------------------------------------------------------------------------
# Single-video cache logic
# ---------------------------------------------------------------------------


def cache_frames(
    video_path: str,
    cache_dir: str,
    sample_fps: float = 1.0,
    resize: Tuple[int, int] | None = None,
    jpeg_quality: int = 95,
    decord_timeout: int = 120,
) -> Tuple[List[str], List[float]]:
    """Extract frames and persist as JPEGs under *cache_dir*.

    Cache layout::

        <cache_dir>/<video_stem>/<param_key>/cache_meta.json
        <cache_dir>/<video_stem>/<param_key>/frame_000001.jpg ...

    ``<param_key>`` encodes ``resize``, ``sample_fps`` and ``jpeg_quality``
    (e.g. ``raw_fps2.0_q95``) so different parameter combinations coexist.

    Returns:
        (cached_jpg_paths, timestamps)
    """

    if sample_fps <= 0:
        raise ValueError("[BusterX] sample_fps must be > 0")

    video_stem = Path(video_path).stem
    param_key = make_image_cache_param_key(sample_fps, resize, jpeg_quality)
    frame_cache = os.path.join(cache_dir, video_stem, param_key)

    # --- cache hit ---
    if os.path.isdir(frame_cache):
        paths, timestamps = _read_cache_dir(frame_cache, video_path)
        if paths:
            return paths, timestamps

    # --- cache miss or stale: rebuild ---
    if os.path.isdir(frame_cache):
        shutil.rmtree(frame_cache)
    os.makedirs(frame_cache, exist_ok=True)

    try:
        if decord_timeout > 0:
            # NOTE: avoid `with` statement — ThreadPoolExecutor.__exit__ calls
            # shutdown(wait=True) which blocks until the decord thread finishes,
            # defeating the purpose of the timeout.
            pool = ThreadPoolExecutor(max_workers=1)
            fut = pool.submit(_extract_video_frames, video_path, sample_fps, resize)
            try:
                frames, dec_timestamps = fut.result(timeout=decord_timeout)
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        else:
            frames, dec_timestamps = _extract_video_frames(
                video_path,
                sample_fps,
                resize,
            )

        if not frames:
            shutil.rmtree(frame_cache, ignore_errors=True)
            return [], []

        # Save PIL images as JPEGs
        cached_paths: List[str] = []
        for idx, img in enumerate(frames):
            fname = f"frame_{idx:06d}.jpg"
            fpath = os.path.join(frame_cache, fname)
            img.save(fpath, quality=jpeg_quality)
            cached_paths.append(fpath)

        # Write meta last (signals successful cache)
        with open(os.path.join(frame_cache, _META_FILENAME), "w") as f:
            json.dump({"video_path": os.path.abspath(video_path), "timestamps": dec_timestamps}, f)

        return cached_paths, dec_timestamps

    except Exception:
        shutil.rmtree(frame_cache, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Batch (multi-threaded) extraction with progress bar
# ---------------------------------------------------------------------------


def batch_extract_frames(
    video_paths: Sequence[str],
    cache_dir: str,
    sample_fps: float = 1.0,
    resize: Tuple[int, int] | None = None,
    jpeg_quality: int = 95,
    decord_timeout: int = 120,
    max_workers: int | None = None,
    desc: str = "[BusterX] Extracting frames",
) -> Dict[str, Tuple[List[str], List[float]]]:
    """Extract & cache frames for many videos in parallel.

    Args:
        video_paths: Videos to process (duplicates are deduplicated automatically).
        cache_dir: Root directory for the frame JPEG cache.
        sample_fps: Frames-per-second sampling rate.
        resize: Target (width, height) or ``None``.
        jpeg_quality: JPEG save quality (1-100).
        max_workers: Thread pool size.  Defaults to ``min(8, cpu_count)``.
        desc: Progress bar description text.

    Returns:
        Mapping from *video_path* → ``(frame_jpg_paths, timestamps)`` for every
        video that was successfully processed.  Failed videos are logged and omitted.
    """
    unique_videos = list(dict.fromkeys(video_paths))  # deduplicate, preserve order
    if not unique_videos:
        return {}

    if max_workers is None:
        max_workers = 8
        logger.info(f"[BusterX] use {max_workers} workers for batch frame extraction ")

    results: Dict[str, Tuple[List[str], List[float]]] = {}
    failed = 0

    def _do(vp: str) -> Tuple[str, List[str], List[float]]:
        paths, timestamps = cache_frames(vp, cache_dir, sample_fps, resize, jpeg_quality, decord_timeout)
        return vp, paths, timestamps

    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures = {pool.submit(_do, vp): vp for vp in unique_videos}
    try:
        with tqdm(total=len(futures), desc=desc, unit="video") as pbar:
            for fut in as_completed(futures):
                vp = futures[fut]
                try:
                    _, paths, timestamps = fut.result()
                    if paths:
                        results[vp] = (paths, timestamps)
                    else:
                        failed += 1
                except Exception as e:
                    logger.warning(f"[BusterX] Frame extraction failed for {vp}, {type(e).__name__} && {e}")
                    failed += 1
                pbar.update(1)
    except KeyboardInterrupt:
        logger.error("\n[BusterX] Received Ctrl+C (KeyboardInterrupt); cancelling queued tasks...")
        for fut in futures:
            fut.cancel()
        if sys.version_info >= (3, 9):
            pool.shutdown(wait=False, cancel_futures=True)
        else:
            pool.shutdown(wait=False)
        logger.error("[BusterX] Force exiting process.")
        os._exit(1)
    finally:
        pool.shutdown(wait=True)

    logger.info(f"[BusterX] Frame cache done: {len(results)} ok, {failed} failed, workers={max_workers}")
    return results
