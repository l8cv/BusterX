import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Tuple

from loguru import logger
from math_verify import parse, verify

from busterx.util import IMAGE_EXTS, VIDEO_EXTS, resolve_prompt
from busterx.vl_utils.extract_frames import batch_extract_frames


class BaseDatasetPP:
    """Base class for BusterX dataset plugins.

    Subclasses must implement:
    - build_dataset: populate self.dataset with List[Dict[str, Any]]
    - calc_metric: compute evaluation metrics from inference results
    """

    user_prompt_default: ClassVar[str] = "Is this video real or fake?"
    dataset_name: ClassVar[str] = ""
    exts: ClassVar[set[str]] = set()

    @classmethod
    def is_media_file(cls, p: Path) -> bool:
        """Whether ``p`` is a regular file with a suffix in ``cls.exts``."""
        return p.is_file() and p.suffix.lower() in cls.exts

    @classmethod
    def list_media_files(cls, d: Path) -> List[Path]:
        """Recursively list media files under ``d`` filtered by ``cls.exts``."""
        if not d.exists():
            return []
        return sorted([p for p in d.rglob("*") if cls.is_media_file(p)])

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dataset: List[Dict[str, Any]] = []
        self._user_prompt_logged = False

    def get_dataset_dir(self) -> Path:
        """Resolve dataset_dir from args."""
        dataset_dir = getattr(self.args, "dataset_dir", "")
        if not dataset_dir:
            raise ValueError("[BusterX] dataset_dir not specified. Set --dataset_dir")
        path = Path(dataset_dir)
        if not path.exists():
            raise FileNotFoundError(f"[BusterX] dataset_dir not found: {path}")
        return path

    def get_user_prompt_template(self, default: str | Path | None = None) -> str:
        """Resolve user prompt template: arg > default > class default."""
        prompt: str | Path = getattr(self.args, "prompt", None) or default or self.user_prompt_default
        return resolve_prompt(str(prompt))

    def build_user_prompt(
        self,
        default: str | Path | None = None,
        prompt_replacements: Dict[str, str] | None = None,
        **prompt_kwargs: Any,
    ) -> str:
        """Build the final user prompt from the resolved template and format variables."""
        prompt_template = self.get_user_prompt_template(default)
        if prompt_kwargs:
            prompt = prompt_template.format(**prompt_kwargs)
        else:
            prompt = prompt_template

        if prompt_replacements:
            for old, new in prompt_replacements.items():
                prompt = prompt.replace(old, new)
        if not self._user_prompt_logged:
            logger.info(f"[BusterX] user prompt:\n{prompt}")
            self._user_prompt_logged = True
        return prompt

    def get_input_mode(self) -> str:
        mode = getattr(self.args, "input_mode", "video")
        if mode not in ("video", "image"):
            raise ValueError(f"[BusterX] Invalid input mode: {mode!r}, must be 'video' or 'image'")
        return mode

    def get_sample_fps(self) -> float:
        return getattr(self.args, "sample_fps", 2.0)

    def get_resize(self) -> Tuple[int, int] | None:
        resize_arg = getattr(self.args, "resize", None)
        if resize_arg and len(resize_arg) == 2:
            return tuple(resize_arg)
        return None

    def get_frame_cache_dir(self, dataset_name: str | None = None) -> str:
        """Arg → Return ``<CACHE_DIR>/<dataset_name>``."""
        cache_dir = getattr(self.args, "cache_dir", "")
        print(f"[BusterX] cache_dir arg: {cache_dir}")
        if not dataset_name and not self.dataset_name:
            raise ValueError("[BusterX] dataset_name not set in subclass, and dataset_name arg not provided.")

        if cache_dir:
            return os.path.join(cache_dir, dataset_name or self.dataset_name, "frames")
        else:
            return os.path.join(".busterx_cache", dataset_name or self.dataset_name, "frames")

    def build_dataset(self) -> List[Dict[str, Any]]:
        """Build dataset and store in self.dataset. Returns the list."""
        raise NotImplementedError

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute evaluation metrics from inference result data list."""
        logger.warning(f"{self.__class__.__name__} has no calc_metric implementation.")
        return {}

    def judge_answer(self, answer: str, solution: str) -> bool:
        """Default ``\\boxed{...}`` judging via ``math_verify``."""
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
        parsed_answer = parse(answer)
        return verify(parsed_answer, solution)

    def save_dataset(self, path: str) -> None:
        """Write self.dataset to a jsonl file."""
        with open(path, "w") as f:
            for item in self.dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


class BaseVideoDatasetPP(BaseDatasetPP):
    """Base class for video-based BusterX dataset plugins.

    Provides shared ``_build_sample`` and ``build_dataset`` plumbing:
    - Subclasses implement ``_collect_samples`` returning a list of dicts with at
      least ``video_path`` and ``solution``; ``subcat`` is optional and only
      written into the sample when provided (and non-empty).
    - ``build_dataset`` handles ``video``/``image`` input mode, frame extraction
      cache and prompt rendering uniformly.
    """

    exts: ClassVar[set[str]] = VIDEO_EXTS
    path_key: ClassVar[str] = "video_path"

    def _make_sample_extra(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract extra fields (solution, subcat, ...) attached to a sample."""
        extra: Dict[str, Any] = {"solution": info["solution"]}
        subcat = info.get("subcat")
        if subcat:
            extra["subcat"] = subcat
        for key, value in info.items():
            if key in (self.path_key, "solution", "subcat"):
                continue
            extra[key] = value
        return extra

    def _build_sample(
        self, video_path: Path, info: Dict[str, Any], input_mode: str, cache_results: dict
    ) -> Dict[str, Any] | None:
        extra = self._make_sample_extra(info)
        if input_mode == "video":
            prompt = self.build_user_prompt(prompt_replacements=None)
            return {
                "messages": [{"content": prompt, "role": "user"}],
                "videos": [str(video_path)],
                **extra,
            }

        cached = cache_results.get(str(video_path))
        if not cached:
            return None
        frame_paths, frame_ts = cached
        image_token_content = "".join(f"<{ts:.1f} seconds><image>" for ts in frame_ts)
        prompt = self.build_user_prompt(prompt_replacements={"<video>": image_token_content})
        return {
            "messages": [{"content": prompt, "role": "user"}],
            "images": frame_paths,
            **extra,
        }

    def _collect_samples(self, dataset_dir: Path) -> List[Dict[str, Any]]:
        """Return per-sample info dicts: ``video_path``, ``solution``, optional ``subcat``."""
        raise NotImplementedError

    def build_dataset(self) -> List[Dict[str, Any]]:
        dataset_dir = self.get_dataset_dir()
        output_file = Path(self.args.output)
        input_mode = self.get_input_mode()
        sample_fps = self.get_sample_fps()
        resize = self.get_resize()
        frame_cache_dir = self.get_frame_cache_dir()

        logger.info(f"Frame cache directory: {frame_cache_dir}")
        logger.info(f"Building {self.dataset_name} from {dataset_dir} with mode={input_mode}")

        dataset_infos = self._collect_samples(dataset_dir)
        all_video_paths = [info[self.path_key] for info in dataset_infos]

        cache_results: Dict[str, Any] = {}
        if input_mode == "image":
            os.makedirs(frame_cache_dir, exist_ok=True)
            unique_videos = list(set(all_video_paths))
            cache_results = batch_extract_frames(
                video_paths=unique_videos,
                cache_dir=str(frame_cache_dir),
                sample_fps=sample_fps,
                resize=resize,
                max_workers=getattr(self.args, "max_workers", 8),
                decord_timeout=getattr(self.args, "decord_timeout", 120),
                jpeg_quality=getattr(self.args, "jpeg_quality", 95),
            )

        for info in dataset_infos:
            sample = self._build_sample(Path(info[self.path_key]), info, input_mode, cache_results)
            if sample:
                self.dataset.append(sample)

        logger.info(f"Total: {len(self.dataset)} samples. Saving to {output_file}")
        logger.info(f"{self.dataset_name} dataset built successfully.")
        return self.dataset


class BaseImageDatasetPP(BaseDatasetPP):
    """Base class for image-based BusterX dataset plugins.

    Subclasses implement ``_collect_samples`` returning a list of dicts with at
    least ``image_path`` and ``solution``; ``subcat`` is optional and only
    written into the sample when non-empty.
    """

    exts: ClassVar[set[str]] = IMAGE_EXTS
    path_key: ClassVar[str] = "image_path"

    def _make_sample_extra(self, info: Dict[str, Any]) -> Dict[str, Any]:
        extra: Dict[str, Any] = {"solution": info["solution"]}
        subcat = info.get("subcat")
        if subcat:
            extra["subcat"] = subcat
        for key, value in info.items():
            if key in (self.path_key, "solution", "subcat"):
                continue
            extra[key] = value
        return extra

    def _build_sample(self, image_path: Path, info: Dict[str, Any]) -> Dict[str, Any]:
        extra = self._make_sample_extra(info)
        prompt = self.build_user_prompt(prompt_replacements={"<video>": "<image>"})
        return {
            "messages": [{"content": prompt, "role": "user"}],
            "images": [str(image_path)],
            **extra,
        }

    def _collect_samples(self, dataset_dir: Path) -> List[Dict[str, Any]]:
        """Return per-sample info dicts: ``image_path``, ``solution``, optional ``subcat``."""
        raise NotImplementedError

    def build_dataset(self) -> List[Dict[str, Any]]:
        dataset_dir = self.get_dataset_dir()
        output_file = Path(self.args.output)

        logger.info(f"Building {self.dataset_name} from {dataset_dir}")

        for info in self._collect_samples(dataset_dir):
            sample = self._build_sample(Path(info[self.path_key]), info)
            if sample:
                self.dataset.append(sample)

        logger.info(f"Total: {len(self.dataset)} samples. Saving to {output_file}")
        logger.info(f"{self.dataset_name} dataset built successfully.")
        return self.dataset
