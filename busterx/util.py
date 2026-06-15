import os
import re
from typing import Tuple

# Common video file extensions considered as valid videos.
VIDEO_EXTS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".ts",
    ".3gp",
    ".vp9",
    ".vp8",
    ".av1",
}

# Common image file extensions considered as valid images.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class AnswerParsing:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_answer_block(self) -> "AnswerParsing":
        answer_matches = re.findall(r"<answer>(.*?)</answer>", self.text, flags=re.IGNORECASE | re.DOTALL)
        if answer_matches:
            self.text = answer_matches[-1].strip()
        else:
            self.text = self.text.strip()
        return self

    def strip_think_blocks(self) -> "AnswerParsing":
        cleaned = re.sub(r"<think>.*?</think>", "", self.text, flags=re.DOTALL)
        self.text = cleaned
        return self


def resolve_prompt(prompt: str) -> str:
    """Return *prompt* itself, or its file contents if *prompt* points to an existing file."""
    if not prompt:
        return prompt
    if os.path.isfile(prompt):
        with open(prompt, "r", encoding="utf-8") as f:
            return f.read()
    return prompt


def make_image_cache_param_key(sample_fps: float, resize: Tuple[int, int] | None, jpeg_quality: int) -> str:
    """Build a cache key encoding all cache-varying params, e.g. ``raw_fps2.0_q95``."""
    r = f"{resize[0]}x{resize[1]}" if resize else "raw"
    return f"{r}_fps{sample_fps}_q{jpeg_quality}"
