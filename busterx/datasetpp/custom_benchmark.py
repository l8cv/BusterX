import json
import os
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from tabulate import tabulate

from busterx.datasetpp.base import BaseVideoDatasetPP
from busterx.registry import DATASET_REGISTRY, DataSets

# Subdirectory names (case-insensitive) that contain real (non-AI-generated) videos.
REAL_SUBCAT_NAMES = {"real"}


def _scan_subcats(dataset_dir: Path, cls: type[BaseVideoDatasetPP]) -> Dict[str, List[Path]]:
    """Scan ``dataset_dir`` for subdirectories that contain at least one valid video.

    Each such subdirectory is considered a subclass; its name is the subclass name
    and the videos found (recursively) inside it are the samples of that subclass.
    """
    subcats: Dict[str, List[Path]] = {}
    for entry in sorted(dataset_dir.iterdir()):
        if not entry.is_dir():
            continue
        # Skip hidden / cache dirs
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue
        videos = cls.list_media_files(entry)
        if videos:
            subcats[entry.name] = videos
    return subcats


@DATASET_REGISTRY.register(name=DataSets.custom_benchmark)
class CustomBenchmark(BaseVideoDatasetPP):
    dataset_name = DataSets.custom_benchmark

    def _collect_samples(self, dataset_dir: Path) -> List[Dict[str, Any]]:
        subcats = _scan_subcats(dataset_dir, type(self))
        if not subcats:
            logger.warning(f"No subclass directories with valid videos found in {dataset_dir}")

        dataset_infos: List[Dict[str, Any]] = []
        for subcat_name, videos in subcats.items():
            # By convention, subdirs named "real" (case-insensitive) hold real videos
            # (label A); everything else is treated as fake / AI-generated (label B).
            solution = "A" if subcat_name.lower() in REAL_SUBCAT_NAMES else "B"
            for v in videos:
                dataset_infos.append({"video_path": str(v), "solution": solution, "subcat": subcat_name})
            logger.info(f"Discovered {len(videos)} videos from subclass '{subcat_name}' (solution={solution}).")
        return dataset_infos

    def _build_sample(
        self, video_path: Path, info: Dict[str, Any], input_mode: str, cache_results: dict
    ) -> Dict[str, Any] | None:
        sample = super()._build_sample(video_path, info, input_mode, cache_results)
        if sample is None:
            return None
        if input_mode == "image" and "images" in sample:
            valid_images = [img for img in sample["images"] if os.path.exists(img)]
            if len(valid_images) != len(sample["images"]):
                logger.warning(
                    f"[BusterX] Dropped {len(sample['images']) - len(valid_images)} missing frames for {video_path}"
                )
                return None
        return sample

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_samples = len(data_list)
        correct_samples = 0

        by_subcat: Dict[str, List[Dict[str, Any]]] = {}
        for data in data_list:
            subcat = data.get("subcat", "unknown_subcat")
            if subcat not in by_subcat:
                by_subcat[subcat] = []
            by_subcat[subcat].append(data)

        subcat_metrics: Dict[str, Any] = {}

        # real / fake aggregates, derived from subclass name via REAL_SUBCAT_NAMES.
        real_total = real_correct = 0
        fake_total = fake_correct = 0

        for subcat, items in by_subcat.items():
            is_real = subcat.lower() in REAL_SUBCAT_NAMES
            subcat_correct = 0
            for data in items:
                prediction = data.get("response", "").strip()
                solution = data.get("solution", "").strip()

                if self.judge_answer(prediction, solution):
                    subcat_correct += 1
                    correct_samples += 1
                    if is_real:
                        real_correct += 1
                    else:
                        fake_correct += 1

            subcat_total = len(items)
            if is_real:
                real_total += subcat_total
            else:
                fake_total += subcat_total
            acc = round(subcat_correct * 100.0 / subcat_total, 2) if subcat_total > 0 else 0.0

            subcat_metrics[subcat] = {"total": subcat_total, "correct": subcat_correct, "accuracy": acc}

        overall_acc = round(correct_samples * 100.0 / total_samples, 2) if total_samples > 0 else 0.0
        real_acc = round(real_correct * 100.0 / real_total, 2) if real_total > 0 else 0.0
        fake_acc = round(fake_correct * 100.0 / fake_total, 2) if fake_total > 0 else 0.0

        all_metrics = {
            "overall_accuracy": overall_acc,
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "real_accuracy": real_acc,
            "real_total": real_total,
            "real_correct": real_correct,
            "fake_accuracy": fake_acc,
            "fake_total": fake_total,
            "fake_correct": fake_correct,
            "subcategories": subcat_metrics,
        }

        logger.info(f"Results: \n{json.dumps(all_metrics, indent=2)}")

        # One-line table: each subclass + aggregated Fake ACC + Overall ACC.
        # Put real subclasses first, then fake subclasses, both alphabetically.
        real_names = sorted(n for n in subcat_metrics if n.lower() in REAL_SUBCAT_NAMES)
        fake_names = sorted(n for n in subcat_metrics if n.lower() not in REAL_SUBCAT_NAMES)
        ordered_names = real_names + fake_names

        headers = [*ordered_names, "Fake ACC", "Overall ACC"]
        row = [subcat_metrics[name]["accuracy"] for name in ordered_names] + [fake_acc, overall_acc]

        table = tabulate([row], headers=headers, tablefmt="grid")
        logger.info(f"\n[BusterX] Evaluation Table:\n{table}")

        latex_row = " & ".join(str(x) for x in row) + r" \\"
        logger.info(f"\n[BusterX] LaTeX format row:\n& {latex_row}\n")

        return all_metrics
