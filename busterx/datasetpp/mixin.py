import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol

from loguru import logger
from tabulate import tabulate
from tqdm import tqdm


class _CollectSamplesProto(Protocol):
    """Static interface required by sample-collecting mixins."""

    path_key: str

    def list_media_files(self, d: Path) -> List[Path]: ...


class _MetricProto(Protocol):
    """Static interface required by metric mixins."""

    judge_answer: Callable[[str, str], bool]


class RealFakeFlatMixin:
    """Mixin: scan ``dataset_dir/{real,fake}/*`` (files placed directly, no subcat dirs)."""

    def _collect_samples(self: _CollectSamplesProto, dataset_dir: Path) -> List[Dict[str, Any]]:
        real_items = self.list_media_files(dataset_dir / "real")
        fake_items = self.list_media_files(dataset_dir / "fake")
        logger.info(f"Discovered {len(real_items)} real items.")
        logger.info(f"Discovered {len(fake_items)} fake items.")

        infos: List[Dict[str, Any]] = []
        for f in real_items:
            infos.append({self.path_key: str(f), "solution": "A"})
        for f in fake_items:
            infos.append({self.path_key: str(f), "solution": "B"})
        return infos


class RealFakeWithSubcatsMixin:
    """Mixin: scan ``dataset_dir/{real,fake}/<subcat>/*`` and tag each sample with its subcat."""

    def _collect_samples(self: _CollectSamplesProto, dataset_dir: Path) -> List[Dict[str, Any]]:
        def _scan(d: Path) -> Dict[str, List[Path]]:
            out: Dict[str, List[Path]] = {}
            if not d.exists():
                return out
            for sub in sorted(d.iterdir()):
                if not sub.is_dir():
                    continue
                files = self.list_media_files(sub)
                if files:
                    out[sub.name] = files
            return out

        real_by_subcat = _scan(dataset_dir / "real")
        fake_by_subcat = _scan(dataset_dir / "fake")

        infos: List[Dict[str, Any]] = []
        real_total = 0
        for subcat, files in real_by_subcat.items():
            real_total += len(files)
            logger.info(f"Discovered {len(files)} real items from {subcat}.")
            for f in files:
                infos.append({self.path_key: str(f), "solution": "A", "subcat": subcat})
        logger.info(f"Discovered total {real_total} real items.")

        fake_total = 0
        for subcat, files in fake_by_subcat.items():
            fake_total += len(files)
            logger.info(f"Discovered {len(files)} fake items from {subcat}.")
            for f in files:
                infos.append({self.path_key: str(f), "solution": "B", "subcat": subcat})
        logger.info(f"Discovered total {fake_total} fake items.")

        return infos


class RealFakeMetricMixin:
    """Mixin: compute real / fake / overall accuracy + ACC_g for flat real-vs-fake benchmarks."""

    def calc_metric(self: _MetricProto, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_samples = len(data_list)
        correct_samples = 0

        real_total = real_correct = 0
        fake_total = fake_correct = 0

        for data in tqdm(data_list, desc="Scoring predictions", unit="sample"):
            prediction = data.get("response", "").strip()
            solution = data.get("solution", "").strip()
            is_correct = self.judge_answer(prediction, solution)

            if solution == "A":
                real_total += 1
                if is_correct:
                    real_correct += 1
                    correct_samples += 1
            else:
                fake_total += 1
                if is_correct:
                    fake_correct += 1
                    correct_samples += 1

        overall_acc = round(correct_samples * 100.0 / total_samples, 2) if total_samples > 0 else 0.0
        real_acc = round(real_correct * 100.0 / real_total, 2) if real_total > 0 else 0.0
        fake_acc = round(fake_correct * 100.0 / fake_total, 2) if fake_total > 0 else 0.0
        acc_g = round((real_acc + fake_acc) / 2.0, 2)

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
            "ACC_g": acc_g,
        }

        logger.info(f"Results: \n{json.dumps(all_metrics, indent=2)}")

        headers = ["Real", "Fake", "Overall ACC", "ACC_g"]
        row = [real_acc, fake_acc, overall_acc, acc_g]

        table = tabulate([row], headers=headers, tablefmt="grid")
        logger.info(f"\n[BusterX] Evaluation Table:\n{table}")

        latex_row = " & ".join(str(x) for x in row) + r" \\"
        logger.info(f"\n[BusterX] LaTeX format row:\n& {latex_row}\n")

        return all_metrics
