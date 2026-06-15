import json
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from tabulate import tabulate

from busterx.datasetpp.base import BaseVideoDatasetPP
from busterx.registry import DATASET_REGISTRY, DataSets


@DATASET_REGISTRY.register(name=DataSets.genbuster_bench)
class GenBusterBench(BaseVideoDatasetPP):
    dataset_name = DataSets.genbuster_bench

    def _collect_samples(self, dataset_dir: Path) -> List[Dict[str, Any]]:
        infos: List[Dict[str, Any]] = []
        tracks = ["id", "ood", "wild"]
        for track in tracks:
            track_dir = dataset_dir / track
            if not track_dir.exists():
                continue

            real_dir = track_dir / "real"
            fake_dir = track_dir / "fake"

            if real_dir.exists():
                real_videos = self.list_media_files(real_dir)
                for v in real_videos:
                    infos.append({"video_path": str(v), "solution": "A", "subcat": "real", "track": track})
                logger.info(f"[{track}] Discovered {len(real_videos)} real videos.")

            if fake_dir.exists():
                fake_videos_total = 0
                for subcat_dir in fake_dir.iterdir():
                    if subcat_dir.is_dir():
                        subcat_name = subcat_dir.name
                        videos = self.list_media_files(subcat_dir)
                        for v in videos:
                            infos.append({"video_path": str(v), "solution": "B", "subcat": subcat_name, "track": track})
                        fake_videos_total += len(videos)
                        logger.info(f"[{track}] Discovered {len(videos)} fake videos from {subcat_name}.")
                logger.info(f"[{track}] Discovered total {fake_videos_total} fake videos.")

        return infos

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_samples = len(data_list)
        correct_samples = 0

        by_track_subcat: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for data in data_list:
            track = data.get("track", "unknown_track")
            subcat = data.get("subcat", "unknown_subcat")
            if track not in by_track_subcat:
                by_track_subcat[track] = {}
            by_track_subcat[track].setdefault(subcat, []).append(data)

        track_metrics: Dict[str, Any] = {}
        id_real_acc = None

        for track, subcats in by_track_subcat.items():
            track_correct = 0
            track_total = 0
            subcat_metrics: Dict[str, Any] = {}

            for subcat, items in subcats.items():
                subcat_correct = 0
                for data in items:
                    prediction = data.get("response", "").strip()
                    solution = data.get("solution", "").strip()

                    if self.judge_answer(prediction, solution):
                        subcat_correct += 1
                        track_correct += 1
                        correct_samples += 1

                subcat_total = len(items)
                track_total += subcat_total
                acc = round(subcat_correct * 100.0 / subcat_total, 2) if subcat_total > 0 else 0.0

                subcat_metrics[subcat] = {"total": subcat_total, "correct": subcat_correct, "accuracy": acc}

                if track == "id" and subcat == "real":
                    id_real_acc = acc

            track_acc = round(track_correct * 100.0 / track_total, 2) if track_total > 0 else 0.0
            track_metrics[track] = {
                "total": track_total,
                "correct": track_correct,
                "accuracy": track_acc,
                "subcategories": subcat_metrics,
            }

        if id_real_acc is not None:
            for track in ["id", "ood", "wild"]:
                if track in track_metrics:
                    fake_correct = 0
                    fake_total = 0
                    for subcat, metrics in track_metrics[track]["subcategories"].items():
                        if subcat != "real":
                            fake_correct += metrics["correct"]
                            fake_total += metrics["total"]

                    track_fake_acc = round(fake_correct * 100.0 / fake_total, 2) if fake_total > 0 else 0.0
                    track_metrics[track]["fake_accuracy"] = track_fake_acc
                    if track in ["ood", "wild"]:
                        acc_g = round((track_fake_acc + id_real_acc) / 2.0, 2)
                        track_metrics[track]["ACC_g"] = acc_g

        overall_acc = round(correct_samples * 100.0 / total_samples, 2) if total_samples > 0 else 0.0

        all_metrics = {
            "overall_accuracy": overall_acc,
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "tracks": track_metrics,
        }

        logger.info(f"Results: \n{json.dumps(all_metrics, indent=2)}")

        headers = [
            "ID Real",
            "ID Fake",
            "ID ACC",
            "Sora",
            "Pika",
            "Gen3",
            "Luma",
            "WanX",
            "Kling",
            "Jimeng",
            "Vidu",
            "Fake",
            "ACC_g",
            "Wild Fake",
            "Wild ACC_g",
        ]

        id_real = track_metrics.get("id", {}).get("subcategories", {}).get("real", {}).get("accuracy", "-")
        id_fake = track_metrics.get("id", {}).get("fake_accuracy", "-")
        id_acc = track_metrics.get("id", {}).get("accuracy", "-")

        ood_sora = track_metrics.get("ood", {}).get("subcategories", {}).get("sora", {}).get("accuracy", "-")
        ood_pika = track_metrics.get("ood", {}).get("subcategories", {}).get("pika", {}).get("accuracy", "-")
        ood_gen3 = track_metrics.get("ood", {}).get("subcategories", {}).get("gen3", {}).get("accuracy", "-")
        ood_luma = track_metrics.get("ood", {}).get("subcategories", {}).get("luma", {}).get("accuracy", "-")
        ood_wanx = track_metrics.get("ood", {}).get("subcategories", {}).get("wanx", {}).get("accuracy", "-")
        ood_kling = track_metrics.get("ood", {}).get("subcategories", {}).get("kling", {}).get("accuracy", "-")
        ood_jimeng = track_metrics.get("ood", {}).get("subcategories", {}).get("jimeng", {}).get("accuracy", "-")
        ood_vidu = track_metrics.get("ood", {}).get("subcategories", {}).get("vidu", {}).get("accuracy", "-")
        ood_fake = track_metrics.get("ood", {}).get("fake_accuracy", "-")
        ood_acc_g = track_metrics.get("ood", {}).get("ACC_g", "-")

        wild_fake = track_metrics.get("wild", {}).get("fake_accuracy", "-")
        wild_acc_g = track_metrics.get("wild", {}).get("ACC_g", "-")

        row = [
            id_real,
            id_fake,
            id_acc,
            ood_sora,
            ood_pika,
            ood_gen3,
            ood_luma,
            ood_wanx,
            ood_kling,
            ood_jimeng,
            ood_vidu,
            ood_fake,
            ood_acc_g,
            wild_fake,
            wild_acc_g,
        ]

        table = tabulate([row], headers=headers, tablefmt="grid")
        logger.info(f"\n[BusterX] Evaluation Table:\n{table}")

        latex_row = " & ".join(str(x) for x in row) + r" & - \\"
        logger.info(f"\n[BusterX] LaTeX format row:\n& {latex_row}\n")

        return all_metrics
