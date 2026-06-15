import argparse
import json
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from busterx.registry import DATASET_REGISTRY


def route_eval(args: argparse.Namespace) -> None:
    dataset_name = args.dataset_pp
    dataset_cls = DATASET_REGISTRY.get(dataset_name)
    if not dataset_cls:
        logger.error(f"Dataset {dataset_name} not found in registry.")
        return

    dataset_instance = dataset_cls(args)

    result_file = Path(args.result_file)
    logger.info(f"Evaluating results from {result_file}")

    if not result_file.exists():
        logger.error(f"Result file not found: {result_file}")
        return

    with open(result_file, "r", encoding="utf-8") as f:
        if result_file.suffix == ".jsonl":
            total_lines = sum(1 for _ in f)
            f.seek(0)
            data_list = [
                json.loads(line) for line in tqdm(f, total=total_lines, desc="Loading eval results", unit="sample")
            ]
        else:
            data_list = json.load(f)

    metrics = dataset_instance.calc_metric(data_list)

    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metrics to {output_file}")


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    eval_parser = subparsers.add_parser("eval", help="Evaluate the accuracy of the generated results")
    eval_parser.add_argument(
        "--dataset_pp", type=str, required=True, help="Name of the dataset to evaluate (e.g. genbuster_bench)"
    )
    eval_parser.add_argument(
        "--result_file", type=str, required=True, help="JSON/JSONL file containing the predicted results"
    )
    eval_parser.add_argument("--output", type=str, default="", help="Optional output JSON file for metrics")
    eval_parser.set_defaults(func=route_eval)
