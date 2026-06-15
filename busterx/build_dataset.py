import argparse

from loguru import logger

from busterx.registry import DATASET_REGISTRY


def route_build_dataset(args: argparse.Namespace) -> None:
    dataset_name = args.dataset_pp
    dataset_cls = DATASET_REGISTRY.get(dataset_name)
    if not dataset_cls:
        logger.error(f"Dataset {dataset_name} not found in registry.")
        return
    dataset_instance = dataset_cls(args)
    dataset_instance.build_dataset()
    dataset_instance.save_dataset(args.output)


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    build_parser = subparsers.add_parser("build_dataset", help="Build a registered dataset in JSONL format")
    build_parser.add_argument(
        "--dataset_pp",
        type=str,
        required=True,
        help="Name of the dataset to build (e.g. genbuster_200k, genbuster_bench)",
    )
    build_parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing dataset files")
    build_parser.add_argument("--output", type=str, required=True, help="Output file path for built dataset (JSONL)")
    build_parser.add_argument(
        "--prompt", type=str, default="prompts/default_user_prompt.txt", help="Path to the user prompt template"
    )
    build_parser.add_argument(
        "--input_mode", type=str, choices=["video", "image"], default="video", help="Mode: video or image"
    )
    build_parser.add_argument("--sample_fps", type=float, default=1.0, help="Frames-per-second sampling rate")
    build_parser.add_argument(
        "--resize", type=int, nargs=2, default=None, help="Target (width, height) to resize frames to (e.g. 224 224)"
    )
    build_parser.add_argument("--cache_dir", type=str, default="", help="Root directory for the frame cache")
    build_parser.add_argument("--max_workers", type=int, default=8, help="Thread pool size for frame extraction")
    build_parser.add_argument("--decord_timeout", type=int, default=120, help="Timeout for decord video extraction")
    build_parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG save quality (1-100)")
    build_parser.set_defaults(func=route_build_dataset)
