import argparse

import busterx.datasetpp  # noqa: F401
from busterx import build_dataset, eval


def main() -> None:
    parser = argparse.ArgumentParser(description="busterx main entrypoint for dataset and evaluation")
    subparsers = parser.add_subparsers(dest="route", help="Available routes: build_dataset, eval")
    subparsers.required = True

    build_dataset.setup_parser(subparsers)
    eval.setup_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
