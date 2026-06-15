from enum import Enum

from busterx.registry.registry import Registry

DATASET_REGISTRY = Registry("DATASET")


class DataSets(str, Enum):
    genbuster_200k = "genbuster_200k"
    genbuster_bench = "genbuster_bench"
    custom_benchmark = "custom_benchmark"
    genbuster_unified_video = "genbuster_unified_video"
    genbuster_unified_image = "genbuster_unified_image"
    genbuster_bench_plusplus_video = "genbuster_bench_plusplus_video"
    genbuster_bench_plusplus_image = "genbuster_bench_plusplus_image"
