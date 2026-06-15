from busterx.datasetpp.base import BaseImageDatasetPP, BaseVideoDatasetPP
from busterx.datasetpp.mixin import RealFakeFlatMixin, RealFakeMetricMixin
from busterx.registry import DATASET_REGISTRY, DataSets


@DATASET_REGISTRY.register(name=DataSets.genbuster_bench_plusplus_video)
class GenBusterBenchPlusplusVideo(RealFakeMetricMixin, RealFakeFlatMixin, BaseVideoDatasetPP):
    dataset_name = DataSets.genbuster_bench_plusplus_video


@DATASET_REGISTRY.register(name=DataSets.genbuster_bench_plusplus_image)
class GenBusterBenchPlusplusImage(RealFakeMetricMixin, RealFakeFlatMixin, BaseImageDatasetPP):
    dataset_name = DataSets.genbuster_bench_plusplus_image
