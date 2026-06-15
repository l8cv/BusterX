from busterx.datasetpp.base import BaseImageDatasetPP, BaseVideoDatasetPP
from busterx.datasetpp.mixin import RealFakeFlatMixin
from busterx.registry import DATASET_REGISTRY, DataSets


@DATASET_REGISTRY.register(name=DataSets.genbuster_unified_video)
class GenBusterUnifiedVideo(RealFakeFlatMixin, BaseVideoDatasetPP):
    dataset_name = DataSets.genbuster_unified_video


@DATASET_REGISTRY.register(name=DataSets.genbuster_unified_image)
class GenBusterUnifiedImage(RealFakeFlatMixin, BaseImageDatasetPP):
    dataset_name = DataSets.genbuster_unified_image
