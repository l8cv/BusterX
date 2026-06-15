from busterx.datasetpp.base import BaseVideoDatasetPP
from busterx.datasetpp.mixin import RealFakeFlatMixin
from busterx.registry import DATASET_REGISTRY, DataSets


@DATASET_REGISTRY.register(name=DataSets.genbuster_200k)
class GenBuster200K(RealFakeFlatMixin, BaseVideoDatasetPP):
    dataset_name = DataSets.genbuster_200k
