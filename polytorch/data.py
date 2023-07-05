from torch import nn
from typing import List
from dataclasses import dataclass, field

from .enums import ContinuousDataLossType, BinaryDataLossType

class PolyData():
    name:str = ""

    def create_module(self, embedding_size:int):
        raise NotImplementedError(f'Use either BinaryData, CategoricalData, OrdinalData or ContinuousData')

    def size(self) -> int:
        raise NotImplementedError(f'Use either BinaryData, CategoricalData, OrdinalData or ContinuousData')

    def get_name(self) -> str:
        return self.name or self.__class__.__name__


def binary_default_factory():
    return ["False", "True"]


@dataclass
class BinaryData(PolyData):
    loss_type:BinaryDataLossType = BinaryDataLossType.CROSS_ENTROPY
    labels:List[str] = field(default_factory=binary_default_factory)
    colors:List[str] = None

    def create_module(self, embedding_size:int):
        return nn.Embedding(2, embedding_size)

    def size(self) -> int:
        return 1


@dataclass
class CategoricalData(PolyData):
    category_count:int
    labels:List[str] = None
    colors:List[str] = None

    def create_module(self, embedding_size:int):
        return nn.Embedding(self.category_count, embedding_size)

    def size(self) -> int:
        return self.category_count


@dataclass
class OrdinalData(CategoricalData):
    color:str = ""
    # add in option to estimate distances or to set them?

    def create_module(self, embedding_size:int):
        from .embedding import OrdinalEmbedding
        return OrdinalEmbedding(self.category_count, embedding_size)


@dataclass
class ContinuousData(PolyData):
    loss_type:ContinuousDataLossType = ContinuousDataLossType.SMOOTH_L1_LOSS
    color:str = ""

    def create_module(self, embedding_size:int):
        from .embedding import ContinuousEmbedding
        return ContinuousEmbedding(embedding_size)

    def size(self) -> int:
        return 1