# -*- coding: future_typing -*-

from torch import nn
from dataclasses import dataclass


class PolyData():
    def create_module(self, embedding_size:int):
        raise NotImplementedError(f'Use either CategoricalData, OrdinalData or ContinuousData')


@dataclass
class CategoricalData(PolyData):
    category_count:int

    def create_module(self, embedding_size:int):
        return nn.Embedding(self.category_count, embedding_size)


@dataclass
class OrdinalData(CategoricalData):
    # add in option to estimate distances or to set them?

    def create_module(self, embedding_size:int):
        from .embedding import OrdinalEmbedding
        return OrdinalEmbedding(self.category_count, embedding_size)


class ContinuousData(PolyData):
    def create_module(self, embedding_size:int):
        from .embedding import ContinuousEmbedding
        return ContinuousEmbedding(embedding_size)
