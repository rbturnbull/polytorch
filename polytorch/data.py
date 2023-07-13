from torch import nn
import abc
from typing import List, Optional
from dataclasses import dataclass, field
import torch.nn.functional as F
from .util import permute_feature_axis, squeeze_prediction
from functools import cached_property

from .enums import ContinuousLossType, BinaryLossType, CategoricalLossType

@dataclass(kw_only=True)
class PolyData(abc.ABC):
    name: str = ""
    _name: str = field(init=False, repr=False)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if type(value) is property or not value:
            value = self.__class__.__name__
        self._name = value

    @abc.abstractmethod
    def embedding_module(self, embedding_size:int) -> nn.Module:
        pass

    @abc.abstractmethod
    def size(self) -> int:
        pass

    @abc.abstractmethod
    def calculate_loss(self, prediction, target, feature_axis:int=-1):
        pass


def binary_default_factory():
    return ["False", "True"]


@dataclass
class BinaryData(PolyData):
    loss_type:BinaryLossType = BinaryLossType.CROSS_ENTROPY
    labels:List[str] = field(default_factory=binary_default_factory)
    colors:Optional[List[str]] = None

    def embedding_module(self, embedding_size:int) -> nn.Module:
        return nn.Embedding(2, embedding_size)

    def size(self) -> int:
        return 1

    def calculate_loss(self, prediction, target, feature_axis:int=-1):
        prediction = squeeze_prediction(prediction, target, feature_axis)

        if self.loss_type == BinaryLossType.CROSS_ENTROPY:
            return F.binary_cross_entropy_with_logits(
                prediction, 
                target.float(), 
            )
        elif self.loss_type == BinaryLossType.IOU:
            from .metrics import calc_iou
            return 1 - calc_iou(
                prediction.sigmoid(), 
                target, 
            )
        elif self.loss_type == BinaryLossType.DICE:
            from .metrics import calc_dice_score
            return 1 - calc_dice_score(
                prediction.sigmoid(), 
                target, 
            )
        
        raise NotImplementedError(f"Unknown loss type: {self.loss_type} for {self.__class__.__name__}")


@dataclass
class CategoricalData(PolyData):
    category_count:int
    loss_type:CategoricalLossType = CategoricalLossType.CROSS_ENTROPY
    labels:Optional[List[str]] = None
    colors:Optional[List[str]] = None

    def embedding_module(self, embedding_size:int) -> nn.Module:
        return nn.Embedding(self.category_count, embedding_size)

    def size(self) -> int:
        return self.category_count

    def calculate_loss(self, prediction, target, feature_axis:int=-1):
        if self.loss_type == CategoricalLossType.CROSS_ENTROPY:
            prediction = permute_feature_axis(prediction, old_axis=feature_axis, new_axis=1)
            return F.cross_entropy(
                prediction, 
                target.long(), 
                reduction="none", 
                # label_smoothing=self.label_smoothing,
            )
        elif self.loss_type == CategoricalLossType.DICE:
            from .metrics import calc_generalized_dice_score
            return 1. - calc_generalized_dice_score(
                prediction.softmax(dim=feature_axis), 
                target, 
                n_classes=self.category_count,
                feature_axis=feature_axis,
            )
        
        raise NotImplementedError(f"Unknown loss type: {self.loss_type} for {self.__class__.__name__}")


@dataclass
class OrdinalData(CategoricalData):
    color:str = ""
    # add in option to estimate distances or to set them?

    def embedding_module(self, embedding_size:int) -> nn.Module:
        from .embedding import OrdinalEmbedding
        return OrdinalEmbedding(self.category_count, embedding_size)


@dataclass
class ContinuousData(PolyData):
    loss_type:ContinuousLossType = ContinuousLossType.SMOOTH_L1_LOSS
    color:str = ""

    def embedding_module(self, embedding_size:int) -> nn.Module:
        from .embedding import ContinuousEmbedding
        return ContinuousEmbedding(embedding_size)

    def size(self) -> int:
        return 1
    
    def calculate_loss(self, prediction, target, feature_axis:int=-1):
        prediction = squeeze_prediction(prediction, target, feature_axis)
        return self.loss_func(prediction, target, reduction="none")

    @cached_property
    def loss_func(self):
        return getattr(F, self.loss_type.name.lower())

