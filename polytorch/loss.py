from typing import List
import torch
from torch import nn
import torch.nn.functional as F

from .data import OrdinalData, ContinuousData, CategoricalData, PolyData
from .util import split_tensor

class PolyLoss(nn.Module):
    def __init__(
        self,
        data_types:List[PolyData],
        feature_axis:int=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_types = data_types   
        self.feature_axis = feature_axis 

    def permute_to_feature_axis(self, prediction:torch.Tensor, new_axis:int=1) -> torch.Tensor:
        """
        Changes the shape of the predictions tensor so that the feature axis is 

        Args:
            predictions (torch.Tensor): The predictions tensor.
            new_axis (int): The desired index of the feature axis.

        Returns:
            torch.Tensor: The predictions tensor with the feature axis at the specified index.
        """
        if self.feature_axis % len(prediction.shape) != new_axis:
            axes = list(range(len(prediction.shape)))
            axes.insert(new_axis, axes.pop(self.feature_axis))
            return prediction.permute(*axes)
        return prediction

    def forward(self, predictions, *targets):
        if not isinstance(predictions, (tuple, list)):
            predictions = split_tensor(predictions, self.data_types, feature_axis=self.feature_axis)

        assert len(predictions) == len(targets) == len(self.data_types)

        loss = 0.0

        for prediction, target, data_type in zip(predictions, targets, self.data_types):
            if isinstance(data_type, ContinuousData):
                target_loss = data_type.loss_type(prediction, target, reduction="none")
            elif isinstance(data_type, CategoricalData) or isinstance(data_type, OrdinalData):
                # TODO Focal Loss
                # TODO Earth mover loss (Wasserstein distance) for ordinal data
                # cross-entropy over axis 1
                prediction = self.permute_to_feature_axis(prediction, new_axis=1)
                target_loss = F.cross_entropy(
                    prediction, 
                    target.long(), 
                    reduction="none", 
                    # label_smoothing=self.label_smoothing,
                )
            else:
                raise ValueError("Unknown data type")
            
            loss += target_loss

        return loss.mean()