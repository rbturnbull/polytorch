from typing import List
import torch
from torch import nn
import torch.nn.functional as F

from .data import OrdinalData, ContinuousData, BinaryData, CategoricalData, PolyData
from .util import split_tensor, permute_feature_axis, squeeze_prediction

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

    def forward(self, predictions, *targets):
        if not isinstance(predictions, (tuple, list)):
            predictions = split_tensor(predictions, self.data_types, feature_axis=self.feature_axis)

        assert len(predictions) == len(targets) == len(self.data_types)

        loss = 0.0
        for prediction, target, data_type in zip(predictions, targets, self.data_types):
            if not hasattr(data_type, "calculate_loss"):
                raise ValueError(f"Data type {data_type} does not have a calculate_loss method")
            
            loss += data_type.calculate_loss(prediction, target, feature_axis=self.feature_axis)

        return loss.mean()