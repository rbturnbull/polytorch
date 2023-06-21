from typing import List
import torch
from torch import nn
import torch.nn.functional as F

from .data import OrdinalData, ContinuousData, CategoricalData, PolyData
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
            if isinstance(data_type, ContinuousData):
                prediction = squeeze_prediction(prediction, target, self.feature_axis)
                target_loss = data_type.loss_type(prediction, target, reduction="none")
            elif isinstance(data_type, CategoricalData) or isinstance(data_type, OrdinalData):
                # TODO Focal Loss
                # TODO Earth mover loss (Wasserstein distance) for ordinal data
                # cross-entropy over axis 1
                prediction = permute_feature_axis(prediction, old_axis=self.feature_axis, new_axis=1)
                target_loss = F.cross_entropy(
                    prediction, 
                    target.long(), 
                    reduction="none", 
                    # label_smoothing=self.label_smoothing,
                )
            else:
                raise ValueError("Unknown data type")
            print(target_loss.mean())
            loss += target_loss

        return loss.mean()