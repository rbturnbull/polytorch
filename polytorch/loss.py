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

    def forward(self, predictions, *targets):
        if not isinstance(predictions, tuple):
            predictions = split_tensor(predictions, self.data_types, feature_axis=self.feature_axis)

        assert len(predictions) == len(targets) == len(self.data_types)

        loss = 0.0

        for prediction, target, data_type in zip(predictions, targets, self.data_types):
            if isinstance(data_type, OrdinalData):
                # TODO Earth Mover Loss
                prediction = prediction.permute(0, 2, 1, 3, 4) # softmax over axis 1
                target_loss = F.cross_entropy(prediction, target.long(), reduction="none")
            elif isinstance(data_type, ContinuousData):
                if data_type.loss_type.lower() == "l1":
                    loss_func = F.l1_loss
                elif data_type.loss_type.lower() == "msa":
                    loss_func = F.mse_loss
                else:
                    # smooth l1 loss default
                    loss_func = F.smooth_l1_loss
                    
                target_loss = loss_func(prediction, target, reduction="none")

            elif isinstance(data_type, CategoricalData):
                # TODO Focal Loss
                prediction = prediction.permute(0, 2, 1, 3, 4) # softmax over axis 1
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