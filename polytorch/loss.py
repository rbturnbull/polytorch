from typing import List
from torch import nn

from .data import PolyData
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
        

        if not isinstance(predictions, (tuple, list)):
            predictions = split_tensor(predictions, self.data_types, feature_axis=self.feature_axis)

        assert len(predictions) == len(targets) == len(self.data_types)

        loss = 0.0
        for prediction, target, data_type in zip(predictions, targets, self.data_types):
            feature_axis = self.feature_axis % len(prediction.shape)

            if not hasattr(data_type, "calculate_loss"):
                raise ValueError(f"Data type {data_type} does not have a calculate_loss method")
            
            loss += data_type.loss_weighting * data_type.calculate_loss(prediction, target, feature_axis=feature_axis).mean()

        return loss