from enum import Enum
import torch.nn.functional as F
from functools import cached_property


class ContinuousDataLossType(Enum):
    L1_LOSS = 0
    MSE_LOSS = 1
    SMOOTH_L1_LOSS = 2

    @cached_property
    def loss_func(self):
        return getattr(F, self.name.lower())

    def __call__(self, *args, **kwargs):
        return self.loss_func(*args, **kwargs)


class BinaryDataLossType(Enum):
    CROSS_ENTROPY = 0
    IOU = 1
    DICE = 2

    def __call__(self, prediction, target):
        if self == BinaryDataLossType.CROSS_ENTROPY:
            return F.binary_cross_entropy_with_logits(
                prediction, 
                target.float(), 
            )
        elif self == BinaryDataLossType.IOU:
            from .metrics import calc_iou
            return 1 - calc_iou(
                prediction.sigmoid(), 
                target, 
            )
        elif self == BinaryDataLossType.DICE:
            from .metrics import calc_dice_score
            return 1 - calc_dice_score(
                prediction.sigmoid(), 
                target, 
            )
        
        raise NotImplementedError("Dice loss not implemented yet")
    