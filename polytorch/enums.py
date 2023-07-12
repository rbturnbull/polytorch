from enum import Enum
import torch.nn.functional as F


class ContinuousLossType(Enum):
    L1_LOSS = 0
    MSE_LOSS = 1
    SMOOTH_L1_LOSS = 2


class BinaryLossType(Enum):
    CROSS_ENTROPY = 0
    IOU = 1
    DICE = 2
    

class CategoricalLossType(Enum):
    CROSS_ENTROPY = 0
    DICE = 1
    # TODO: Focal Loss
    # TODO Wasserstein distance (Earth mover loss)

