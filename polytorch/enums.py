"""Loss function choices for PolyTorch data types."""

from enum import Enum
import torch.nn.functional as F


class ContinuousLossType(Enum):
    """Regression losses: absolute error, squared error, or smooth L1."""

    L1_LOSS = 0
    MSE_LOSS = 1
    SMOOTH_L1_LOSS = 2


class BinaryLossType(Enum):
    """Binary losses: logit cross-entropy, intersection-over-union, or Dice."""

    CROSS_ENTROPY = 0
    IOU = 1
    DICE = 2


class CategoricalLossType(Enum):
    """Categorical losses: cross-entropy or generalized Dice."""

    CROSS_ENTROPY = 0
    DICE = 1
    # TODO: Focal Loss
    # TODO Wasserstein distance (Earth mover loss)
