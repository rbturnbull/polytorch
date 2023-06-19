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
