from torch import nn
from typing import List

from .data import PolyData
from .util import total_size, split_tensor


class PolyLayerError(RuntimeError):
    pass


class PolyLayerMixin():
    def __init__(self, output_types:List[PolyData], out_features=None, **kwargs):
        self.output_types = output_types

        if out_features is not None:
            raise PolyLayerError(
                "Trying to create a PolyLinear Layer by explicitly setting `out_features`. "
                "This value should be determined from the list of output types and not the `out_features` argument."
            )

        super().__init__(out_features=total_size(self.output_types), **kwargs)

    def forward(self, *inputs):
        outputs = super().forward(*inputs)
        return split_tensor(outputs, self.output_types, feature_axis=-1)


class PolyLinear(PolyLayerMixin, nn.Linear):
    """
    Creates a linear layer designed to be the final layer in a neural network model that produces unnormalized scores given to PolyLoss.

    The `out_features` value is set internally from root.layer_size and cannot be given as an argument.
    """

class PolyLazyLinear(PolyLayerMixin, nn.LazyLinear):
    """
    Creates a lazy linear layer designed to be the final layer in a neural network model that produces unnormalized scores given to PolyLoss.

    The `out_features` value is set internally from root.layer_size and cannot be given as an argument.
    """
