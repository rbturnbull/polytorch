"""Linear output layers that infer and split heterogeneous prediction widths."""

from torch import nn
from typing import List

from .data import PolyData
from .util import total_size, split_tensor


class PolyLayerError(RuntimeError):
    """Raised when a PolyTorch layer receives an explicit output feature count."""

    pass


class PolyLayerMixin:
    """Derive a linear layer width from output types and split its predictions."""

    def __init__(self, output_types: List[PolyData], out_features=None, **kwargs):
        """Configure the layer's output types and inferred prediction width.

        Args:
            output_types: Ordered definitions used to determine and split outputs.
            out_features: Must be None; the total output width is inferred.
            **kwargs: Arguments forwarded to the underlying PyTorch linear layer.

        Raises:
            PolyLayerError: An explicit out_features value is provided.
        """
        self.output_types = output_types

        if out_features is not None:
            raise PolyLayerError(
                "Trying to create a PolyLinear Layer by explicitly setting `out_features`. "
                "This value should be determined from the list of output types and not the `out_features` argument."
            )

        super().__init__(out_features=total_size(self.output_types), **kwargs)

    def forward(self, *inputs):
        """Apply the linear layer and return a prediction tensor for each output type."""
        outputs = super().forward(*inputs)
        return split_tensor(outputs, self.output_types, feature_axis=-1)


class PolyLinear(PolyLayerMixin, nn.Linear):
    """
    Creates a linear layer designed to be the final layer in a neural network model that produces unnormalized scores given to PolyLoss.

    The ``out_features`` value is the sum of the output data type sizes and
    cannot be supplied explicitly. Predictions are returned as a tuple in
    output_types order. Other arguments follow the underlying PyTorch layer.
    """


class PolyLazyLinear(PolyLayerMixin, nn.LazyLinear):
    """
    Creates a lazy linear layer designed to be the final layer in a neural network model that produces unnormalized scores given to PolyLoss.

    The ``out_features`` value is the sum of the output data type sizes and
    cannot be supplied explicitly. Predictions are returned as a tuple in
    output_types order. Other arguments follow the underlying PyTorch layer.
    """
