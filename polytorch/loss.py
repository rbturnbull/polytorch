"""Weighted loss aggregation for models with multiple output types."""

from typing import List
from torch import nn

from .data import PolyData
from .util import split_tensor


class PolyLoss(nn.Module):
    """Combine the mean losses for multiple output data types.

    Args:
        data_types: Ordered output data types, including their loss weights.
        feature_axis: Feature axis in prediction tensors.
        **kwargs: Additional arguments passed to nn.Module.
    """

    def __init__(
        self,
        data_types: List[PolyData],
        feature_axis: int = -1,
        **kwargs,
    ):
        """Initialize the embedding parameters and configuration described by the class."""
        super().__init__(**kwargs)
        self.data_types = data_types
        self.feature_axis = feature_axis

    def forward(self, predictions, *targets):
        """Return the sum of weighted mean losses for the supplied targets.

        Args:
            predictions: A concatenated prediction tensor or a sequence containing
                one prediction tensor per data type.
            *targets: One target tensor per data type, in matching order.

        Returns:
            A scalar tensor summing each mean loss times its loss_weighting.

        Raises:
            AssertionError: Prediction, target, and data type counts do not match.
            ValueError: A data type does not provide calculate_loss.
        """
        if not isinstance(predictions, (tuple, list)):
            predictions = split_tensor(
                predictions, self.data_types, feature_axis=self.feature_axis
            )

        assert len(predictions) == len(targets) == len(self.data_types)

        loss = 0.0
        for prediction, target, data_type in zip(predictions, targets, self.data_types):
            feature_axis = self.feature_axis % len(prediction.shape)

            if not hasattr(data_type, "calculate_loss"):
                raise ValueError(
                    f"Data type {data_type} does not have a calculate_loss method"
                )

            loss += (
                data_type.loss_weighting
                * data_type.calculate_loss(
                    prediction, target, feature_axis=feature_axis
                ).mean()
            )

        return loss
