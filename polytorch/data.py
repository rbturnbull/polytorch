"""Feature definitions for embeddings, prediction widths, and losses."""

from torch import nn
import abc
from typing import List, Optional, Protocol, Any
from attrs import define, Factory, field, validators
import torch.nn.functional as F
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode

from .util import permute_feature_axis, squeeze_prediction
from .enums import ContinuousLossType, BinaryLossType, CategoricalLossType


class Getter(Protocol):
    """Protocol for values accessible by integer row index."""

    def __getitem__(self, key: int) -> Any:
        """Return the value at the given row index."""
        ...


DELIMITER = "/"


@define(kw_only=True)
class PolyData(abc.ABC):
    """Describe a feature's prediction width, embedding, and loss.

    Attributes:
        name: Feature name, defaulting to the concrete class name.
        loss_weighting: Multiplier applied by PolyLoss to this feature's loss.
    """

    name: str = Factory(lambda self: self.__class__.__name__, takes_self=True)
    loss_weighting: float = field(default=1.0)

    def __setstate__(self, state):
        # Handle (super_state, our_state) tuple pattern
        """Restore serialized attributes, including compatibility with older saved data types."""
        if isinstance(state, tuple) and len(state) == 2:
            _, actual_state = state
        else:
            actual_state = state

        for k, v in actual_state.items():
            setattr(self, k, v)
        if not hasattr(self, "loss_weighting"):
            self.loss_weighting = 1.0

    @abc.abstractmethod
    def embedding_module(self, embedding_size: int) -> nn.Module:
        """Create a module mapping feature values to embedding vectors.

        Args:
            embedding_size: Number of coordinates in each embedding vector.

        Returns:
            A PyTorch module accepting values for this data type.
        """
        pass

    @abc.abstractmethod
    def size(self) -> int:
        """The prediction size of this data type."""
        pass

    @abc.abstractmethod
    def calculate_loss(self, prediction, target, feature_axis: int = -1):
        """Calculate the configured loss from predictions and targets.

        Args:
            prediction: Model predictions, including a feature axis.
            target: Ground-truth values for this data type.
            feature_axis: Axis containing predicted features; defaults to the last.

        Returns:
            A loss tensor whose reduction depends on the concrete data type.
        """
        pass

    @classmethod
    def from_series(cls, name: str, series) -> tuple["PolyData", Getter]:
        """Construct a named data type and expose the series values unchanged.

        Args:
            name: Name assigned to the constructed data type.
            series: A pandas Series containing feature values.

        Returns:
            A pair containing the data type instance and indexable row values.
        """
        return cls(name=name), series.values


def binary_default_factory():
    """Return the default labels for false and true binary values."""
    return ["False", "True"]


@define
class BinaryData(PolyData):
    """Describe a binary feature predicted by a single logit.

    Attributes:
        loss_type: Binary cross-entropy, intersection-over-union, or Dice loss.
        labels: Labels for the false and true values.
        colors: Optional plotting colors for the two values.
    """

    loss_type: BinaryLossType = BinaryLossType.CROSS_ENTROPY
    labels: List[str] = field(factory=binary_default_factory)
    colors: Optional[List[str]] = None

    def embedding_module(self, embedding_size: int) -> nn.Module:
        """Create the embedding module for this feature type."""
        return nn.Embedding(2, embedding_size)

    def size(self) -> int:
        """Return the number of prediction features required by this data type."""
        return 1

    def calculate_loss(self, prediction, target, feature_axis: int = -1):
        """Calculate the configured loss using the supplied prediction feature axis."""
        prediction = squeeze_prediction(prediction, target, feature_axis)

        if self.loss_type == BinaryLossType.CROSS_ENTROPY:
            return F.binary_cross_entropy_with_logits(
                prediction,
                target.float(),
            )
        elif self.loss_type == BinaryLossType.IOU:
            from .metrics import calc_iou

            return 1 - calc_iou(
                prediction.sigmoid(),
                target,
            )
        elif self.loss_type == BinaryLossType.DICE:
            from .metrics import calc_dice_score

            return 1 - calc_dice_score(
                prediction.sigmoid(),
                target,
            )

        raise NotImplementedError(
            f"Unknown loss type: {self.loss_type} for {self.__class__.__name__}"
        )

    def __setstate__(self, state):
        """Restore serialized attributes, including compatibility with older saved data types."""
        super().__setstate__(state)


@define
class CategoricalData(PolyData):
    """Describe an unordered categorical feature.

    Attributes:
        category_count: Number of categories and prediction logits.
        loss_type: Cross-entropy or generalized Dice loss.
        labels: Optional category labels.
        colors: Optional category plotting colors.
        label_smoothing: Cross-entropy smoothing in the inclusive range [0, 1].
    """

    category_count: int
    loss_type: CategoricalLossType = CategoricalLossType.CROSS_ENTROPY
    labels: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    label_smoothing: float = field(
        default=0.0, validator=[validators.ge(0.0), validators.le(1.0)]
    )

    def embedding_module(self, embedding_size: int) -> nn.Module:
        """Create the embedding module for this feature type."""
        return nn.Embedding(self.category_count, embedding_size)

    def size(self) -> int:
        """Return the number of prediction features required by this data type."""
        return self.category_count

    def calculate_loss(self, prediction, target, feature_axis: int = -1):
        """Calculate the configured loss using the supplied prediction feature axis."""
        if self.loss_type == CategoricalLossType.CROSS_ENTROPY:
            prediction = permute_feature_axis(
                prediction, old_axis=feature_axis, new_axis=1
            )
            return F.cross_entropy(
                prediction,
                target.long(),
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
        elif self.loss_type == CategoricalLossType.DICE:
            from .metrics import calc_generalized_dice_score

            return 1.0 - calc_generalized_dice_score(
                prediction.softmax(dim=feature_axis),
                target,
                feature_axis=feature_axis,
            )

        raise NotImplementedError(
            f"Unknown loss type: {self.loss_type} for {self.__class__.__name__}"
        )

    def __setstate__(self, state):
        """Restore serialized attributes, including compatibility with older saved data types."""
        super().__setstate__(state)

    @classmethod
    def from_series(cls, name: str, series) -> "PolyData":
        """Infer the category count and return the original series values.

        Args:
            name: Feature name.
            series: A pandas Series; its non-null unique count sets category_count.

        Returns:
            A pair containing the data type and unchanged series values. String
            labels are not encoded, and numeric values are not remapped to indices.
        """
        return cls(name=name, category_count=series.nunique()), series.values


@define
class OrdinalData(CategoricalData):
    """Describe ordered categories with learned distances in the embedding.

    Attributes:
        color: Optional plotting color. Category count and loss configuration
            are inherited from CategoricalData.
    """

    color: str = ""
    # add in option to estimate distances or to set them?

    def embedding_module(self, embedding_size: int) -> nn.Module:
        """Create the embedding module for this feature type."""
        from .embedding import OrdinalEmbedding

        return OrdinalEmbedding(self.category_count, embedding_size)

    def __setstate__(self, state):
        """Restore serialized attributes, including compatibility with older saved data types."""
        super().__setstate__(state)


@define
class ContinuousData(PolyData):
    """Describe a scalar feature with optional target normalization.

    Attributes:
        loss_type: PyTorch regression loss to apply.
        color: Optional plotting color.
        mean: Value subtracted from targets and embedding inputs when set.
        stdev: Divisor applied after centering when set.
    """

    loss_type: ContinuousLossType = ContinuousLossType.SMOOTH_L1_LOSS
    color: str = ""
    mean: Optional[float] = None
    stdev: Optional[float] = None

    def embedding_module(self, embedding_size: int) -> nn.Module:
        """Create the embedding module for this feature type."""
        from .embedding import ContinuousEmbedding

        return ContinuousEmbedding(embedding_size, mean=self.mean, stdev=self.stdev)

    def size(self) -> int:
        """Return the number of prediction features required by this data type."""
        return 1

    def calculate_loss(self, prediction, target, feature_axis: int = -1):
        """Calculate the configured loss using the supplied prediction feature axis."""
        prediction = squeeze_prediction(prediction, target, feature_axis)
        if self.mean is not None:
            target = target - self.mean
        if self.stdev is not None:
            target = target / self.stdev
        return self.loss_func(prediction, target, reduction="none")

    @property
    def loss_func(self):
        """Return the PyTorch functional loss selected by loss_type."""
        return getattr(F, self.loss_type.name.lower())

    def __setstate__(self, state):
        """Restore serialized attributes, including compatibility with older saved data types."""
        super().__setstate__(state)


@define
class HierarchicalData(PolyData):
    """Describe classes arranged in a hierarchical softmax tree.

    Attributes:
        root: Root SoftmaxNode defining the prediction structure and node IDs.

    Notes:
        Hierarchical data supports prediction and loss, but not embedding.
    """

    root: SoftmaxNode

    def embedding_module(self, embedding_size: int) -> nn.Module:
        """Raise NotImplementedError because hierarchical embeddings are unsupported."""
        raise NotImplementedError(
            "Hierarchical data types are not yet supported for embedding."
        )

    def size(self) -> int:
        """Return the number of prediction features required by this data type."""
        self.root.set_indexes_if_unset()
        return self.root.layer_size

    def calculate_loss(self, prediction, target, feature_axis: int = -1):
        """Calculate the configured loss using the supplied prediction feature axis."""
        prediction = permute_feature_axis(prediction, old_axis=feature_axis, new_axis=1)
        return self.loss_module(prediction, target)

    @property
    def loss_module(self):
        """Create the hierarchical softmax loss associated with this tree."""
        return HierarchicalSoftmaxLoss(root=self.root)

    def __setstate__(self, state):
        """Restore serialized attributes, including compatibility with older saved data types."""
        super().__setstate__(state)

    @classmethod
    def from_series(cls, name: str, series) -> "PolyData":
        """Build a shared hierarchy and map each input path to its terminal node ID.

        Args:
            name: Feature name.
            series: Series of paths separated by ``/``. Components are stripped of
                surrounding whitespace, and empty components are ignored.

        Returns:
            A pair containing the data type and a list of node IDs in input order.
            Identical normalized paths reuse nodes. Multiple top-level nodes are
            attached to a virtual root named ``__root__``.

        Raises:
            ValueError: The series is empty or an entry has no nonempty components.
        """
        lineage_to_node = {}
        roots = []
        nodes = []

        for item in series:
            components = tuple(
                component.strip()
                for component in str(item).split(DELIMITER)
                if component.strip()
            )

            if not components:
                raise ValueError(f"Invalid hierarchical entry '{item}' for '{name}'.")

            current_node = None
            for component_index, component in enumerate(components):
                my_lineage = tuple(components[: component_index + 1])
                if my_lineage not in lineage_to_node:
                    lineage_to_node[my_lineage] = SoftmaxNode(
                        name=component, parent=current_node
                    )
                    if current_node is None:
                        roots.append(lineage_to_node[my_lineage])
                current_node = lineage_to_node[my_lineage]

            nodes.append(current_node)

        assert len(nodes) == len(series)

        if len(roots) > 1:
            root = SoftmaxNode(name="__root__", parent=None, children=roots)
        elif len(roots) == 1:
            root = roots[0]
        else:
            raise ValueError("No data found to build HierarchicalData.")

        root.set_indexes_if_unset()
        node_ids = [root.node_to_id[node] for node in nodes]

        return cls(name=name, root=root), node_ids
