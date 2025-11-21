from torch import nn
import abc
from typing import List, Optional, Protocol, Any
from attrs import define, Factory, field, validators
import torch.nn.functional as F
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode

from .util import permute_feature_axis, squeeze_prediction
from .enums import ContinuousLossType, BinaryLossType, CategoricalLossType


class Getter(Protocol):
    def __getitem__(self, key: int) -> Any: ...

DELIMITER = "→"

@define(kw_only=True)
class PolyData(abc.ABC):
    name: str = Factory(lambda self: self.__class__.__name__, takes_self=True)
    loss_weighting: float = field(default=1.0)

    def __setstate__(self, state):
        # Handle (super_state, our_state) tuple pattern
        if isinstance(state, tuple) and len(state) == 2:
            _, actual_state = state
        else:
            actual_state = state

        for k, v in actual_state.items():
            setattr(self, k, v)
        if not hasattr(self, "loss_weighting"):
            self.loss_weighting = 1.0

    @abc.abstractmethod
    def embedding_module(self, embedding_size:int) -> nn.Module:
        pass

    @abc.abstractmethod
    def size(self) -> int:
        """ The prediction size of this data type. """
        pass

    @abc.abstractmethod
    def calculate_loss(self, prediction, target, feature_axis:int=-1):
        pass

    @classmethod
    def from_series(cls, name:str, series) -> tuple["PolyData", Getter]:
        return cls(name=name), series.values


def binary_default_factory():
    return ["False", "True"]


@define
class BinaryData(PolyData):
    loss_type:BinaryLossType = BinaryLossType.CROSS_ENTROPY
    labels:List[str] = field(factory=binary_default_factory)
    colors:Optional[List[str]] = None

    def embedding_module(self, embedding_size:int) -> nn.Module:
        return nn.Embedding(2, embedding_size)

    def size(self) -> int:
        return 1

    def calculate_loss(self, prediction, target, feature_axis:int=-1):
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
        
        raise NotImplementedError(f"Unknown loss type: {self.loss_type} for {self.__class__.__name__}")

    def __setstate__(self, state):
        super().__setstate__(state)


@define
class CategoricalData(PolyData):
    category_count:int
    loss_type:CategoricalLossType = CategoricalLossType.CROSS_ENTROPY
    labels:Optional[List[str]] = None
    colors:Optional[List[str]] = None
    label_smoothing:float = field(default=0.0, validator=[validators.ge(0.0), validators.le(1.0)])

    def embedding_module(self, embedding_size:int) -> nn.Module:
        return nn.Embedding(self.category_count, embedding_size)

    def size(self) -> int:
        return self.category_count

    def calculate_loss(self, prediction, target, feature_axis:int=-1):
        if self.loss_type == CategoricalLossType.CROSS_ENTROPY:
            prediction = permute_feature_axis(prediction, old_axis=feature_axis, new_axis=1)
            return F.cross_entropy(
                prediction, 
                target.long(), 
                reduction="none", 
                label_smoothing=self.label_smoothing,
            )
        elif self.loss_type == CategoricalLossType.DICE:
            from .metrics import calc_generalized_dice_score
            return 1. - calc_generalized_dice_score(
                prediction.softmax(dim=feature_axis), 
                target, 
                feature_axis=feature_axis,
            )
        
        raise NotImplementedError(f"Unknown loss type: {self.loss_type} for {self.__class__.__name__}")

    def __setstate__(self, state):
        super().__setstate__(state)
        
    @classmethod
    def from_series(cls, name:str, series) -> "PolyData":
        return cls(name=name, category_count=series.nunique()), series.values


@define
class OrdinalData(CategoricalData):
    color:str = ""
    # add in option to estimate distances or to set them?

    def embedding_module(self, embedding_size:int) -> nn.Module:
        from .embedding import OrdinalEmbedding
        return OrdinalEmbedding(self.category_count, embedding_size)

    def __setstate__(self, state):
        super().__setstate__(state)


@define
class ContinuousData(PolyData):
    loss_type:ContinuousLossType = ContinuousLossType.SMOOTH_L1_LOSS
    color:str = ""
    mean:Optional[float] = None
    stdev:Optional[float] = None

    def embedding_module(self, embedding_size:int) -> nn.Module:
        from .embedding import ContinuousEmbedding
        return ContinuousEmbedding(embedding_size, mean=self.mean, stdev=self.stdev)

    def size(self) -> int:
        return 1
    
    def calculate_loss(self, prediction, target, feature_axis:int=-1):
        prediction = squeeze_prediction(prediction, target, feature_axis)
        if self.mean is not None:
            target = target - self.mean
        if self.stdev is not None:
            target = target / self.stdev
        return self.loss_func(prediction, target, reduction="none")

    @property
    def loss_func(self):
        return getattr(F, self.loss_type.name.lower())

    def __setstate__(self, state):
        super().__setstate__(state)


@define
class HierarchicalData(PolyData):
    root:SoftmaxNode

    def embedding_module(self, embedding_size:int) -> nn.Module:
        raise NotImplementedError("Hierarchical data types are not yet supported for embedding.")

    def size(self) -> int:
        self.root.set_indexes_if_unset()
        return self.root.layer_size
    
    def calculate_loss(self, prediction, target, feature_axis:int=-1):
        prediction = permute_feature_axis(prediction, old_axis=feature_axis, new_axis=1)
        return self.loss_module(prediction, target)

    @property
    def loss_module(self):
        return HierarchicalSoftmaxLoss(root=self.root)

    def __setstate__(self, state):
        super().__setstate__(state)

    @classmethod
    def from_series(cls, name:str, series) -> "PolyData":
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
                my_lineage = tuple(components[:component_index + 1])
                if my_lineage not in lineage_to_node:
                    lineage_to_node[my_lineage] = SoftmaxNode(name=component, parent=current_node)
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
        
