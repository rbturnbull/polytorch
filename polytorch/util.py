from typing import List, Tuple
from torch import Tensor

from .data import PolyData


def total_size(data_types:List[PolyData]) -> int:
    """
    Calculates the total number of features required to predict a list of output types.

    Args:
        data_types (List[PolyData]): The data types to predict.

    Returns:
        int: The number of features required to predict the given data types.
    """
    return sum(data_type.size() for data_type in data_types)


def split_tensor(tensor:Tensor, data_types:List[PolyData], feature_axis:int=-1) -> Tuple[Tensor]:
    """
    Splits a tensor into a tuple of tensors, one for each data type.

    Args:
        tensor (Tensor): The predictions tensor.
        data_types (List[PolyData]): The data types to predict.
        feature_axis (int, optional): The axis which has the features to predict. Defaults to last axis.

    Returns:
        Tuple[Tensor]: A tuple of tensores, one for each data type.
    """
    current_index = 0
    split_tensors = []
    slice_indices = [slice(0, None)] * len(tensor.shape)
    for data_type in data_types:
        size = data_type.size()
        slice_indices[feature_axis] = slice(current_index,current_index+size)
        split_tensors.append(tensor[slice_indices])
        current_index += size
    
    assert current_index == tensor.shape[feature_axis]

    return tuple(split_tensors)