from typing import List, Tuple
from torch import Tensor


def total_size(data_types) -> int:
    """
    Calculates the total number of features required to predict a list of output types.

    Args:
        data_types (List[PolyData]): The data types to predict.

    Returns:
        int: The number of features required to predict the given data types.
    """
    return sum(data_type.size() for data_type in data_types)


def split_tensor(tensor:Tensor, data_types, feature_axis:int=-1) -> Tuple[Tensor, ...]:
    """
    Splits a tensor into a tuple of tensors, one for each data type.

    Args:
        tensor (Tensor): The predictions tensor.
        data_types (List[PolyData]): The data types to predict.
        feature_axis (int, optional): The axis which has the features to predict. Defaults to last axis.

    Returns:
        Tuple[Tensor, ...]: A tuple of tensors, one for each data type.
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


def permute_feature_axis(tensor:Tensor, old_axis:int, new_axis:int) -> Tensor:
    """
    Changes the shape of a tensor so that the feature axis is in a new axis.

    Args:
        tensor (torch.Tensor): The tensor to permute.
        new_axis (int): The desired index of the feature axis.

    Returns:
        torch.Tensor: The predictions tensor with the feature axis at the specified index.
    """
    axes_count = len(tensor.shape)
    if old_axis % axes_count != new_axis % axes_count:
        axes = list(range(axes_count))
        axes.insert(new_axis, axes.pop(old_axis))
        return tensor.permute(*axes)
    return tensor


def squeeze_prediction(prediction:Tensor, target:Tensor, feature_axis:int):
    """
    Squeeze feature axis if necessary
    """
    feature_axis = feature_axis % len(prediction.shape)
    if (
        len(prediction.shape) == len(target.shape) + 1 and 
        prediction.shape[:feature_axis] + prediction.shape[feature_axis+1:] == target.shape
    ):
        prediction = prediction.squeeze(feature_axis)
    return prediction
