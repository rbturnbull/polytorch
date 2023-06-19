from typing import List
from torch import Tensor


def total_size(data_types) -> int:
    return sum(data_type.size() for data_type in data_types)


def split_tensor(tensor, data_types, feature_axis=-1) -> List[Tensor]:
    current_index = 0
    split_tensors = []
    slice_indices = [slice(0, None)] * len(tensor.shape)
    for data_type in data_types:
        size = data_type.size()
        slice_indices[feature_axis] = slice(current_index,current_index+size)
        split_tensors.append(tensor[slice_indices])
        current_index += size
    
    assert current_index == tensor.shape[feature_axis]

    return split_tensors