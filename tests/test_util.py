
from polytorch.data import OrdinalData, ContinuousData, CategoricalData
from polytorch.util import total_size, split_tensor
import torch


def test_total_size():
    assert total_size([OrdinalData(category_count=7), ContinuousData(), CategoricalData(category_count=5)]) == 13


def test_split_tensor():
    data_types = [OrdinalData(category_count=7), ContinuousData(), CategoricalData(category_count=5)]
    size = total_size(data_types)
    batch_size = 10
    
    x = torch.zeros( (batch_size, size))

    y = split_tensor(x, data_types)
    assert len(y) == len(data_types)
    assert y[0].shape == (batch_size, 7)
    assert y[1].shape == (batch_size, 1)
    assert y[2].shape == (batch_size, 5)


def test_split_tensor_feature_dim():
    data_types = [OrdinalData(category_count=7), ContinuousData(), CategoricalData(category_count=5)]
    size = total_size(data_types)
    timesteps = 4
    batch_size = 10
    width = height = 128
    
    x = torch.zeros( (batch_size, timesteps, size, width, height))

    y = split_tensor(x, data_types, feature_axis=2)
    assert len(y) == len(data_types)
    assert y[0].shape == (batch_size, timesteps, 7, width, height)
    assert y[1].shape == (batch_size, timesteps, 1, width, height)
    assert y[2].shape == (batch_size, timesteps, 5, width, height)
