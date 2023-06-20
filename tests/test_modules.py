import torch

from polytorch.modules import PolyLinear
from polytorch import CategoricalData, ContinuousData, OrdinalData


def test_linear():
    batch_size = 5
    input_size = 8

    x = torch.randn((batch_size, input_size))

    module = PolyLinear(in_features=input_size, output_types=[ContinuousData()])

    result = module(x)
    len(result) == 1
    assert result[0].shape == (batch_size, 1)

def test_linear_multi():
    batch_size = 5
    input_size = 8
    data_types = [OrdinalData(category_count=7), ContinuousData(), CategoricalData(category_count=5)]

    x = torch.randn((batch_size, input_size))

    module = PolyLinear(in_features=input_size, output_types=data_types)

    result = module(x)
    len(result) == len(data_types)
    assert result[0].shape == (batch_size, 7)
    assert result[1].shape == (batch_size, 1)
    assert result[2].shape == (batch_size, 5)

    