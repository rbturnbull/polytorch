from polytorch.data import PolyData, BinaryData, CategoricalData
import pytest
import torch


def test_polydata_abstract():
    with pytest.raises(TypeError):
        PolyData()


def test_binary_data_loss_junk():
    data = BinaryData(loss_type="junk")
    with pytest.raises(NotImplementedError):
        target = torch.tensor([False, True, False, True, False]).unsqueeze(1)
        data.calculate_loss(target, target)


def test_categorical_data_loss_junk():
    data = CategoricalData(loss_type="junk", category_count=4)
    with pytest.raises(NotImplementedError):
        target = torch.tensor([0, 2, 3, 2, 3]).unsqueeze(1)
        data.calculate_loss(target, target)        


def test_categorical_data_loss_label_smoothing_validator():
    CategoricalData(label_smoothing=0.0, category_count=4)
    CategoricalData(label_smoothing=1.0, category_count=4)
    with pytest.raises(ValueError):
        CategoricalData(label_smoothing=1.1, category_count=4)
    with pytest.raises(ValueError):
        CategoricalData(label_smoothing=-0.1, category_count=4)
        
