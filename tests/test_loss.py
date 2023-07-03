import torch

from polytorch import PolyLoss, BinaryData, CategoricalData, ContinuousData, OrdinalData
from polytorch.enums import ContinuousDataLossType
import pytest


def test_loss_junk():
    batch_size = 5
    category_count = batch_size
    
    prediction = torch.diag(torch.full((category_count,), 10.0))
    target = torch.arange(category_count)

    loss_fn = PolyLoss(["CategoricalData"])
    with pytest.raises(ValueError):
        loss_fn((prediction,), target)


def test_loss_categorical():
    batch_size = 5
    category_count = batch_size
    
    prediction = torch.diag(torch.full((category_count,), 10.0))
    target = torch.arange(batch_size)

    loss_fn = PolyLoss([CategoricalData(category_count)])
    loss = loss_fn(prediction, target)
    assert loss.item() < 0.01

    # change targets
    loss = loss_fn(prediction, target % 3)
    assert loss.item() > 4.0


def test_loss_binary():
    target = torch.tensor([False, True, False, True, False]).unsqueeze(1)
    prediction = (target.float() - 0.5) * 10.0

    loss_fn = PolyLoss([BinaryData()])
    loss = loss_fn(prediction, target)
    assert loss.item() < 0.01

    # change targets
    # loss = loss_fn(prediction, ~target)
    # assert loss.item() > 4.0


def test_loss_categorical_complex():
    batch_size = 5
    category_count = batch_size
    timesteps = 3
    height = width = 128
    
    prediction = torch.zeros((batch_size, timesteps, category_count, height, width))
    target = torch.zeros( (batch_size, timesteps, height, width), dtype=int)
    for i in range(batch_size):
        prediction[i, :, i, :, :] = 10.0
        target[i, :, :, :] = i

    loss_fn = PolyLoss([CategoricalData(category_count)], feature_axis=2)
    loss = loss_fn(prediction, target)
    assert loss.item() < 0.01

    # change targets
    loss = loss_fn(prediction, target % 3)
    assert loss.item() > 4.0


def test_loss_ordinal():
    batch_size = 5
    category_count = batch_size
    
    prediction = torch.diag(torch.full((category_count,), 10.0))
    target = torch.arange(category_count)

    loss_fn = PolyLoss([OrdinalData(category_count)])
    loss = loss_fn(prediction, target)
    assert loss.item() < 0.01

    # change targets
    loss = loss_fn(prediction, target % 3)
    assert loss.item() > 4.0


def test_loss_continuous_l1():
    batch_size = 5

    prediction = torch.randn((batch_size, 1))
    target = prediction

    loss_fn = PolyLoss([ContinuousData(loss_type=ContinuousDataLossType.L1_LOSS)])
    loss = loss_fn(prediction, target)
    torch.testing.assert_close(loss.item(), 0.0)

    loss = loss_fn(prediction+0.1, target)
    torch.testing.assert_close(loss.item(), 0.1)

    loss = loss_fn(prediction-0.1, target)
    torch.testing.assert_close(loss.item(), 0.1)


def test_loss_continuous_mse():
    batch_size = 5

    prediction = torch.randn((batch_size, 1))
    target = prediction

    loss_fn = PolyLoss([ContinuousData(loss_type=ContinuousDataLossType.MSE_LOSS)])
    loss = loss_fn(prediction, target)
    torch.testing.assert_close(loss.item(), 0.0)

    loss = loss_fn(prediction+0.1, target)
    torch.testing.assert_close(loss.item(), 0.01)

    loss = loss_fn(prediction-0.1, target)
    torch.testing.assert_close(loss.item(), 0.01)


def test_loss_continuous_smooth_l1():
    batch_size = 5

    prediction = torch.randn((batch_size, 1))
    target = prediction

    loss_fn = PolyLoss([ContinuousData(loss_type=ContinuousDataLossType.SMOOTH_L1_LOSS)])
    loss = loss_fn(prediction, target)
    torch.testing.assert_close(loss.item(), 0.0)

    loss = loss_fn(prediction+0.1, target)
    torch.testing.assert_close(loss.item(), 0.005)

    loss = loss_fn(prediction-0.1, target)
    torch.testing.assert_close(loss.item(), 0.005)


def test_loss_continuous_complex():
    batch_size = 5
    timesteps = 3
    height = width = 128

    prediction = torch.randn((batch_size, timesteps, 1, height, width))
    target = prediction.squeeze()

    loss_fn = PolyLoss([ContinuousData(loss_type=ContinuousDataLossType.L1_LOSS)], feature_axis=2)
    loss = loss_fn(prediction, target)
    torch.testing.assert_close(loss.item(), 0.0)

    loss = loss_fn(prediction+0.1, target)
    torch.testing.assert_close(loss.item(), 0.1)

    loss = loss_fn(prediction-0.1, target)
    torch.testing.assert_close(loss.item(), 0.1)


