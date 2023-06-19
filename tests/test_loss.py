import torch

from polytorch import PolyLoss, CategoricalData, ContinuousData, OrdinalData
from polytorch.enums import ContinuousDataLossType


def test_loss_categorical():
    batch_size = 5
    category_count = batch_size
    
    prediction = torch.diag(torch.full((category_count,), 10.0))
    target = torch.arange(category_count)

    loss_fn = PolyLoss([CategoricalData(category_count)])
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


