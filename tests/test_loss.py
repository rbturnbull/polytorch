import torch
import pytest
from hierarchicalsoftmax import SoftmaxNode
from polytorch import PolyLoss, BinaryData, CategoricalData, ContinuousData, OrdinalData, HierarchicalData
from polytorch import ContinuousLossType, BinaryLossType, CategoricalLossType

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
    loss = loss_fn(prediction, torch.logical_not(target))
    assert loss.item() > 4.0


def test_loss_binary_iou():
    target = torch.tensor([False, True, False, True, False]).unsqueeze(1)
    prediction = (target.float() - 0.5) * 10.0

    loss_fn = PolyLoss([BinaryData(loss_type=BinaryLossType.IOU)])
    loss = loss_fn(prediction, target)
    assert loss.item() < 0.012

    # change targets
    loss = loss_fn(prediction, torch.logical_not(target))
    assert 0.82 < loss.item() < 0.84


def test_loss_binary_dice():
    target = torch.tensor([False, True, False, True, False]).unsqueeze(1)
    prediction = (target.float() - 0.5) * 10.0

    loss_fn = PolyLoss([BinaryData(loss_type=BinaryLossType.DICE)])
    loss = loss_fn(prediction, target)
    assert loss.item() < 0.01

    # change targets
    loss = loss_fn(prediction, torch.logical_not(target))
    assert 0.82 < loss.item() < 0.84


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


def test_loss_categorical_label_smoothing():
    batch_size = 5
    category_count = batch_size
    timesteps = 3
    height = width = 128
    
    prediction = torch.zeros((batch_size, timesteps, category_count, height, width))
    target = torch.zeros( (batch_size, timesteps, height, width), dtype=int)
    for i in range(batch_size):
        prediction[i, :, i, :, :] = 10.0
        target[i, :, :, :] = i

    loss_fn = PolyLoss([CategoricalData(category_count, label_smoothing=0.0)], feature_axis=2)
    loss = loss_fn(prediction, target)
    assert loss.item() < 0.01

    loss_fn = PolyLoss([CategoricalData(category_count, label_smoothing=0.05)], feature_axis=2)
    loss = loss_fn(prediction, target)
    assert 0.40 < loss.item() < 0.41

    loss_fn = PolyLoss([CategoricalData(category_count, label_smoothing=0.1)], feature_axis=2)
    loss = loss_fn(prediction, target)
    assert 0.80 < loss.item() < 0.81

    # make less certain
    loss = loss_fn(prediction/2, target)
    assert 0.42 < loss.item() < 0.43


def test_loss_categorical_dice():
    batch_size = 5
    category_count = batch_size
    timesteps = 3
    height = width = 128
    
    prediction = torch.zeros((batch_size, timesteps, category_count, height, width))
    target = torch.zeros( (batch_size, timesteps, height, width), dtype=int)
    for i in range(batch_size):
        prediction[i, :, i, :, :] = 10.0
        target[i, :, :, :] = i

    loss_fn = PolyLoss([CategoricalData(category_count, loss_type=CategoricalLossType.DICE)], feature_axis=2)
    loss = loss_fn(prediction, target)
    assert loss.item() < 0.01

    # change targets
    loss = loss_fn(prediction, target % 3)
    torch.testing.assert_close(loss.item(), 1.0)


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

    loss_fn = PolyLoss([ContinuousData(loss_type=ContinuousLossType.L1_LOSS)])
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

    loss_fn = PolyLoss([ContinuousData(loss_type=ContinuousLossType.MSE_LOSS)])
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

    loss_fn = PolyLoss([ContinuousData(loss_type=ContinuousLossType.SMOOTH_L1_LOSS)])
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

    loss_fn = PolyLoss([ContinuousData(loss_type=ContinuousLossType.L1_LOSS)], feature_axis=2)
    loss = loss_fn(prediction, target)
    torch.testing.assert_close(loss.item(), 0.0)

    loss = loss_fn(prediction+0.1, target)
    torch.testing.assert_close(loss.item(), 0.1)

    loss = loss_fn(prediction-0.1, target)
    torch.testing.assert_close(loss.item(), 0.1)


# def test_loss_continuous_complex_normalize():
#     batch_size = 5
#     timesteps = 3
#     height = width = 128
#     mean = 1_000_000
#     stdev = 100_000

#     torch.manual_seed(0)
#     prediction = torch.randn((batch_size, timesteps, 1, height, width)) * stdev + mean
#     target = prediction.squeeze()

#     breakpoint()
#     loss_fn = PolyLoss([ContinuousData(loss_type=ContinuousLossType.L1_LOSS, mean=mean, stdev=stdev),], feature_axis=2)
#     loss = loss_fn(prediction, target)
#     torch.testing.assert_close(loss.item(), 0.0)

#     loss = loss_fn(prediction+0.1, target)
#     torch.testing.assert_close(loss.item(), 0.1)

#     loss = loss_fn(prediction-0.1, target)
#     torch.testing.assert_close(loss.item(), 0.1)


def test_hierarchical_data_loss():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    targets = [aa,ba,bb, ab]
    target_tensor = root.get_node_ids_tensor(targets)

    predictions = torch.zeros( (len(targets), root.layer_size) )

    # Test blank is inaccurate
    loss_fn = PolyLoss([HierarchicalData(root=root)])

    value = loss_fn(predictions, target_tensor)
    assert value > 1.38

    # Test accurate
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    value = loss_fn(predictions, target_tensor)
    assert value < 0.0001

