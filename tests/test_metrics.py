import pytest
import torch
from torch.nn import functional as F
from hierarchicalsoftmax import SoftmaxNode

from polytorch.metrics import (
    categorical_accuracy, 
    mse, l1, smooth_l1, 
    binary_accuracy, binary_dice, binary_iou, 
    calc_generalized_dice_score, generalized_dice,
    CategoricalAccuracy,
    PolyMetric,
    HierarchicalGreedyAccuracy,
)

def test_categorical_accuracy():
    batch_size = 5
    category_count = batch_size
    
    prediction = torch.diag(torch.full((category_count,), 10.0))
    target = torch.arange(category_count)

    accuracy = categorical_accuracy(prediction, target, data_index=0)
    torch.testing.assert_close(accuracy.item(), 1.0)

    # change targets
    accuracy = categorical_accuracy(prediction, target % 3, data_index=0)
    torch.testing.assert_close(accuracy.item(), 0.6)


def test_categorical_accuracy_class():
    batch_size = 5
    category_count = batch_size
    
    prediction = torch.diag(torch.full((category_count,), 10.0))
    target = torch.arange(category_count)

    metric = CategoricalAccuracy(data_index=0)
    accuracy = metric(prediction, target)
    torch.testing.assert_close(accuracy.item(), 1.0)

    # change targets
    accuracy = metric(prediction, target % 3)
    torch.testing.assert_close(accuracy.item(), 0.6)

    assert metric.name == metric.__name__ == "CategoricalAccuracy"


def test_binary_accuracy():
    target = torch.tensor([False, True, False, True, False]).unsqueeze(1)
    prediction = (target.float() - 0.5) * 10.0

    accuracy = binary_accuracy(prediction, target, data_index=0)
    torch.testing.assert_close(accuracy.item(), 1.0)

    # change predictions
    accuracy = binary_accuracy(torch.ones_like(prediction), target, data_index=0)
    torch.testing.assert_close(accuracy.item(), 0.4)


def test_binary_dice():
    target = torch.tensor([False, True, False, True, False]).unsqueeze(1)
    prediction = (target.float() - 0.5) * 10.0

    dice = binary_dice(prediction, target, data_index=0)
    torch.testing.assert_close(dice.item(), 1.0)

    # change predictions
    dice = binary_dice(torch.ones_like(prediction), target, data_index=0)
    torch.testing.assert_close(dice.item(), 0.625)


def test_binary_iou():
    target = torch.tensor([False, True, False, True, False]).unsqueeze(1)
    prediction = (target.float() - 0.5) * 10.0

    iou = binary_iou(prediction, target, data_index=0)
    torch.testing.assert_close(iou.item(), 1.0)

    # change predictions
    iou = binary_iou(torch.ones_like(prediction), target, data_index=0)
    torch.testing.assert_close(iou.item(), 0.5)


def test_categorical_accuracy_complex():
    batch_size = 5
    category_count = batch_size
    timesteps = 3
    height = width = 128
    
    prediction = torch.zeros((batch_size, timesteps, category_count, height, width))
    target = torch.zeros( (batch_size, timesteps, height, width), dtype=int)
    for i in range(batch_size):
        prediction[i, :, i, :, :] = 10.0
        target[i, :, :, :] = i
    
    accuracy = categorical_accuracy(prediction, target, data_index=0, feature_axis=2)
    torch.testing.assert_close(accuracy.item(), 1.0)

    # change targets
    accuracy = categorical_accuracy(prediction, target % 3, data_index=0, feature_axis=2)
    torch.testing.assert_close(accuracy.item(), 0.6)


def test_metric_l1():
    batch_size = 5

    prediction = torch.randn((batch_size, 1))
    target = prediction

    result = l1(prediction, target, data_index=0, feature_axis=-1)
    torch.testing.assert_close(result.item(), 0.0)

    result = l1(prediction+.1, target, data_index=0, feature_axis=-1)
    torch.testing.assert_close(result.item(), 0.1)

    result = l1(prediction+.1, target, data_index=0, feature_axis=-1)
    torch.testing.assert_close(result.item(), 0.1)


def test_metric_l1_class():
    batch_size = 5

    prediction = torch.randn((batch_size, 1))
    target = prediction

    metric = PolyMetric(data_index=0, function=F.l1_loss)

    result = metric(prediction, target)
    torch.testing.assert_close(result.item(), 0.0)

    result = metric(prediction-0.1, target)
    torch.testing.assert_close(result.item(), 0.1)

    result = metric(prediction+0.1, target)
    torch.testing.assert_close(result.item(), 0.1)

    assert metric.name == metric.__name__ == "l1_loss"


def test_metric_mse():
    batch_size = 5

    prediction = torch.randn((batch_size, 1))
    target = prediction

    result = mse(prediction, target, data_index=0, feature_axis=-1)
    torch.testing.assert_close(result.item(), 0.0)

    result = mse(prediction+.1, target, data_index=0, feature_axis=-1)
    torch.testing.assert_close(result.item(), 0.01)

    result = mse(prediction+.1, target, data_index=0, feature_axis=-1)
    torch.testing.assert_close(result.item(), 0.01)


def test_metric_smooth_l1():
    batch_size = 5

    prediction = torch.randn((batch_size, 1))
    target = prediction

    result = smooth_l1(prediction, target, data_index=0, feature_axis=-1)
    torch.testing.assert_close(result.item(), 0.0)

    result = smooth_l1(prediction+.1, target, data_index=0, feature_axis=-1)
    torch.testing.assert_close(result.item(), 0.005)

    result = smooth_l1(prediction+.1, target, data_index=0, feature_axis=-1)
    torch.testing.assert_close(result.item(), 0.005)


def test_metric_smooth_l1_squeeze():
    batch_size = 5

    prediction = torch.randn((batch_size, 1))
    target = prediction.squeeze()

    result = smooth_l1(prediction, target, data_index=0, feature_axis=-1)
    torch.testing.assert_close(result.item(), 0.0)

    result = smooth_l1(prediction+.1, target, data_index=0, feature_axis=-1)
    torch.testing.assert_close(result.item(), 0.005)

    result = smooth_l1(prediction+.1, target, data_index=0, feature_axis=-1)
    torch.testing.assert_close(result.item(), 0.005)


def test_calc_generalized_dice_score():
    batch_size = 5
    n_classes = 9
    width = 64
    height = 128

    target = torch.randint(n_classes, (batch_size, height, width))
    prediction = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()

    result = calc_generalized_dice_score(prediction, target, feature_axis=1)
    torch.testing.assert_close(result.item(), 1.0)


def test_generalized_dice():
    batch_size = 5
    n_classes = 9
    width = 64
    height = 128

    target = torch.randint(n_classes, (batch_size, height, width))
    prediction = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    prediction *= 100.0 # convert to logits

    result = generalized_dice(prediction, target, data_index=0, feature_axis=1)
    torch.testing.assert_close(result.item(), 1.0)


def test_poly_metric_no_function():
    metric = PolyMetric(name="test", data_index=0)
    with pytest.raises(NotImplementedError):
        metric(torch.randn(1, 1), torch.randn(1, 1))


def test_hierarchical_greedy_accuracy():
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
    # Test accurate
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.index_in_softmax_layer ] = 20.0
            target = target.parent

    # switch one prediction to be wrong at level 2
    predictions[ -1, : ] = 0.0
    predictions[ -1, bb.index_in_softmax_layer ] = 20.0

    greedy = HierarchicalGreedyAccuracy(data_index=0, root=root)
    value = greedy(predictions, target_tensor)
    assert value == 0.75
    assert greedy.name == greedy.__name__ == "HierarchicalGreedyAccuracy"

    greedy_depth_one = HierarchicalGreedyAccuracy(data_index=0, root=root, max_depth=1, name="greedy_depth_one")
    value = greedy_depth_one(predictions, target_tensor)
    assert value == 1.0
    assert greedy_depth_one.name == greedy_depth_one.__name__ == "greedy_depth_one"
