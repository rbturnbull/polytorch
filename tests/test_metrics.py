import torch
from torch.nn import functional as F
from polytorch.metrics import (
    categorical_accuracy, 
    mse, l1, smooth_l1, 
    binary_accuracy, binary_dice, binary_iou, 
    calc_generalized_dice_score, generalized_dice,
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