import torch
import torch.nn.functional as F

from .util import squeeze_prediction


def get_predictions_target_for_index(predictions, *targets, data_index=None, feature_axis=-1):
    if not isinstance(predictions, (tuple, list)):
        predictions = (predictions,)

    assert len(predictions) == len(targets)
    assert data_index is not None, "data_index must be specified in categorical_accuracy"

    my_predictions = predictions[data_index]
    my_targets = targets[data_index]

    return my_predictions, my_targets


def function_metric(predictions, *targets, data_index=None, feature_axis=-1, function=None):
    my_predictions, my_targets = get_predictions_target_for_index(predictions, *targets, data_index=data_index, feature_axis=feature_axis)
    my_predictions = squeeze_prediction(my_predictions, my_targets, feature_axis)
    return function(my_predictions, my_targets)


def categorical_accuracy(predictions, *targets, data_index=None, feature_axis=-1):
    my_predictions, my_targets = get_predictions_target_for_index(predictions, *targets, data_index=data_index, feature_axis=feature_axis)
    my_predictions = torch.argmax(my_predictions, dim=feature_axis)

    accuracy = (my_predictions == my_targets).float().mean()
    return accuracy


def binary_accuracy(predictions, *targets, data_index=None, feature_axis=-1):
    my_predictions, my_targets = get_predictions_target_for_index(predictions, *targets, data_index=data_index, feature_axis=feature_axis)
    my_predictions = my_predictions >= 0.0
    my_predictions = squeeze_prediction(my_predictions, my_targets, feature_axis)

    accuracy = (my_predictions == my_targets).float().mean()
    return accuracy


def mse(predictions, *targets, data_index=None, feature_axis=-1):
    return function_metric(predictions, *targets, data_index=data_index, feature_axis=feature_axis, function=F.mse_loss)


def l1(predictions, *targets, data_index=None, feature_axis=-1):
    return function_metric(predictions, *targets, data_index=data_index, feature_axis=feature_axis, function=F.l1_loss)


def smooth_l1(predictions, *targets, data_index=None, feature_axis=-1):
    return function_metric(predictions, *targets, data_index=data_index, feature_axis=feature_axis, function=F.smooth_l1_loss)


def calc_dice_score(predictions, target, smooth:float=1.):
    predictions = predictions.view(-1)
    target = target.view(-1)
    intersection = (predictions * target).sum()
    
    return ((2. * intersection + smooth) /
              (predictions.sum() + target.sum() + smooth)
    )


def calc_iou(predictions, target, smooth:float=1.):
    predictions = predictions.view(-1)
    target = target.view(-1)
    intersection = (predictions * target).sum()
    union = predictions.sum() + target.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)


def binary_dice(predictions, *targets, data_index=None, feature_axis=-1):
    my_predictions, my_targets = get_predictions_target_for_index(predictions, *targets, data_index=data_index, feature_axis=feature_axis)
    my_predictions = my_predictions >= 0.0
    my_predictions = squeeze_prediction(my_predictions, my_targets, feature_axis)

    return calc_dice_score(my_predictions, my_targets)


def binary_iou(predictions, *targets, data_index=None, feature_axis=-1):
    my_predictions, my_targets = get_predictions_target_for_index(predictions, *targets, data_index=data_index, feature_axis=feature_axis)
    my_predictions = my_predictions >= 0.0
    my_predictions = squeeze_prediction(my_predictions, my_targets, feature_axis)

    return calc_iou(my_predictions, my_targets)
