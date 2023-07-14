import torch
import torch.nn.functional as F

from .util import squeeze_prediction


def get_predictions_target_for_index(predictions, *targets, data_index=None, feature_axis=-1):
    if not isinstance(predictions, (tuple, list)):
        predictions = (predictions,)

    assert len(predictions) == len(targets)
    assert data_index is not None, "data_index must be specified"

    my_predictions = predictions[data_index]
    my_targets = targets[data_index]

    return my_predictions, my_targets


def function_metric(predictions, *targets, data_index=None, feature_axis=-1, function=None) -> torch.Tensor:
    my_predictions, my_targets = get_predictions_target_for_index(predictions, *targets, data_index=data_index, feature_axis=feature_axis)
    my_predictions = squeeze_prediction(my_predictions, my_targets, feature_axis)
    return function(my_predictions, my_targets)


def categorical_accuracy(predictions, *targets, data_index=None, feature_axis=-1) -> torch.Tensor:
    my_predictions, my_targets = get_predictions_target_for_index(predictions, *targets, data_index=data_index, feature_axis=feature_axis)
    my_predictions = torch.argmax(my_predictions, dim=feature_axis)

    accuracy = (my_predictions == my_targets).float().mean()
    return accuracy


def binary_accuracy(predictions, *targets, data_index=None, feature_axis=-1) -> torch.Tensor:
    my_predictions, my_targets = get_predictions_target_for_index(predictions, *targets, data_index=data_index, feature_axis=feature_axis)
    my_predictions = my_predictions >= 0.0
    my_predictions = squeeze_prediction(my_predictions, my_targets, feature_axis)

    accuracy = (my_predictions == my_targets).float().mean()
    return accuracy


def mse(predictions, *targets, data_index=None, feature_axis=-1) -> torch.Tensor:
    return function_metric(predictions, *targets, data_index=data_index, feature_axis=feature_axis, function=F.mse_loss)


def l1(predictions, *targets, data_index=None, feature_axis=-1) -> torch.Tensor:
    return function_metric(predictions, *targets, data_index=data_index, feature_axis=feature_axis, function=F.l1_loss)


def smooth_l1(predictions, *targets, data_index=None, feature_axis=-1) -> torch.Tensor:
    return function_metric(predictions, *targets, data_index=data_index, feature_axis=feature_axis, function=F.smooth_l1_loss)


def calc_dice_score(predictions, target, smooth:float=1.) -> torch.Tensor:
    predictions = predictions.view(-1)
    target = target.view(-1)
    intersection = (predictions * target).sum()
    
    return ((2. * intersection + smooth) /
              (predictions.sum() + target.sum() + smooth)
    )


def calc_generalized_dice_score(predictions, target, power:float=2.0, smooth:float=1.0, feature_axis:int=-1) -> torch.Tensor:
    """
    A generalized Dice score for multi-class segmentation.

    If power=0.0, this is equivalent to normal Dice score (i.e. volume "implicit" weighting)
    If power=1.0, this is equivalent to 'equal' weighting.
    If power=2.0, this is equivalent to 'inverse volume' weighting.

    See:
        - https://www.sciencedirect.com/science/article/pii/S2590005619300049#bib73
        - https://arxiv.org/pdf/1707.03237.pdf
        - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1717643
    """
    n_classes = predictions.shape[feature_axis]
    numerator = 0.0
    denominator = 0.0
    slice_indices = [slice(0, None)] * len(predictions.shape)
    for i in range(n_classes):
        my_target = (target == i)
        my_target_sum = my_target.sum()
        weight = 1/(my_target_sum**power + smooth)

        slice_indices[feature_axis] = i
        my_predictions = predictions[slice_indices]

        numerator += weight * (my_predictions*my_target).sum()
        denominator += weight * (my_predictions.sum() + my_target_sum)
    
    return 2. * numerator / denominator


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


def generalized_dice(predictions, *targets, data_index=None, feature_axis=-1, perform_softmax:bool=True, power:float=2.0) -> torch.Tensor:
    """
    Calculate the generalized dice score for a single data index. Used for for multi-class segmentation.


    See:
        - https://www.sciencedirect.com/science/article/pii/S2590005619300049#bib73
        - https://arxiv.org/pdf/1707.03237.pdf
        - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1717643

    Args:
        predictions (_type_): Logits or probabilities. If probabilities, perform_softmax must be False.
        data_index (_type_, optional): The index of the target to . Defaults to None.
        feature_axis (int, optional): _description_. Defaults to -1.
        perform_softmax (bool, optional): Whether or not to normalize the predictions using the softmax function. Defaults to True.
        power (float, optional): The power to use for the generalized dice score. Defaults to 2.0 (i.e. inverse volume weighting)
            If power=0.0, this is equivalent to normal Dice score (i.e. volume "implicit" weighting)
            If power=1.0, this is equivalent to 'equal' weighting.
            If power=2.0, this is equivalent to 'inverse volume' weighting.

    Returns:
        torch.Tensor: The generalized dice score for the given data index.
    """
    my_predictions, my_targets = get_predictions_target_for_index(predictions, *targets, data_index=data_index, feature_axis=feature_axis)
    if perform_softmax:
        my_predictions = my_predictions.softmax(dim=feature_axis)
    
    score = calc_generalized_dice_score(
        my_predictions, 
        my_targets, 
        feature_axis=feature_axis,
        power=power,
    )
    return score


