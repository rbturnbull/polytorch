import torch

def categorical_accuracy(predictions, *targets, data_index=None, feature_axis=-1):
    if not isinstance(predictions, (tuple, list)):
        predictions = (predictions,)

    assert len(predictions) == len(targets)
    assert data_index is not None, "data_index must be specified in categorical_accuracy"

    my_predictions = torch.argmax(predictions[data_index], dim=feature_axis)
    my_targets = targets[data_index]

    accuracy = (my_predictions == my_targets).float().mean()
    return accuracy


