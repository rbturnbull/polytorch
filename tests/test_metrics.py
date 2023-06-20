import torch

from polytorch.metrics import categorical_accuracy

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


