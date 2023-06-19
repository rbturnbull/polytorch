import torch

from polytorch import PolyLoss, CategoricalData

def test_loss_categorical():
    batch_size = 5
    category_count = batch_size
    
    prediction = torch.diag(torch.full((category_count,), 10.0))
    target = torch.arange(category_count)

    loss_fn = PolyLoss([CategoricalData(category_count)])
    loss = loss_fn(prediction, target)
    assert loss.item() < 0.01