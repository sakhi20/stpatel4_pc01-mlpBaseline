import models
import torch

def test_loss_Advanced():
    device = torch.device('cpu')
    loss_fn = models.loss_Advanced(device)
    
    # Check that the loss function is an instance of CrossEntropyLoss
    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss), "Expected loss function to be an instance of CrossEntropyLoss"
    
    # Check that the class weights are set correctly
    expected_weights = torch.tensor([1.0, 4.0, 4.0, 1.0], dtype=torch.float)
    assert torch.allclose(loss_fn.weight.cpu(), expected_weights), f"Expected class weights {1.0,4.0,4.0,1.0}, got {loss_fn.weight.cpu().tolist()}"