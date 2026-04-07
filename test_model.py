import models
import torch

def test_MLP_Advanced():
    input_dim = 32 # MFCC Features Extracted
    output_dim = 4 # Number of classes
    
    model = models.MLP_Advanced(input_dim, output_dim)
    
    # Check that the model has the correct structure
    layers = list(list(model.children())[0])
    
    # Should have: Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear (7 layers total)
    assert len(layers) == 7, f"Expected 7 layers, got {len(layers)}"
    
    # Check first hidden layer: 64 neurons
    assert layers[0].out_features == 64, f"First layer should output 64 neurons, got {layers[0].out_features}"
    
    # Check dropout after first hidden layer
    assert isinstance(layers[1], torch.nn.ReLU)
    assert isinstance(layers[2], torch.nn.Dropout)
    assert layers[2].p == 0.2, f"First dropout rate should be 0.2, got {layers[2].p}"
    
    # Check second hidden layer: 64 neurons
    assert layers[3].out_features == 64, f"Second layer should output 64 neurons, got {layers[3].out_features}"
    
    # Check dropout after second hidden layer
    assert isinstance(layers[4], torch.nn.ReLU)
    assert isinstance(layers[5], torch.nn.Dropout)
    assert layers[5].p == 0.2, f"Second dropout rate should be 0.2, got {layers[5].p}"
    
    # Check output layer
    assert isinstance(layers[6], torch.nn.Linear)
