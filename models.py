import torch.nn as nn
import torch

class MLP_Baseline(nn.Module):
    """ A simple Multi-Layer Perceptron (MLP) model for classification tasks.
    """
    def __init__(self, input_dim, output_dim):
        super(MLP_Baseline, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class MLP_Advanced(nn.Module):
    """ A more complex MLP model for classification tasks.
    """
    def __init__(self, input_dim, output_dim):
        super(MLP_Advanced, self).__init__()

        ### <--- START OF YOUR CODE

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

        ### END OF YOUR CODE --->

    def forward(self, x):
        return self.network(x)

def loss_Baseline(device):
    """ Returns the loss function for the baseline MLP model.
    """
    loss = nn.CrossEntropyLoss()
    return loss

def loss_Advanced(device):
    """ Returns a more advanced loss function.
    """
    
    ### <--- START OF YOUR CODE

    weights = torch.tensor([1.0, 4.0, 4.0, 1.0]).to(device)
    loss = nn.CrossEntropyLoss(weight=weights)

    ### END OF YOUR CODE --->

    return loss