import torch.nn as nn

class NMRNetwork_Base(nn.Module):
    def __init__(self, input_size=2048, output_size=128, initial_hidden_size=512, n_layers=3):
        super(NMRNetwork_Base, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, initial_hidden_size))
        layers.append(nn.ReLU())

        # Dynamically add hidden layers, halving the size each time
        hidden_size = initial_hidden_size
        for _ in range(n_layers - 1):
            next_hidden_size = hidden_size // 2  # Halving the hidden layer size
            layers.append(nn.Linear(hidden_size, next_hidden_size))
            layers.append(nn.ReLU())
            hidden_size = next_hidden_size

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        # Combine all layers
        self.model = nn.Sequential(*layers)

    def forward(self, input, target):
        out = self.model(input)
        loss = nn.MSELoss()(out, target)
        return loss, out

