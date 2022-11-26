"""Autoencoder Implementation using Skorch."""
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet

from drcomp import DimensionalityReducer


class Encoder(nn.Module):
    """Simple Fully Connected Encoder."""

    def __init__(self, input_size, hidden_size):
        """Initialize the Encoder."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size), nn.ReLU()
        )

    def forward(self, x):
        """Forward pass."""
        return self.encoder(x)


class Decoder(nn.Module):
    """Simple Fully Connected Decoder."""

    def __init__(self, hidden_size, output_size):
        """Initialize the Decoder."""
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size), nn.ReLU()
        )

    def forward(self, x):
        """Forward pass."""
        return self.decoder(x)


class Autoencoder(NeuralNet, DimensionalityReducer):
    """Autoencoder Implementation using Skorch."""

    def __init__(
        self,
        encoder: nn.Module = Encoder,
        decoder: nn.Module = Decoder,
        lr: float = 1e-3,
        max_epochs: int = 10,
        **kwargs
    ):
        """Initialize the autoencoder."""
        net = nn.Sequential(encoder, decoder)
        super(self).__init__(
            net,
            criterion=nn.MSELoss,
            optimizer=optim.Adam,
            lr=lr,
            max_epochs=max_epochs,
            **kwargs
        )
