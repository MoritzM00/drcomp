"""Autoencoder Implementation using Skorch."""
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet

from drcomp import DimensionalityReducer


class SimpleEncoder(nn.Module):
    """Simple Fully Connected Encoder."""

    def __init__(self, input_size, intrinsic_dim):
        """Initialize the Encoder."""
        super().__init__()
        self.input_size = input_size
        self.intrinsic_dim = intrinsic_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.intrinsic_dim), nn.ReLU()
        )

    def forward(self, x):
        """Forward pass."""
        return self.encoder(x)


class SimpleDecoder(nn.Module):
    """Simple Fully Connected Decoder."""

    def __init__(self, intrinsic_dim, output_size):
        """Initialize the Decoder."""
        super().__init__()
        self.intrinsic_dim = intrinsic_dim
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.Linear(self.intrinsic_dim, self.output_size), nn.ReLU()
        )

    def forward(self, x):
        """Forward pass."""
        return self.decoder(x)


class AutoEncoder(NeuralNet, DimensionalityReducer):
    """Autoencoder Implementation using Skorch."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float = 1e-3,
        max_epochs: int = 10,
        **kwargs
    ):
        """Initialize the autoencoder."""
        self.encoder = encoder
        self.decoder = decoder

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded, encoded

        super().__init__(
            Net,
            criterion=nn.MSELoss,
            optimizer=optim.Adam,
            lr=lr,
            max_epochs=max_epochs,
            **kwargs
        )

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        decoded, _ = y_pred
        reconstr_loss = super().get_loss(decoded, y_true, *args, **kwargs)
        return reconstr_loss

    def transform(self, X):
        _, encoded = self.forward(X)
        return encoded
