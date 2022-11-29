"""Autoencoder Implementation using Skorch."""
from typing import List

import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet

from drcomp import DimensionalityReducer


class AutoEncoder(NeuralNet, DimensionalityReducer):
    """Autoencoder Implementation using Skorch."""

    def __init__(
        self,
        input_size: int,
        intrinsic_dim: int,
        hidden_layer_sizes: List[int] = [],
        act_fn: object = nn.ReLU,
        lr: float = 1e-3,
        max_epochs: int = 10,
        **kwargs
    ):
        """Initialize the autoencoder."""
        self.intrinsic_dim = intrinsic_dim
        layer_sizes = [input_size, *hidden_layer_sizes, intrinsic_dim]

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                encoder = nn.ModuleList()
                decoder = nn.ModuleList()
                depth = len(layer_sizes) - 1
                for i in range(depth):
                    encoder.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                    encoder.append(act_fn())

                    decoder.append(
                        nn.Linear(layer_sizes[depth - i], layer_sizes[depth - i - 1])
                    )
                    decoder.append(act_fn())
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):
                # Encode
                for layer in self.encoder:
                    x = layer(x)
                encoded = x
                # Decode
                for layer in self.decoder:
                    x = layer(x)
                decoded = x
                return decoded, encoded

        # skorch neural net provides a wrapper around pytorch, which includes training loop etc.
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
        return encoded.detach().numpy()
