"""Autoencoder Implementation using Skorch."""

import numpy as np
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet
from torch import from_numpy

from drcomp import DimensionalityReducer
from drcomp.autoencoder.base import AbstractAutoEncoder


class AutoEncoder(NeuralNet, DimensionalityReducer):
    """Autoencoder Implementation using Skorch to adhere to the DimensionalityReducer interface."""

    def __init__(
        self,
        AutoEncoderClass: AbstractAutoEncoder,
        lr: float = 1e-3,
        max_epochs: int = 10,
        criterion=nn.MSELoss,
        **kwargs
    ):
        """Initialize the autoencoder."""
        self.supports_inverse_transform = True
        # skorch neural net provides a wrapper around pytorch, which includes training loop etc.
        super().__init__(
            AutoEncoderClass,
            criterion=criterion,
            optimizer=optim.Adam,
            lr=lr,
            max_epochs=max_epochs,
            **kwargs
        )

    def fit(self, X, y=None):
        super().fit(X, X)
        return self

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        decoded, _ = y_pred
        reconstr_loss = super().get_loss(decoded, y_true, *args, **kwargs)
        return reconstr_loss

    def transform(self, X) -> np.ndarray:
        _, encoded = self.forward(X, training=False)
        return encoded.detach().numpy()

    def inverse_transform(self, X) -> np.ndarray:
        X = from_numpy(X)
        decoded = self.module_.decoder(X)
        return decoded.detach().numpy()
