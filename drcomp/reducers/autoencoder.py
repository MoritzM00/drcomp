"""Autoencoder Implementation using Skorch."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet
from torch import from_numpy
from torch.autograd.functional import jacobian

from drcomp import DimensionalityReducer
from drcomp.autoencoder.base import AbstractAutoEncoder


class AutoEncoder(NeuralNet, DimensionalityReducer):
    """Autoencoder Implementation using Skorch to adhere to the DimensionalityReducer interface.

    Parameters
    ----------
    AutoEncoderClass : AbstractAutoEncoder
        The autoencoder class to use. Must implement the AbstractAutoEncoder, i.e. it must
        define an encoder and decoder. It can be the actual instance (already instantiated) or the class itself.
        If the class itself is passed, you must pass the required parameters to instantiate it to this class via `module__<name>=<value>`.
    criterion : torch criterion (class)
        The uninitialized criterion (loss) used to optimize the
        module.
    optimizer : torch optim (class, default=torch.optim.SGD)
        The uninitialized optimizer (update rule) used to optimize the
        module.
    lr : float (default=0.01)
        Learning rate passed to the optimizer. You may use ``lr`` instead
        of using ``optimizer__lr``, which would result in the same outcome.
    max_epochs : int (default=10)
        The number of epochs to train for each ``fit`` call. Note that you
        may keyboard-interrupt training at any time.
    batch_size : int (default=128)
        Mini-batch size. Use this instead of setting
        ``iterator_train__batch_size`` and ``iterator_test__batch_size``,
        which would result in the same outcome. If ``batch_size`` is -1,
        a single batch with all the data will be used during training
        and validation.
    callbacks : None, "disable", or list of Callback instances (default=None)
        Which callbacks to enable. There are three possible values:
        If ``callbacks=None``, only use default callbacks,
        those returned by ``get_default_callbacks``.
        If ``callbacks="disable"``, disable all callbacks, i.e. do not run
        any of the callbacks, not even the default callbacks.
        If ``callbacks`` is a list of callbacks, use those callbacks in
        addition to the default callbacks. Each callback should be an
        instance of :class:`.Callback`.
    device : str, torch.device, or None (default='cpu')
        The compute device to be used. If set to 'cuda' in order to use
        GPU acceleration, data in torch tensors will be pushed to cuda
        tensors before being sent to the module. If set to None, then
        all compute devices will be left unmodified.
    **kwargs
        Additional keyword arguments that are passed to skorch.NeuralNet.
    """

    def __init__(
        self,
        AutoEncoderClass: AbstractAutoEncoder,
        criterion=nn.MSELoss,
        batch_size: int = 128,
        max_epochs: int = 10,
        lr: float = 1e-3,
        optimizer=optim.Adam,
        device="cpu",
        callbacks=None,
        contractive: bool = False,
        contractive_lambda: float = 1e-4,
        **kwargs
    ):
        """Initialize the autoencoder."""
        self.supports_inverse_transform = True
        self.contractive = contractive
        self.contractive_lambda = contractive_lambda
        # skorch neural net provides a wrapper around pytorch, which includes training loop etc.
        super().__init__(
            AutoEncoderClass,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            callbacks=callbacks,
            **kwargs
        )

    def fit(self, X, y=None):
        super().fit(X, X)
        return self

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        decoded, _ = y_pred
        reconstr_loss = super().get_loss(decoded, y_true, *args, **kwargs)
        if self.contractive:
            contr_loss = torch.norm(
                jacobian(self.module_.encoder, y_pred[0], create_graph=True)
            )
        else:
            contr_loss = 0
        return reconstr_loss + self.contractive_lambda * contr_loss

    def transform(self, X) -> np.ndarray:
        _, encoded = self.forward(X, training=False)
        return encoded.detach().numpy()

    def inverse_transform(self, X) -> np.ndarray:
        X = from_numpy(X)
        decoded = self.module_.decoder(X)
        return decoded.detach().numpy()
