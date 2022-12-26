"""Autoencoder Implementation using Skorch."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet
from torch.autograd.functional import jacobian

from drcomp import DimensionalityReducer
from drcomp.autoencoder import AbstractAutoEncoder


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
    contractive : bool (default=False)
        Whether to use contractive autoencoder loss. Be aware that this is computationally expensive. Even on
        small fully connected models with two hidden layers on MNIST Dataset, this can be prohibitive.
    contractive_lambda : float (default=1e-4)
        The lambda parameter for the contractive loss. Only used if contractive is True.
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
        DimensionalityReducer.__init__(
            self,
            intrinsic_dim=AutoEncoderClass.intrinsic_dim,
            supports_inverse_transform=True,
        )
        self.contractive = contractive
        self.contractive_lambda = contractive_lambda
        if self.contractive:
            device = "cpu"  # contractive loss is not supported on GPU
        # skorch neural net provides a wrapper around pytorch, which includes training loop etc.
        NeuralNet.__init__(
            self,
            AutoEncoderClass,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            callbacks=callbacks,
        )

    def fit(self, X, y=None):
        super().fit(X, X)
        self.fitted_ = True
        return self

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        decoded, _ = y_pred
        reconstr_loss = super().get_loss(decoded, y_true, *args, **kwargs)
        if self.contractive:
            contr_loss = torch.norm(
                jacobian(self.module_.encoder, y_true, create_graph=True)
            )
        else:
            contr_loss = 0
        return reconstr_loss + self.contractive_lambda * contr_loss

    def transform(self, X) -> np.ndarray:
        _, encoded = self.forward(X, training=False)
        return encoded.detach().cpu().numpy()

    def inverse_transform(self, Y) -> np.ndarray:
        Y = torch.from_numpy(Y).to(self.device)
        decoded = self.module_.decoder(Y)
        return decoded.detach().cpu().numpy()
