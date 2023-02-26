from abc import ABCMeta, abstractmethod

import torch.nn as nn


class AbstractAutoEncoder(nn.Module, metaclass=ABCMeta):
    """
    Base Class for all PyTorch AutoEncoder Modules.

    Defines the forward pass and the following instance attributes:
    - intrinsic_dim: the dimensionality of the latent space
    - encoder: the encoder network (nn.Sequential)
    - decoder: the decoder network (nn.Sequential)

    The forward pass returns a tuple of the decoded and encoded tensors.
    """

    @abstractmethod
    def __init__(self, intrinsic_dim: int = 2):
        super().__init__()
        self.intrinsic_dim = intrinsic_dim
        self.encoder: nn.Sequential = None
        self.decoder: nn.Sequential = None

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
