from abc import ABCMeta, abstractmethod

import torch.nn as nn


class AbstractAutoEncoder(nn.Module, metaclass=ABCMeta):
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
