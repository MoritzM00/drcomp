from abc import ABCMeta, abstractmethod

import torch.nn as nn


class AbstractAutoEncoder(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.encoder: nn.Sequential = None
        self.decoder: nn.Sequential = None

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
