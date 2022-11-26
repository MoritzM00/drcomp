"""Autoencoder Implementation using Skorch."""
import torch.nn as nn
from skorch import NeuralNet


class Encoder(nn.Module):
    pass


class Decoder(nn.Module):
    pass


class Autoencoder(NeuralNet):
    """Autoencoder Implementation using Skorch."""

    def __init__(self, encoder: nn.Module = Encoder, decoder: nn.Module = Decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, x):
        y = self.encoder(x)
        x_hat = self.decoder(y)
        return x_hat
