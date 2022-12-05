"""Fully Connected Autoencoder"""
import torch.nn as nn

from drcomp.autoencoder.base import AbstractAutoEncoder


class FullyConnectedAE(AbstractAutoEncoder):
    def __init__(
        self,
        input_size: int,
        intrinsic_dim: int,
        hidden_layer_dims: list[int] = [],
        act_fn: object = nn.ReLU,
    ):
        super().__init__()
        self.intrinsic_dim = intrinsic_dim
        layer_sizes = [input_size, *hidden_layer_dims, intrinsic_dim]

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
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
