"""Fully Connected Autoencoder"""
import torch.nn as nn

from drcomp.autoencoder.base import AbstractAutoEncoder


class FullyConnectedAE(AbstractAutoEncoder):
    def __init__(
        self,
        input_size: int,
        intrinsic_dim: int,
        hidden_layer_dims: list[int] = [],
        act_fn: object = nn.Sigmoid,
        include_batch_norm: bool = False,
    ):
        super().__init__(intrinsic_dim=intrinsic_dim)
        self.input_size = input_size
        self.hidden_layer_dims = hidden_layer_dims
        self.act_fn = act_fn
        self.include_batch_norm = include_batch_norm

        # build encoder and decoder
        layer_dims = [input_size, *hidden_layer_dims, intrinsic_dim]
        encoder = nn.ModuleList()
        decoder = nn.ModuleList()
        depth = len(layer_dims) - 1
        for i in range(depth):
            encoder.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if include_batch_norm:
                encoder.append(nn.BatchNorm1d(layer_dims[i + 1]))
            encoder.append(act_fn())

            decoder.append(nn.Linear(layer_dims[depth - i], layer_dims[depth - i - 1]))
            if self.include_batch_norm:
                decoder.append(nn.BatchNorm1d(layer_dims[depth - i - 1]))
            decoder.append(act_fn())
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
