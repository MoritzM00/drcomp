"""Fully Connected Autoencoder"""
import logging
from typing import Union

import torch
import torch.nn as nn

from drcomp.autoencoder.base import AbstractAutoEncoder

logger = logging.getLogger(__name__)


class FullyConnectedAE(AbstractAutoEncoder):
    """Fully Connected Autoencoder.

    Parameters
    ----------
    input_size : int
        Input dimension of the data.
    intrinsic_dim : int
        The intrinsic dimensionality determines the size of the bottleneck layer.
    hidden_layer_dims : list[int], default=[]
        List of hidden layer dimensions of the Encoder, by default []. Then a shallow autoencoder with one hidden layer is created.
    act_fn : Union[object, list[object]], default=nn.Sigmoid
        Activation function for the hidden layers, by default nn.Sigmoid. If a list is provided,
        then each element of the list is used for the corresponding hidden layer.
    include_batch_norm : bool, default=False
        Whether to include batch normalization, by default False.
    """

    def __init__(
        self,
        input_size: int,
        intrinsic_dim: int,
        hidden_layer_dims: list[int] = [],
        act_fn: Union[object, list[object]] = nn.Sigmoid,
        include_batch_norm: bool = False,
        tied_weights: bool = False,
    ):
        super().__init__(intrinsic_dim=intrinsic_dim)
        self.input_size = input_size
        self.hidden_layer_dims = hidden_layer_dims
        self.act_fn = act_fn
        self.include_batch_norm = include_batch_norm
        if not isinstance(act_fn, list):
            act_fn = [act_fn] * (len(hidden_layer_dims) + 1)
        assert (
            len(act_fn) == len(hidden_layer_dims) + 1
        ), "act_fn must be a list of length len(hidden_layer_dims) + 1"

        # build encoder and decoder
        layer_dims = [input_size, *hidden_layer_dims, intrinsic_dim]
        logger.debug(f"The encoder layer dimensions are {layer_dims}")
        encoder = nn.ModuleList()
        decoder = nn.ModuleList()
        depth = len(layer_dims) - 1  # this is the depth of the encoder
        logger.debug(f"Depth of the encoder is {depth}")
        for i in range(depth):
            encoder_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            encoder.append(encoder_layer)
            if include_batch_norm:
                encoder.append(nn.BatchNorm1d(layer_dims[i + 1]))
            encoder.append(act_fn[i]())

            decoder_layer = nn.Linear(layer_dims[depth - i], layer_dims[depth - i - 1])
            if tied_weights:
                # tie the weights by setting the decoder layer weight to be the transpose of the encoder layer weight.
                logger.debug(
                    f"Tying the weights of the encoder layer {i} and decoder layer {depth - i}."
                )
                layer_weight = torch.empty_like(encoder_layer.weight)
                encoder_layer.weight = nn.Parameter(layer_weight)
                decoder_layer.weight = nn.Parameter(layer_weight.t())
            decoder.append(decoder_layer)
            if self.include_batch_norm:
                decoder.append(nn.BatchNorm1d(layer_dims[depth - i - 1]))
            decoder.append(act_fn[depth - i - 1]())

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
