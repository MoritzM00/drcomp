"""This Module provides Classes for the Autoencoders Reducer class."""

from .base import AbstractAutoEncoder
from .contractive_loss import ContractiveLoss
from .convolutional_ae import Cifar10ConvAE, MnistConvAE
from .fully_connected_ae import FullyConnectedAE

__all__ = [
    "AbstractAutoEncoder",
    "ContractiveLoss",
    "FullyConnectedAE",
    "MnistConvAE",
    "Cifar10ConvAE",
]
