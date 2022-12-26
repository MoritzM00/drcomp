"""This Module provides Classes for the Autoencoders Reducer class."""

from .base import AbstractAutoEncoder
from .convolutional_ae import (
    Cifar10ConvAE,
    Fer2013ConvAE,
    LfwPeopleConvAE,
    MnistConvAE,
    OlivettiFacesConvAE,
)
from .fully_connected_ae import FullyConnectedAE

__all__ = [
    "AbstractAutoEncoder",
    "FullyConnectedAE",
    "MnistConvAE",
    "Cifar10ConvAE",
    "Fer2013ConvAE",
    "LfwPeopleConvAE",
    "OlivettiFacesConvAE",
]
