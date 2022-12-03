"""This Module provides Classes for the Autoencoders Reducer class."""

from .convolutional_ae import MnistConvolutionalAE
from .fully_connected_ae import FullyConnectedAutoencoder

__all__ = ["FullyConnectedAutoencoder", "MnistConvolutionalAE"]
