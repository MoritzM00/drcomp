import torch.nn as nn

from drcomp.autoencoder.base import AbstractAutoEncoder


class MnistConvolutionalAE(AbstractAutoEncoder):
    def __init__(self, intrinsic_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28x28x1 -> 14x14x16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 14x14s16 -> 7x7x32
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, intrinsic_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(intrinsic_dim, 7 * 7 * 32),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32, 7, 7)),
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  # 7x7x32 -> 14x14x16
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, 3, stride=2, padding=1, output_padding=1
            ),  # 14x14x16 -> 28x28x1
            nn.ReLU(),
        )
