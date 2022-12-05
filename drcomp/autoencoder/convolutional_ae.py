import torch.nn as nn

from drcomp.autoencoder.base import AbstractAutoEncoder


class MnistConvAE(AbstractAutoEncoder):
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


class Cifar10ConvAE(AbstractAutoEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # b, 16, 16, 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # b, 32, 8, 8
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # b, 64, 4, 4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # b, 128, 2, 2
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(
                128, 64, 3, stride=2, padding=1, output_padding=1
            ),  # b, 64, 4, 4
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),  # b, 32, 8, 8
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  # b, 16, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 3, 3, stride=2, padding=1, output_padding=1
            ),  # b, 3, 32, 32
        )
