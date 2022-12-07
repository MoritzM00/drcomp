import torch.nn as nn

from drcomp.autoencoder.base import AbstractAutoEncoder


class MnistConvAE(AbstractAutoEncoder):
    def __init__(self, intrinsic_dim: int = 16):
        super().__init__(intrinsic_dim=intrinsic_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, 3, stride=2, padding=1),  # 1x28x28 -> 128x14x14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 128x14x14 -> 256x7x7
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 256x7x7 -> 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),  # 512x4x4 -> 1024x2x2
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024 * 2 * 2, intrinsic_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(intrinsic_dim, 1024 * 2 * 2),
            nn.BatchNorm1d(1024 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(1024, 2, 2)),
            nn.ConvTranspose2d(
                1024, 512, 3, stride=2, padding=1, output_padding=1
            ),  # 1024x2x2 -> 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(
                512, 256, 3, stride=2, padding=1, output_padding=1
            ),  # 512x4x4 -> 256x7x7
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 3, stride=2, padding=1, output_padding=1),
        )


class Cifar10ConvAE(AbstractAutoEncoder):
    def __init__(self, intrinsic_dim: int = 64):
        super().__init__(intrinsic_dim=intrinsic_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # b, 16, 16, 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # b, 32, 8, 8
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # b, 64, 4, 4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # b, 128, 2, 2
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, intrinsic_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(intrinsic_dim, 128 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(
                128, 64, 3, stride=2, padding=1, output_padding=1
            ),  # b, 64, 4, 4
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),  # b, 32, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  # b, 16, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 3, 3, stride=2, padding=1, output_padding=1
            ),  # b, 3, 32, 32
        )
