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
                1024,
                512,
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # 1024x2x2 -> 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(
                512,
                256,
                3,
                stride=2,
                padding=1,
                output_padding=0,  # output padding zero here, otherwise we get an 8x8 image
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
            nn.Conv2d(3, 128, 4, stride=2, padding=1),  # 3x32x32 -> 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 128x16x16 -> 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 256x8x8 -> 512x4x4
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
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 3, stride=2, padding=1, output_padding=1),
        )


class Fer2013ConvAE(AbstractAutoEncoder):
    def __init__(self, intrinsic_dim: int = 2):
        super().__init__(intrinsic_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 1x48x48 -> 32x24x24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x24x24 -> 64x12x12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),  # 32x12x12 -> 64x6x6
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, intrinsic_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(intrinsic_dim, 64 * 6 * 6),
            nn.BatchNorm1d(64 * 6 * 6),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(64, 6, 6)),
            nn.ConvTranspose2d(
                64, 64, 3, stride=2, padding=1, output_padding=1
            ),  # 64x6x6 -> 64x12x12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),  # 64x12x12 -> 32x24x24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, 3, stride=2, padding=1, output_padding=1
            ),  # 32x24x24 -> 1x48x48
        )


class LfwPeopleConvAE(AbstractAutoEncoder):
    def __init__(self, intrinsic_dim: int = 2):
        super().__init__(intrinsic_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (4, 3), stride=2, padding=(0, 1)),  # 1x62x47 -> 32x30x24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=(0, 1)),  # 32x30x24 -> 64x14x12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=(0, 1)),  # 64x14x12 -> 64x6x6
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, intrinsic_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(intrinsic_dim, 64 * 6 * 6),
            nn.BatchNorm1d(64 * 6 * 6),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(64, 6, 6)),
            nn.ConvTranspose2d(
                64, 64, 4, stride=2, padding=(0, 1)
            ),  # 64x6x6 -> 64x14x12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, (4, 4), stride=2, padding=(0, 1)
            ),  # 64x14x12 -> 32x30x24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, (5, 3), stride=2, padding=1, output_padding=(1, 0)
            ),
        )


class OlivettiFacesConvAE(AbstractAutoEncoder):
    def __init__(self, intrinsic_dim: int = 2):
        super().__init__(intrinsic_dim)
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
