import torch.nn as nn


class MnistConvolutionalAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        encoded = self.encoder.forward(x)
        decoded = self.decoder.forward(encoded)
        return decoded, encoded
