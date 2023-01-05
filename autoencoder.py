import torch.nn as nn

from generic_modules import (
    Flatten,
    GenericUnflatten,
    BilinearConvUpsample,
    Conv2dDownscale,
    Conv2dDoubleDownscale,
    Conv2dBlock,
)


class Decoder(nn.Module):
    """Decoder with a single linear input layer, multiple
    BilinearConvUpsample upscaling layers, and batch normalization. Works on a 64x64
    image output size,
    """

    def __init__(self, kernel_size=3):
        super().__init__()
        self.decoder = nn.Sequential(
            # Input: batch_size by 256 by 16 by 16
            # Conv2dBlock(256, 256),
            Conv2dBlock(256, 224, kernel_size=kernel_size),
            BilinearConvUpsample(224, 200, kernel_size=kernel_size, scale=2.0),
            # Input: batch_size by 200 by 32 by 32
            Conv2dBlock(200, 160, kernel_size=kernel_size),
            BilinearConvUpsample(160, 128, kernel_size=kernel_size, scale=2.0),
            # Input: batch_size by 128 by 64 by 64
            Conv2dBlock(128, 95, kernel_size=kernel_size),
            Conv2dBlock(95, 95, kernel_size=kernel_size),
            nn.Conv2d(
                95,
                95,
                padding=kernel_size // 2,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, z):
        return self.decoder(z)


class Encoder(nn.Module):
    """
    Encoder, with a single output linear layer, and then another
    single linear layer producing either mu or sigma^2. This encoder works only
    with 64x64 input image resolution and a 8*8*8 = 512 latent dimension.
    """

    def __init__(self, kernel_size=3):
        super().__init__()

        self.encoder = nn.Sequential(
            # Size comments are based on an input shape of batch_size by 95 by
            # 64 by 64
            # Input batch_size x 95 x 64 x 64
            Conv2dBlock(95, 95, kernel_size=kernel_size),
            Conv2dBlock(95, 95, kernel_size=kernel_size),
            Conv2dBlock(95, 128, kernel_size=kernel_size),
            Conv2dDownscale(128, 160, kernel_size=kernel_size),
            # Input batch_size x 160 x 32 x 32
            Conv2dBlock(160, 200, kernel_size=kernel_size),
            Conv2dDownscale(200, 224, kernel_size=kernel_size),
            # Input batch_size x 160 x 16 x 16
            nn.Conv2d(
                224, 256, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
            ),
            # Input batch_size x 256 x 32 x 32
            # Conv2dBlock(256, 256),
        )

    def forward(self, x):
        return self.encoder(x)
