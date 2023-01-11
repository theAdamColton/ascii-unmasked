import math
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
        # Input: batch_size by 256 by 16 by 16
        # Conv2dBlock(256, 256),
        self.c1 = Conv2dBlock(512, 512, kernel_size=kernel_size)
        self.c1 = Conv2dBlock(512, 452, kernel_size=kernel_size)
        self.bl1 = BilinearConvUpsample(452, 392, kernel_size=kernel_size)
        # Input: batch_size by 200 by 32 by 32
        self.c2 = Conv2dBlock(392, 332, kernel_size=kernel_size)
        self.bl2 = BilinearConvUpsample(332, 272, kernel_size=kernel_size)
        # Input: batch_size by 128 by 64 by 64
        self.c3 = Conv2dBlock(272, 212, kernel_size=kernel_size)
        self.c4 = Conv2dBlock(212, 152, kernel_size=kernel_size)
        self.c5 = Conv2dBlock(
            152,
            95,
            kernel_size=kernel_size,
        )
        self.c6 = Conv2dBlock(
            95,
            95,
            kernel_size=kernel_size,
        )
        self.c7 = nn.Conv2d(95, 95, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, z, x_res):
        """
        Assumes square shape

        x_res is the desired output res of the decoder
        """
        z_res = z.shape[2]
        scale_factor1 = math.ceil(x_res / 2) / z_res
        scale_factor2 = x_res / math.ceil(x_res / 2)
        return self.c7(
            self.c6(
                self.c5(
                    self.c4(
                        self.c3(
                            self.bl2(
                                self.c2(self.bl1(self.c1(z), scale=scale_factor1)),
                                scale=scale_factor2,
                            )
                        )
                    )
                )
            )
        )


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
            Conv2dBlock(95, 152, kernel_size=kernel_size),
            Conv2dBlock(152, 212, kernel_size=kernel_size),
            Conv2dBlock(212, 272, kernel_size=kernel_size),
            Conv2dDownscale(272, 332, kernel_size=kernel_size),
            # Input batch_size x 160 x 32 x 32
            Conv2dBlock(332, 392, kernel_size=kernel_size),
            Conv2dDownscale(392, 452, kernel_size=kernel_size),
            # Input batch_size x 160 x 16 x 16
            nn.Conv2d(
                452, 512, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
            ),
            # Input batch_size x 256 x 32 x 32
            # Conv2dBlock(256, 256),
        )

    def forward(self, x):
        return self.encoder(x)
