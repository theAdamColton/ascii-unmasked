import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dDownscale(nn.Module):
    """
    Halves the input res
    """

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()

        stride = 2
        zero_padding = kernel_size // 2

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=zero_padding,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class Conv2dDoubleDownscale(nn.Module):
    """
    Divides the input res by 4
    """

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()

        stride = 4
        zero_padding = 1
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=zero_padding,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        stride = 1
        zero_padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=zero_padding,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class BilinearConvUpsample(nn.Module):
    """
    Multiplies the resolution by 2
    """

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()

        # This sets the zero_pad so that the conv2d layer will have
        # the same output width and height as its input
        assert kernel_size % 2 == 1
        zero_pad = kernel_size // 2

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=zero_pad
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, scale=2.0):
        x = F.upsample(x, scale_factor=scale, mode="bilinear")
        return self.layers(x)


class Flatten(nn.Module):
    def forward(self, input):
        return input.flatten(start_dim=1, end_dim=-1)


class UnFlatten(nn.Module):
    def __init__(self, size):
        super(UnFlatten, self).__init__()
        self.size = size

    def forward(self, input):
        return input.view(input.size(0), self.size, 1, 1)


class GenericUnflatten(nn.Module):
    def __init__(self, *shape):
        super(GenericUnflatten, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.shape[0], *self.shape)


class ArgMax(nn.Module):
    def forward(self, input):
        return torch.argmax(input, 1)
