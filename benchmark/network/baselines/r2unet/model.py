"""Self-contained R2U-Net baseline.

The implementation follows ``r2unet.py`` from:
https://github.com/navamikairanda/R2U-Net

The public repository uses channel widths 64/128/256/512/1024.  The
manuscript complexity row (9.78 M parameters) is reproduced by setting
``base_channels=32``; this gives widths 32/64/128/256/512.  The architecture
is otherwise unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


class RecurrentBlock(nn.Module):
    """A shared convolution applied recurrently ``t`` times."""

    def __init__(self, channels: int, t: int = 2) -> None:
        super().__init__()
        if t < 1:
            raise ValueError("t must be >= 1")
        self.t = int(t)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        for _ in range(1, self.t):
            x1 = self.conv(x + x1)
        return x1


class RRCNNBlock(nn.Module):
    """Recurrent residual convolutional block used by R2U-Net."""

    def __init__(self, in_channels: int, out_channels: int, t: int = 2) -> None:
        super().__init__()
        self.conv_1x1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.recurrent = nn.Sequential(
            RecurrentBlock(out_channels, t=t),
            RecurrentBlock(out_channels, t=t),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1x1(x)
        return x + self.recurrent(x)


class UpConv(nn.Module):
    """Nearest-neighbor upsampling followed by convolution, BN and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class R2UNet(nn.Module):
    """R2U-Net with a benchmark-compatible ``[B, C, H, W]`` output.

    Parameters
    ----------
    n_channels:
        Number of input image channels.
    n_classes:
        Number of output logits/channels.
    base_channels:
        Width of the first encoder stage.  ``32`` reproduces the 9.78M
        parameter row in the manuscript; ``64`` is the upstream default.
    t:
        Number of recurrent applications in each recurrent block.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 2,
        *,
        base_channels: int = 32,
        t: int = 2,
    ) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be >= 1")

        f = [base_channels * (2**i) for i in range(5)]
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = RRCNNBlock(n_channels, f[0], t=t)
        self.enc2 = RRCNNBlock(f[0], f[1], t=t)
        self.enc3 = RRCNNBlock(f[1], f[2], t=t)
        self.enc4 = RRCNNBlock(f[2], f[3], t=t)
        self.enc5 = RRCNNBlock(f[3], f[4], t=t)

        self.up5 = UpConv(f[4], f[3])
        self.dec5 = RRCNNBlock(f[4], f[3], t=t)
        self.up4 = UpConv(f[3], f[2])
        self.dec4 = RRCNNBlock(f[3], f[2], t=t)
        self.up3 = UpConv(f[2], f[1])
        self.dec3 = RRCNNBlock(f[2], f[1], t=t)
        self.up2 = UpConv(f[1], f[0])
        self.dec2 = RRCNNBlock(f[1], f[0], t=t)

        self.out = nn.Conv2d(f[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.maxpool(x1))
        x3 = self.enc3(self.maxpool(x2))
        x4 = self.enc4(self.maxpool(x3))
        x5 = self.enc5(self.maxpool(x4))

        d5 = self.up5(x5)
        d5 = self.dec5(torch.cat((x4, d5), dim=1))
        d4 = self.up4(d5)
        d4 = self.dec4(torch.cat((x3, d4), dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat((x2, d3), dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat((x1, d2), dim=1))
        return self.out(d2)


# Alias matching the upstream repository's class name.
R2U_Net = R2UNet

__all__ = ["R2UNet", "R2U_Net", "RecurrentBlock", "RRCNNBlock"]
