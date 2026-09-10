"""DAttUNet baseline (Dual-Decoder Attention U-Net) for the benchmark harness.

Upstream repository: https://github.com/faresbougourzi/PDAtt-Unet
Upstream file:       ``Architectures.py`` (repository root), class ``DAttUNet``
Exact revision:      branch ``main2`` at commit
                     ``87a25dc8137eccc3e3b13d4444a705e300b8eec8``

The architecture below is a verbatim copy of the upstream ``DAttUNet`` plus its
upstream helper blocks (``DoubleConv`` and ``Attention_block``); no layer has
been changed, added, removed or reordered. The only adaptation is the
harness-facing constructor, which renames the upstream ``input_channels`` /
``num_classes`` arguments to the harness convention ``n_channels`` /
``n_classes`` and forwards them to the identical upstream submodules. The
upstream ``deep_supervision`` argument is kept for fidelity even though
upstream ``DAttUNet`` never reads it.

Return-value adaptation (important)
-----------------------------------
Upstream ``DAttUNet.forward`` returns the 2-tuple ``(output, output2)`` -- an
infection head and a lung head, each ``[B, num_classes, H, W]`` raw logits.
The benchmark harness expects a single tensor, so this adapter **returns only
``output``** by default::

    model = DAttUNet(n_channels=3, n_classes=2)
    logits = model(x)          # x: [B, 3, H, W] -> [B, 2, H, W] float32

Passing ``return_aux=True`` restores the upstream behaviour and returns the
unchanged upstream tuple ``(output, output2)``::

    logits, aux = model(x, return_aux=True)

``forward`` returns raw logits (no sigmoid/softmax), matching the other
baselines in this harness. Unlike ``PAttUNet``, upstream ``DAttUNet`` has no
pyramid branch and calls no functional resize: its only size constraint is the
four 2x pool/upsample stages, so H and W that are multiples of 16 are safe
(this is what the harness uses).

Licensing caveat: the upstream repository ships **no LICENSE file** (and no
COPYING/NOTICE file, and no licensing statement in its README), so the default
is "all rights reserved". Redistribution rights are therefore NOT granted by
the upstream author; resolve this before publishing or redistributing this
file. See the accompanying ``SOURCE_NOTICE.md``.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2, as defined upstream."""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Attention_block(nn.Module):
    """Attention gate (additive, sigmoid-normalised), as defined upstream."""

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class DAttUNet(nn.Module):
    """Dual-Decoder Attention U-Net, upstream ``Architectures.py::DAttUNet``."""

    def __init__(self, n_channels=3, n_classes=2, deep_supervision=False, return_aux=False):
        super(DAttUNet, self).__init__()

        # Harness-facing names for the upstream ``input_channels`` /
        # ``num_classes`` arguments. ``deep_supervision`` is upstream-accepted
        # but unused by this architecture.
        input_channels = n_channels
        num_classes = n_classes
        self.n_channels = n_channels
        self.n_classes = n_classes
        # When True, ``forward`` returns the upstream ``(output, output2)``
        # tuple; when False (default) it returns only ``output``.
        self.return_aux = return_aux

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        nb_filter = [32, 64, 128, 256, 512]

        self.conv0_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv3_1_2 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2_2 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3_2 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4_2 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.Att4 = Attention_block(F_g=nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[2])
        self.Att3 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[1])
        self.Att2 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att1 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0], F_int=int(nb_filter[0] / 2))
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.Att4_2 = Attention_block(F_g=nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[2])
        self.Att3_2 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[1])
        self.Att2_2 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att1_2 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0], F_int=int(nb_filter[0] / 2))
        self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input, return_aux=None):
        # ``return_aux`` defaults to the constructor value.
        if return_aux is None:
            return_aux = self.return_aux

        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # decoding + concat path
        # Att1
        x3_1 = self.up(x4_0)
        x3_0_1 = self.Att4(g=x3_1, x=x3_0)
        x3_1 = self.conv3_1(torch.cat((x3_0_1, x3_1), dim=1))

        x3_1_2 = self.up(x4_0)
        x3_0_2 = self.Att4_2(g=x3_1_2, x=x3_0)
        x3_1_2 = self.conv3_1_2(torch.cat((x3_0_2, x3_1_2), dim=1))

        # Att2
        x2_2 = self.up(x3_1)
        x2_0_1 = self.Att3(g=x2_2, x=x2_0)
        x2_2 = self.conv2_2(torch.cat((x2_0_1, x2_2), dim=1))

        x2_2_2 = self.up(x3_1_2)
        x2_0_2 = self.Att3_2(g=x2_2_2, x=x2_0)
        x2_2_2 = self.conv2_2_2(torch.cat((x2_0_2, x2_2_2), dim=1))

        # Att3
        x1_3 = self.up(x2_2)
        x1_0_1 = self.Att2(g=x1_3, x=x1_0)
        x1_3 = self.conv1_3(torch.cat((x1_0_1, x1_3), dim=1))

        x1_3_2 = self.up(x2_2_2)
        x1_0_2 = self.Att2_2(g=x1_3_2, x=x1_0)
        x1_3_2 = self.conv1_3_2(torch.cat((x1_0_2, x1_3_2), dim=1))

        # Att4
        x0_4 = self.up(x1_3)
        x0_0_1 = self.Att1(g=x0_4, x=x0_0)
        x0_4 = self.conv0_4(torch.cat((x0_0_1, x0_4), dim=1))

        x0_4_2 = self.up(x1_3_2)
        x0_0_2 = self.Att1_2(g=x0_4_2, x=x0_0)
        x0_4_2 = self.conv0_4_2(torch.cat((x0_0_2, x0_4_2), dim=1))

        output = self.final(x0_4)
        output2 = self.final2(x0_4_2)

        if return_aux:
            # Upstream return value, unchanged.
            return output, output2
        return output


__all__ = ["DAttUNet"]
