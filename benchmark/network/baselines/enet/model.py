"""ENet baseline adapted from davidtvs/PyTorch-ENet.

The ENet topology follows the upstream ``models/enet.py`` implementation.
This adapter returns logits with shape ``[B, n_classes, H, W]``.
"""

import torch
import torch.nn as nn


class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, relu=True):
        super().__init__()
        activation = nn.ReLU if relu else nn.PReLU
        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias,
        )
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        return self.out_activation(self.batch_norm(torch.cat((main, ext), dim=1)))


class RegularBottleneck(nn.Module):
    def __init__(
        self,
        channels,
        internal_ratio=4,
        kernel_size=3,
        padding=0,
        dilation=1,
        asymmetric=False,
        dropout_prob=0,
        bias=False,
        relu=True,
    ):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > channels:
            raise ValueError(
                f"internal_ratio must be in [1, {channels}], got {internal_ratio}"
            )
        internal_channels = channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation(),
        )
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias,
                ),
                nn.BatchNorm2d(internal_channels),
                activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias,
                ),
                nn.BatchNorm2d(internal_channels),
                activation(),
            )
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                ),
                nn.BatchNorm2d(internal_channels),
                activation(),
            )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(channels),
            activation(),
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x):
        ext = self.ext_conv3(self.ext_conv2(self.ext_conv1(x)))
        return self.out_activation(x + self.ext_regul(ext))


class DownsamplingBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        internal_ratio=4,
        return_indices=False,
        dropout_prob=0,
        bias=False,
        relu=True,
    ):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise ValueError(
                f"internal_ratio must be in [1, {in_channels}], got {internal_ratio}"
            )
        self.return_indices = return_indices
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU

        self.main_max1 = nn.MaxPool2d(
            2,
            stride=2,
            return_indices=return_indices,
        )
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias,
            ),
            nn.BatchNorm2d(internal_channels),
            activation(),
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            nn.BatchNorm2d(internal_channels),
            activation(),
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            activation(),
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x):
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)
            max_indices = None

        ext = self.ext_regul(self.ext_conv3(self.ext_conv2(self.ext_conv1(x))))
        n, _, h, w = ext.shape
        ch_main = main.shape[1]
        if ch_main < ext.shape[1]:
            padding = ext.new_zeros(n, ext.shape[1] - ch_main, h, w)
            main = torch.cat((main, padding), dim=1)
        out = self.out_activation(main + ext)
        return (out, max_indices) if self.return_indices else out


class UpsamplingBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        internal_ratio=4,
        dropout_prob=0,
        bias=False,
        relu=True,
    ):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise ValueError(
                f"internal_ratio must be in [1, {in_channels}], got {internal_ratio}"
            )
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU

        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
        )
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation(),
        )
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias,
        )
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        main = self.main_unpool1(
            self.main_conv1(x),
            max_indices,
            output_size=output_size,
        )
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_activation(self.ext_tconv1_bnorm(ext))
        ext = self.ext_regul(self.ext_conv2(ext))
        return self.out_activation(main + ext)


class ENet(nn.Module):
    def __init__(
        self,
        n_classes=2,
        n_channels=3,
        encoder_relu=False,
        decoder_relu=True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.initial_block = InitialBlock(n_channels, 16, relu=encoder_relu)

        self.downsample1_0 = DownsamplingBottleneck(
            16, 64, return_indices=True, dropout_prob=0.01, relu=encoder_relu
        )
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        self.downsample2_0 = DownsamplingBottleneck(
            64, 128, return_indices=True, dropout_prob=0.1, relu=encoder_relu
        )
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        self.regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(
            16, n_classes, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, x):
        input_size = x.size()
        x = self.initial_block(x)

        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_4(self.regular1_3(self.regular1_2(self.regular1_1(x))))

        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular5_1(x)
        return self.transposed_conv(x, output_size=input_size)


__all__ = ["ENet"]
