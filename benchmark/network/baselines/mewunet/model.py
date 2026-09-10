"""MEW-UNet benchmark adapter.

This is a self-contained PyTorch port of ``models/mewunet.py`` from the
official MEW-UNet repository:
https://github.com/JCruan519/MEW-UNet

The training and dataset code is intentionally omitted.  The forward pass is
kept compatible with the upstream implementation and returns raw segmentation
logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Conv2dGNGELU(nn.Sequential):
    def __init__(
        self, in_channel: int, out_channel: int, kernel_size: int = 3,
        stride: int = 1, groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channel, out_channel, kernel_size, stride, padding,
                groups=groups, bias=False,
            ),
            nn.GroupNorm(4, out_channel),
            nn.GELU(),
        )


class Conv1dGNGELU(nn.Sequential):
    def __init__(
        self, in_channel: int, out_channel: int, kernel_size: int = 3,
        stride: int = 1, groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(
                in_channel, out_channel, kernel_size, stride, padding,
                groups=groups, bias=False,
            ),
            nn.GroupNorm(4, out_channel),
            nn.GELU(),
        )


class DepthWiseConv2d(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, kernel_size: int = 3,
        padding: int = 1, stride: int = 1, dilation: int = 1,
        norm_type: str = "gn", gn_num: int = 4,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim_in, dim_in, kernel_size=kernel_size, padding=padding,
            stride=stride, dilation=dilation, groups=dim_in,
        )
        if norm_type == "bn":
            self.norm_layer = nn.BatchNorm2d(dim_in)
        elif norm_type == "in":
            self.norm_layer = nn.InstanceNorm2d(dim_in)
        elif norm_type == "gn":
            self.norm_layer = nn.GroupNorm(gn_num, dim_in)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.norm_layer(self.conv1(x)))


class InvertedDepthWiseConv2d(nn.Module):
    def __init__(
        self, in_channel: int, out_channel: int, kernel_size: int = 3,
        stride: int = 1, expand_ratio: int = 2,
    ) -> None:
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel
        self.conv = nn.Sequential(
            Conv2dGNGELU(in_channel, hidden_channel, kernel_size=1),
            Conv2dGNGELU(
                hidden_channel, hidden_channel, kernel_size=kernel_size,
                stride=stride, groups=hidden_channel,
            ),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.conv(x)
        return x + result if self.use_shortcut else result


class InvertedDepthWiseConv1d(nn.Module):
    def __init__(
        self, in_channel: int, out_channel: int, kernel_size: int = 3,
        stride: int = 1, expand_ratio: int = 2,
    ) -> None:
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel
        self.conv = nn.Sequential(
            Conv1dGNGELU(in_channel, hidden_channel, kernel_size=1),
            Conv1dGNGELU(
                hidden_channel, hidden_channel, kernel_size=kernel_size,
                stride=stride, groups=hidden_channel,
            ),
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.conv(x)
        return x + result if self.use_shortcut else result


class MEW(nn.Module):
    def __init__(
        self, dim: int, bias: bool = False, a: int = 16, b: int = 16,
        c_h: int = 16, c_w: int = 16,
    ) -> None:
        super().__init__()
        # These tensors are buffers in the upstream implementation and are
        # intentionally retained for state-dict compatibility.
        self.register_buffer("dim", torch.as_tensor(dim))
        self.register_buffer("a", torch.as_tensor(a))
        self.register_buffer("b", torch.as_tensor(b))
        self.register_buffer("c_h", torch.as_tensor(c_h))
        self.register_buffer("c_w", torch.as_tensor(c_w))

        quarter = dim // 4
        self.a_weight = nn.Parameter(torch.Tensor(2, 1, quarter, a))
        self.b_weight = nn.Parameter(torch.Tensor(2, 1, quarter, b))
        self.c_weight = nn.Parameter(torch.Tensor(2, quarter, c_h, c_w))
        nn.init.ones_(self.a_weight)
        nn.init.ones_(self.b_weight)
        nn.init.ones_(self.c_weight)

        self.dw_conv = InvertedDepthWiseConv2d(quarter, quarter)
        self.wg_a = nn.Sequential(
            InvertedDepthWiseConv1d(quarter, 2 * quarter),
            InvertedDepthWiseConv1d(2 * quarter, 2 * quarter),
            InvertedDepthWiseConv1d(2 * quarter, quarter),
        )
        self.wg_b = nn.Sequential(
            InvertedDepthWiseConv1d(quarter, 2 * quarter),
            InvertedDepthWiseConv1d(2 * quarter, 2 * quarter),
            InvertedDepthWiseConv1d(2 * quarter, quarter),
        )
        self.wg_c = nn.Sequential(
            InvertedDepthWiseConv2d(quarter, 2 * quarter),
            InvertedDepthWiseConv2d(2 * quarter, 2 * quarter),
            InvertedDepthWiseConv2d(2 * quarter, quarter),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        _, channels, height, width = x1.size()

        # Axis-a frequency mixing.
        x1 = x1.permute(0, 2, 1, 3)
        x1 = torch.fft.rfft2(x1, dim=(2, 3), norm="ortho")
        a_weight = self.wg_a(
            F.interpolate(
                self.a_weight, size=x1.shape[2:4], mode="bilinear",
                align_corners=True,
            ).squeeze(1)
        ).unsqueeze(1).permute(1, 2, 3, 0)
        x1 = x1 * torch.view_as_complex(a_weight.contiguous())
        x1 = torch.fft.irfft2(
            x1, s=(channels, width), dim=(2, 3), norm="ortho"
        ).permute(0, 2, 1, 3)

        # Axis-b frequency mixing.
        x2 = x2.permute(0, 3, 1, 2)
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm="ortho")
        b_weight = self.wg_b(
            F.interpolate(
                self.b_weight, size=x2.shape[2:4], mode="bilinear",
                align_corners=True,
            ).squeeze(1)
        ).unsqueeze(1).permute(1, 2, 3, 0)
        x2 = x2 * torch.view_as_complex(b_weight.contiguous())
        x2 = torch.fft.irfft2(
            x2, s=(channels, height), dim=(2, 3), norm="ortho"
        ).permute(0, 2, 3, 1)

        # Spatial/frequency mixing over the third axis.
        x3 = torch.fft.rfft2(x3, dim=(2, 3), norm="ortho")
        c_weight = self.wg_c(
            F.interpolate(
                self.c_weight, size=x3.shape[2:4], mode="bilinear",
                align_corners=True,
            )
        ).permute(1, 2, 3, 0)
        x3 = x3 * torch.view_as_complex(c_weight.contiguous())
        x3 = torch.fft.irfft2(
            x3, s=(height, width), dim=(2, 3), norm="ortho"
        )

        x4 = self.dw_conv(x4)
        return torch.cat([x1, x2, x3, x4], dim=1)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            InvertedDepthWiseConv2d(dim, mlp_ratio * dim),
            InvertedDepthWiseConv2d(mlp_ratio * dim, dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(4, dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class MEWB(nn.Module):
    def __init__(self, dim: int, depth: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [PreNorm(dim, MEW(dim)), PreNorm(dim, MLP(dim, mlp_ratio))]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, feed_forward in self.layers:
            x = attn(x) + x
            x = feed_forward(x) + x
        return x


class MEWUNet(nn.Module):
    """Official MEW-UNet architecture with benchmark-compatible arguments."""

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 2,
        *,
        dim: list[int] | None = None,
        depth: list[int] | None = None,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        dim = [32, 64, 128, 256, 512] if dim is None else list(dim)
        depth = [1, 2, 2, 4] if depth is None else list(depth)
        if len(dim) != 5 or len(depth) != 4:
            raise ValueError("MEW-UNet expects five dims and four depths")
        if any(ch % 4 for ch in dim):
            raise ValueError("all MEW-UNet channel widths must be divisible by 4")

        self.e0 = nn.Sequential(DepthWiseConv2d(n_channels, dim[0], norm_type="bn"))
        self.e1 = nn.Sequential(
            DepthWiseConv2d(dim[0], dim[1]), MEWB(dim[1], depth[0], mlp_ratio)
        )
        self.e2 = nn.Sequential(
            DepthWiseConv2d(dim[1], dim[2]), MEWB(dim[2], depth[1], mlp_ratio)
        )
        self.e3 = nn.Sequential(
            DepthWiseConv2d(dim[2], dim[3]), MEWB(dim[3], depth[2], mlp_ratio)
        )
        self.e4 = nn.Sequential(
            DepthWiseConv2d(dim[3], dim[4]), MEWB(dim[4], depth[3], mlp_ratio)
        )

        self.d4 = nn.Sequential(
            MEWB(dim[4], depth[3], mlp_ratio), DepthWiseConv2d(dim[4], dim[3])
        )
        self.d3 = nn.Sequential(
            MEWB(dim[3], depth[2], mlp_ratio), DepthWiseConv2d(dim[3], dim[2])
        )
        self.d2 = nn.Sequential(
            MEWB(dim[2], depth[1], mlp_ratio), DepthWiseConv2d(dim[2], dim[1])
        )
        self.d1 = nn.Sequential(
            MEWB(dim[1], depth[0], mlp_ratio), DepthWiseConv2d(dim[1], dim[0])
        )
        self.d0 = nn.Sequential(nn.Conv2d(dim[0], n_classes, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x0 = F.max_pool2d(self.e0(x), 2, 2)
        x1 = F.max_pool2d(self.e1(x0), 2, 2)
        x2 = F.max_pool2d(self.e2(x1), 2, 2)
        x3 = F.max_pool2d(self.e3(x2), 2, 2)
        x4 = F.max_pool2d(self.e4(x3), 2, 2)

        out4 = F.interpolate(
            self.d4(x4), scale_factor=(2, 2), mode="bilinear", align_corners=True
        )
        out4 = x3 + out4
        out3 = F.interpolate(
            self.d3(out4), scale_factor=(2, 2), mode="bilinear", align_corners=True
        )
        out3 = x2 + out3
        out2 = F.interpolate(
            self.d2(out3), scale_factor=(2, 2), mode="bilinear", align_corners=True
        )
        out2 = x1 + out2
        out1 = F.interpolate(
            self.d1(out2), scale_factor=(2, 2), mode="bilinear", align_corners=True
        )
        out1 = x0 + out1
        return F.interpolate(
            self.d0(out1), scale_factor=(2, 2), mode="bilinear", align_corners=True
        )


__all__ = [
    "MEWUNet",
    "MEW",
    "MEWB",
    "MLP",
    "DepthWiseConv2d",
    "InvertedDepthWiseConv2d",
]
