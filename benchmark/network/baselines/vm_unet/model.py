"""Benchmark adapter for the official VM-UNet implementation."""

from __future__ import annotations

import importlib.util

import torch
from torch import nn


def _require_dependencies() -> None:
    missing = [name for name in ("einops", "timm") if importlib.util.find_spec(name) is None]
    if importlib.util.find_spec("mamba_ssm") is None and importlib.util.find_spec("selective_scan") is None:
        missing.append("mamba_ssm or selective_scan CUDA extension")
    if missing:
        raise ImportError(
            "VM-UNet requires optional packages/extensions: " + ", ".join(missing)
            + ". Install the CUDA selective-scan dependency on the Linux workstation."
        )


class VMUNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 2, **kwargs):
        super().__init__()
        depths = kwargs.pop("depths", [2, 2, 2, 2])
        depths_decoder = kwargs.pop("depths_decoder", [2, 2, 2, 1])
        _require_dependencies()
        from .vmunet_upstream import VMUNet as UpstreamVMUNet

        self.model = UpstreamVMUNet(
            input_channels=n_channels,
            num_classes=n_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=0.2,
            load_ckpt_path=None,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


__all__ = ["VMUNet"]
