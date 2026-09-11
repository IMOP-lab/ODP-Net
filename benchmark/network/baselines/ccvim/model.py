"""Benchmark adapter for the official CCViM segmentation model."""

from __future__ import annotations

import importlib.util

import torch
from torch import nn


def _require_dependencies() -> None:
    missing = [name for name in ("einops", "timm") if importlib.util.find_spec(name) is None]
    if importlib.util.find_spec("selective_scan_cuda") is None and importlib.util.find_spec("selective_scan_vmamba_pt202") is None:
        missing.append("selective_scan_cuda or selective_scan_vmamba_pt202 CUDA extension")
    if missing:
        raise ImportError(
            "CCViM requires optional packages/extensions: " + ", ".join(missing)
            + ". Install the CUDA selective-scan dependency on the Linux workstation."
        )


class CCViM(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 2, **kwargs):
        super().__init__()
        depths = kwargs.pop("depths", [2, 2, 2, 2])
        depths_decoder = kwargs.pop("depths_decoder", [2, 2, 2, 1])
        _require_dependencies()
        from .ccvim_upstream import CCViM as upstream_factory

        self.model = upstream_factory(
            in_chans=n_channels,
            num_classes=n_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=0.2,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


__all__ = ["CCViM"]
