"""Explicit VMamba backbone adapter.

The official VMamba repository releases a classifier and a multi-scale
``Backbone_VSSM``; it does not release a dense segmentation decoder.  This
adapter therefore exposes the backbone features only and is intentionally not
registered as a benchmark segmentation model until a decoder is specified.
"""

from __future__ import annotations

import importlib.util

from torch import nn


class VMambaBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        missing = [name for name in ("einops", "timm", "fvcore") if importlib.util.find_spec(name) is None]
        if missing:
            raise ImportError("VMamba requires optional packages: " + ", ".join(missing))
        from .vmamba_upstream import Backbone_VSSM

        self.backbone = Backbone_VSSM(
            out_indices=(0, 1, 2, 3),
            pretrained=None,
            **kwargs,
        )

    def forward(self, x):
        return self.backbone(x)


__all__ = ["VMambaBackbone"]
