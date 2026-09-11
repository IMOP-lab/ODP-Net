"""Adapters for the official VMamba classification/backbone implementations.

The official repository does not release a dense segmentation decoder.  The
benchmark adapter therefore measures the published ``vmamba_tiny_s1l8``
classification configuration unless a separate decoder is supplied.
"""

from __future__ import annotations

import importlib.util

from torch import nn


class VMambaClassifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        missing = [name for name in ("einops", "timm") if importlib.util.find_spec(name) is None]
        if missing:
            raise ImportError("VMamba requires optional packages: " + ", ".join(missing))
        from .vmamba_upstream import vmamba_tiny_s1l8

        self.model = vmamba_tiny_s1l8(channel_first=True)

    def forward(self, x):
        return self.model(x)


class VMambaBackbone(nn.Module):
    """Expose official multi-scale features for decoder experiments."""

    def __init__(self, **kwargs):
        super().__init__()
        missing = [name for name in ("einops", "timm") if importlib.util.find_spec(name) is None]
        if missing:
            raise ImportError("VMamba requires optional packages: " + ", ".join(missing))
        from .vmamba_upstream import Backbone_VSSM

        self.backbone = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=None, **kwargs)

    def forward(self, x):
        return self.backbone(x)


__all__ = ["VMambaClassifier", "VMambaBackbone"]
