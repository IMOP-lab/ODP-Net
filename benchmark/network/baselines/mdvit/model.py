"""Benchmark adapter for the official MDViT segmentation model.

The upstream implementation is kept in the neighboring ``*_upstream.py``
files.  MDViT has four domain-specific auxiliary decoders and returns
``[main_logits, auxiliary_logits]``; the benchmark uses domain ``0`` and the
main branch, with a 1x1 projection to the harness' two-channel contract.
"""

from __future__ import annotations

import importlib.util

import torch
from torch import nn


def _require_dependencies() -> None:
    missing = [name for name in ("einops", "timm") if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(
            "MDViT requires optional packages: " + ", ".join(missing)
            + ". Install them in the Linux benchmark environment."
        )


class MDViT(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 2, img_size: int = 224, **kwargs):
        super().__init__()
        if img_size % 32:
            raise ValueError("MDViT requires an input size divisible by 32")
        _require_dependencies()
        from .mdvit_upstream import MDViT as UpstreamMDViT

        self.backbone = UpstreamMDViT(
            img_size=img_size,
            in_chans=n_channels,
            adapt_method="Sup",
            num_domains=4,
            decoder_name="MLPFM",
            drop_rate=0.1,
            drop_path_rate=0.1,
            **kwargs,
        )
        self.head = nn.Conv2d(1, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The Sup variant requires a one-hot domain vector in every attention
        # stage.  Benchmark all samples against the first domain, matching the
        # fixed auxiliary branch selected below.
        domain_label = x.new_zeros((x.shape[0], 4))
        domain_label[:, 0] = 1
        result = self.backbone(x, domain_label=domain_label, d="0")
        if not isinstance(result, (list, tuple)) or not result:
            raise RuntimeError("MDViT returned an unexpected output")
        return self.head(result[0])


__all__ = ["MDViT"]
