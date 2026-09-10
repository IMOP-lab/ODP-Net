# Polyp-PVT baseline source

Adapted from:

<https://github.com/DengPingFan/Polyp-PVT>

Branch: `main`.

Upstream model files:

- <https://github.com/DengPingFan/Polyp-PVT/blob/main/lib/pvt.py>
  (`class PolypPVT`, the CFM / SAM / capsule decoder) -> adapted into
  `model.py`.
- <https://github.com/DengPingFan/Polyp-PVT/blob/main/lib/pvtv2.py>
  (`PyramidVisionTransformerImpr`, `pvt_v2_b2()`, the PVT-v2-B2 backbone) ->
  adapted into `pvtv2.py`.

## License

**The upstream repository contains no LICENSE file.**  The only statement of
terms is in its README:

> This repo is the official implementation of "Polyp-PVT: Polyp Segmentation
> with Pyramid Vision Transformers". ... The code is free for research and
> education use only. Any comercial use should get formal permission first.

That is a **non-commercial, research-and-education-only** restriction with no
grant of redistribution rights, and no license is invented here.  Confirm the
terms with the authors before redistributing this adapted implementation or
using it commercially.  `lib/pvtv2.py` is itself derived from PVTv2 (upstream
<https://github.com/whai362/PVT>, Apache-2.0), but that license is not restated
in the Polyp-PVT repository.

## Adaptations made here

The architecture is copied verbatim; only the interface and the imports were
changed.  See the module docstrings of `model.py` and `pvtv2.py` for the
complete list.  In short:

- `PolypPVT(n_channels=3, n_classes=2, channel=32, pretrained=None,
  return_multi=False)`; `n_channels` is threaded into the backbone's first
  overlap patch embedding and `n_classes` sizes both output heads (upstream:
  `in_chans=3` and two 1-channel heads).
- `forward` returns a single `[B, n_classes, H, W]` logit tensor by default
  (the element-wise mean of the two upstream head outputs, which is how the
  upstream evaluation combines them); `return_multi=True` restores the upstream
  `(prediction1_8, prediction2_8)` tuple.
- The unconditional `torch.load('./pretrained_pth/pvt_v2_b2.pth')` is optional:
  nothing is read from disk unless `pretrained` is a path, in which case it is
  loaded with `map_location='cpu'`.
- `timm.models.registry.register_model` is dropped and `pvt_v2_b2` is a plain
  factory function; the `timm.models.layers` import is guarded with a
  `timm.layers` (timm >= 0.9) first / `timm.models.layers` fallback.
- Relative import (`from .pvtv2 import pvt_v2_b2`) instead of `from lib.pvtv2
  import pvt_v2_b2`.

Dependencies: **torch + timm only** (no einops, no mmcv/mmseg, no
segmentation-models-pytorch, no CUDA kernels, and no network access or
checkpoint files at construction time).
