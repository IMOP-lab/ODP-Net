# Source notice

The files ending in `_upstream.py` and the `mamba2/` directory are copied from
the official VMamba repository: https://github.com/MzeroMiko/VMamba (retrieved
2026-09-11). The upstream project is MIT licensed. `model.py` exposes the
official `vmamba_tiny_s1l8` classifier and multi-scale `Backbone_VSSM`. The
`vmamba` benchmark entry measures the classifier only; VMamba's repository does
not provide a dense segmentation decoder.
