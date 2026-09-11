# Source notice

The files ending in `_upstream.py` and the `mamba2/` directory are copied from
the official VMamba repository: https://github.com/MzeroMiko/VMamba (retrieved
2026-09-11). The upstream project is MIT licensed. `model.py` exposes the
official multi-scale `Backbone_VSSM`; VMamba's repository does not provide a
dense segmentation decoder, so this directory is not registered as a model in
`benchmark.py`.
