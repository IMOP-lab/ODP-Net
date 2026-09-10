# Polyp-PVT

## Manuscript entry

- Citation: B. Dong, W. Wang, D.-P. Fan, J. Li, H. Fu, L. Shao, "Polyp-PVT:
  Polyp Segmentation with Pyramid Vision Transformers", CAAI Artificial
  Intelligence Research 2, 149-168 (2023) (`\cite{dong2023polyp}`).
- Complexity-table fingerprint (`tab:complexity`): **25.11 M** parameters,
  **5.30 GFLOPs**, 9.10 ms.
- Parameters reproduce **exactly**: `sum(p.numel())` = **25,107,604
  (25.108 M)**, of which the PVT-v2-B2 backbone is 24,849,856 (98.97%, stage
  split [1.0515, 2.4108, 9.9386, 9.5196] M) and the decoder/capsule head is
  257,748.
- The **GFLOPs entry is not a 256 x 256 measurement.** The Polyp-PVT authors
  publish no parameter/FLOPs table at all, and the canonical third-party figure
  is 10.005 G at **352 x 352**. The manuscript's 5.30 G equals that value
  rescaled by `(256/352)^2 = 0.5289`. An independent THOP count at 256 x 256
  gives about **5.77 G MACs**. Polyp-PVT is canonically trained and benchmarked
  at 352 x 352, so this row mixes resolutions.

## Upstream source

- Repository: <https://github.com/DengPingFan/Polyp-PVT> (official; the paper
  links it and the README states "This repo is the official implementation").
- **License: there is no LICENSE file.** The README grants use "free for
  research and education use only. Any comercial use should get formal
  permission first." This is a non-commercial restriction and must be checked
  before this benchmark code is redistributed. `lib/pvtv2.py` is derived from
  PVTv2 (upstream `whai362/PVT` is Apache-2.0), but that license is not restated
  in this repository.
- Model files:
  - `lib/pvt.py` -> `class PolypPVT(nn.Module)`
  - `lib/pvtv2.py` -> `pvt_v2_b2()` -> `PyramidVisionTransformerImpr`

## Configuration and interface

- Constructor: `PolypPVT(channel=32)`. That is the **only** argument; there is
  no `in_channels` and no `num_classes`.
- Backbone configuration: `patch_size=4`, `embed_dims=[64,128,320,512]`,
  `num_heads=[1,2,5,8]`, `mlp_ratios=[8,8,4,4]`, `depths=[3,4,6,3]`,
  `sr_ratios=[8,4,2,1]`, `drop_path_rate=0.1`, LayerNorm with `eps=1e-6`.
- Input size: there is no assertion anywhere in the model code. The `352`
  appears only in the scripts (`Train.py --trainsize 352`, `Test.py
  --testsize 352`). All positional-embedding code is commented out in
  `pvtv2.py`, so there is no fixed-size or interpolation requirement; any H, W
  works. Divisible by 32 gives clean stage shapes (patch-embed strides 4, 2, 2,
  2). Output is at full input resolution.
- Normalization: LayerNorm in the backbone and **`nn.BatchNorm2d`** in every
  decoder `BasicConv2d`. Batch size 1 is fine in `eval()`; in train mode the
  decoder batch statistics would be meaningless at batch 1, so always call
  `model.eval()`.
- **Output is a tuple of two tensors**: `return prediction1_8, prediction2_8`,
  each `[B, 1, H, W]` raw logits at full resolution (no sigmoid/softmax inside).
  The class count is hardcoded to 1 (`nn.Conv2d(channel, 1, 1)` for both
  `out_SAM` and `out_CFM`). A benchmark wrapper is required: combine the two
  maps and replace the 1-channel heads with `n_classes`-channel heads. Each head
  gains 33 parameters (32 weights + 1 bias), so moving from the upstream
  1-channel heads to 2-channel heads adds **66** parameters: the upstream
  `PolypPVT(n_channels=3, n_classes=1)` count is 25,107,604, and the harness
  model `PolypPVT(n_channels=3, n_classes=2)` is **25,107,670**, which still
  rounds to the tabulated 25.11 M.

## Mandatory adaptation

- `PolypPVT.__init__` **unconditionally** runs
  `torch.load('./pretrained_pth/pvt_v2_b2.pth')` relative to the current working
  directory. Without that file construction raises `FileNotFoundError`, and
  there is no `pretrained=False` switch. Either guard the load or supply any
  state_dict containing the backbone keys. Note that this `torch.load` has no
  `map_location` (a CUDA-saved checkpoint fails on a CPU-only machine) and on
  torch 2.5.1 defaults to `weights_only=False`.
- Dependencies: **torch + timm only**. `pvtv2.py` imports
  `timm.models.layers`, `timm.models.registry` and
  `timm.models.vision_transformer`; all three still resolve on modern timm via
  deprecation shims (FutureWarning only). No einops, no mmcv/mmseg, no
  segmentation-models-pytorch, and no CUDA kernel compilation.

## Measurement caveats

- THOP's `total_params` equals `sum(p.numel())` = 25,107,604 for this model
  (there are no parameters on non-leaf modules), so the parameter column is
  unaffected by the THOP artifact seen in the state-space rows.
- THOP undercounts MACs: the `torch.matmul` in `Attention.forward` has no hook,
  the two `F.interpolate(..., scale_factor=8)` full-resolution upsamples are not
  counted, and `F.softmax` is not counted.
- Independent THOP-style counts: **5.766 G at 256 x 256**, **10.901 G at
  352 x 352** (the ratio is exactly `(256/352)^2`).
- Pretrained PVT weights do not change the parameter count; they are loaded into
  pre-existing modules.

## Open items

- Decide explicitly which resolution this row reports. If the row is
  re-measured at 256 x 256 with THOP, expect about 5.77 G MACs, not 5.30 G.
- Confirm the upstream revision/commit used and record it here.

## Integration status and benchmark command

Integrated as `model.py` (decoder) plus `pvtv2.py` (PVT-v2-B2 backbone) in this
directory, exported as `PolypPVT` from `network/__init__.py`. `forward` returns
the mean of the two heads as a single `[B, n_classes, H, W]` tensor; pass
`return_multi=True` for the upstream tuple. No checkpoint file and no network
access are needed to build it (`pretrained=None` by default).

```bash
cd ~/hgp/ODP-Net
CUDA_VISIBLE_DEVICES=0 python benchmark/benchmark.py \
    --model polyp_pvt --size 256 --warmup 100 --runs 500 \
    --output benchmark/polyp_pvt_256_gpu0.json
```

Expected: `parameters` = 25,107,670 (the upstream 1-channel model is 25,107,604;
the two heads gain 66 parameters in total) and `thop.thop_gmacs` about 5.77 G at
256 x 256. Requires `timm`. Note that Polyp-PVT is canonically benchmarked at
352 x 352, so measure `--size 256` for table parity and optionally
`--size 352` for the canonical setting.
