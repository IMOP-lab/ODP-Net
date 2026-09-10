# MEWUNet

## Manuscript entry

- Citation: J. Ruan, J. Gao, M. Xie, S. Xiang, "Learning multi-axis representation
  in frequency domain for medical image segmentation", Machine Learning 114, 10
  (2025) (`\cite{ruan2025learning}`).
- Complexity-table fingerprint (`tab:complexity`): **138.89 M** parameters,
  **40.52 GFLOPs**, 48.24 ms.
- **138.89 M is THOP's reported value, not the true parameter count.** Measured
  from the unmodified upstream source:
  - true `sum(p.numel())` = **140,268,525** (`out_c=9`, the repository default)
    or **140,268,294** (`out_c=2`, the benchmark setting)
  - THOP-reported `params` = **138,894,765** / **138,894,534**, i.e. 138.89 M,
    matching the manuscript exactly
  - the 1,373,760-parameter difference is fully explained by two THOP blind
    spots: `nn.GroupNorm` has no counting rule in THOP (544,320 parameters,
    692 GroupNorm instances), and the `MEW` module registers
    `a_weight`/`b_weight`/`c_weight` as raw `nn.Parameter` on a custom module
    that matches no rule (829,440 parameters).
    `140,268,294 - 544,320 - 829,440 = 138,894,534`.
- This is the **third** row with the same signature (see `../vm_unet/README.md`
  and `../ccvim/README.md`), which is now strong evidence that the table's
  `Params (M)` column was produced with `thop.profile()`. Re-measuring with
  `sum(p.numel())` will give 140.27 M and the table will no longer reproduce.
- **One cell does not reproduce:** THOP MACs at 256 x 256 are **39.51 G** (or
  40.07 G if GroupNorm MACs are added), against the tabulated 40.52 G. No
  configuration of the official repository gives 40.52 G. This single cell
  needs re-measurement.

## Upstream source

- Repository: <https://github.com/JCruan519/MEW-UNet> (official; the journal
  paper itself states "The code is publicly available at" this URL, and the
  owner handle matches the first author). Verified revision: `main` at
  `dcde8729c70413a27ecbb5258f1eb06937032280`.
- License: Apache License 2.0.
- Model file: `models/mewunet.py` -> `class MEWUNet`, with `MEW`, `MEWB`, `MLP`,
  `PreNorm`, `DepthWiseConv2d`, `InvertedDepthWiseConv2d`,
  `InvertedDepthWiseConv1d`, `Conv2dGNGELU`, `Conv1dGNGELU`.
- There is no pretrained backbone: MEW-UNet is trained from scratch (depthwise
  convolutions + GroupNorm + FFT), and no branch, tag or config in the official
  repository yields 138.89 M. The `dim=512` stage dominates the parameter count
  (e4 + d4 = 120.5 M of 140.3 M).

## Configuration and interface

```python
MEWUNet(in_c=3, out_c=9, dim=[32, 64, 128, 256, 512], depth=[1, 2, 2, 4], mlp_ratio=4)
```

- Argument names are `in_c` / `out_c`, **not** `n_channels` / `n_classes`.
  Instantiate as `MEWUNet(3, 2)`. Beware the repository default `out_c=9`
  (Synapse), which must be overridden.
- Input size: **H and W must be divisible by 32.** Verified working at 32, 64,
  96, 128, 160, 192, 224, 256; failing at 48, 112, 200, 216, 232, 240, 250, 255.
  The cause is five stride-2 max-pools against five x2 `F.interpolate` upsamples
  with size-exact decoder residual additions. Both 224 and 256 work.
- Normalization: `nn.GroupNorm(4, C)` throughout (692 instances), with the only
  exception being the stem `DepthWiseConv2d(..., norm_type='bn')` ->
  `nn.BatchNorm2d(3)`. No InstanceNorm or LayerNorm. Batch size 1 is safe, and
  the journal paper explicitly motivates GroupNorm over BatchNorm "to mitigate
  the influence of batch size". Note the single BatchNorm is a 3-channel
  parameter set, and THOP counts it as 4x input elements.
- Output: a single tensor `[B, out_c, H, W]` of raw logits (no softmax/argmax;
  the reference code trains with a BCE-Dice loss). With `out_c=2` this satisfies
  the benchmark contract directly.

## Dependencies

- **`torch` only** (`torch.fft` is built into PyTorch; `numpy` is imported but
  unused). No timm, no einops, no mmcv, no
  segmentation-models-pytorch, and nothing to compile, so this baseline runs on
  PyTorch 2.5.1+cu121 as-is.
- Constructs fully offline; no pretrained weights are downloaded or required.
  The only weight I/O in the repository is an optional evaluation checkpoint in
  `test.py`, hosted on Baidu Pan (not automatable, and irrelevant here).

## Measurement caveats

- **THOP is almost blind to this model's core contribution.** Every
  frequency-domain operation is a functional call and THOP hooks only
  `nn.Module` instances: `torch.fft.rfft2` / `torch.fft.irfft2` (about 108 calls
  per forward), the complex multiply, `torch.chunk`/`cat`/`permute`, the
  `F.interpolate` calls that resample the learnable external weights, and
  `F.max_pool2d` are all invisible; `nn.GELU` has no rule either. The reported
  MAC count is therefore essentially convolution-only.
- Cross-model inconsistency to be aware of when discussing the table:
  `nn.Upsample` **is** counted by THOP (as 11x output elements for bilinear)
  while `F.interpolate` is not. The PDAtt models use `nn.Upsample` and MEW-UNet
  uses `F.interpolate`, so the two rows are not counted on the same basis.
- `thop.profile(..., report_missing=True)` will emit warnings for `GroupNorm`,
  `GELU` and every custom module in this file, treating them as zero.

## Open items

- Re-measure the GFLOPs cell: THOP gives 39.51 G at 256 x 256, not 40.52 G.
- Decide how to report parameters: the true 140.27 M, or 138.89 M documented as
  the THOP-reported value. This is a manuscript decision.
- Confirm the upstream revision/commit used and record it here.
