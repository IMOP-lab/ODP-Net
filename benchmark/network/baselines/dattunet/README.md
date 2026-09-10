# DAttUNet

## Manuscript entry

- Citation: F. Bougourzi, C. Distante, F. Dornaika, A. Taleb-Ahmed, "PDAtt-Unet:
  Pyramid dual-decoder attention Unet for Covid-19 infection segmentation from
  CT-scans", Medical Image Analysis 86, 102797 (2023)
  (`\cite{bougourzi2023pdatt}`).
- Complexity-table fingerprint (`tab:complexity`): **11.25 M** parameters,
  **24.73 GFLOPs**, 4.63 ms.
- **Both values reproduce exactly** from the unmodified upstream source at
  256 x 256 with `num_classes=2`:
  - `sum(p.numel())` = **11,249,564** (11.25 M)
  - THOP MACs = **24.730 G** (24.73 G)
- As with `PAttUNet`, the 224 x 224 count is far lower, which confirms that the
  tabulated figures are 256 x 256 and are MACs rather than 2x MACs.

## Upstream source

This variant lives in the **same repository and the same file** as
`../pattunet/README.md`; that file documents the shared details. In brief:

- Repository: <https://github.com/faresbougourzi/PDAtt-Unet> (official), branch
  `main2` at `87a25dc8137eccc3e3b13d4444a705e300b8eec8`.
- Model file: `Architectures.py` -> `class DAttUNet`.
- **License: there is no LICENSE/COPYING/NOTICE file in the repository**, so the
  default is all rights reserved. Flag before redistribution.
- The `PAttUNet` / `DAttUNet` / `PYAttUNet` variant is selected by the
  module-level string literal `modl_n` in
  `detailed train and test/train_test_PDAttUnet.py`, not by a constructor
  argument. The shipped default is `'PYAttUNet'`.

## Configuration and interface

```python
DAttUNet(input_channels=3, num_classes=1, deep_supervision=False)
```

- Instantiate as `DAttUNet(input_channels=3, num_classes=2)`.
- `deep_supervision` is accepted but ignored.
- **Output is a tuple, not a tensor**: `forward` returns
  `(output, output2)`, each `[B, num_classes, H, W]` raw logits (an infection
  head and a lung head). The benchmark harness must take `model(x)[0]`.
  `PYAttUNet` returns a 2-tuple as well.
- Input size: **H and W must be divisible by 16**, imposed by the four 2x
  max-pool / 2x upsample stages. Unlike `PAttUNet`, this variant has no pyramid
  branch and calls no `TF.resize`. Both 224 and 256 work.
- Normalization: `nn.BatchNorm2d` exclusively; batch size 1 is fine in `eval()`.

## Dependencies

- `torch` + `torchvision`; nothing to compile. Constructs fully offline with no
  pretrained weights.

## Measurement caveats

- THOP parameter counting equals `sum(p.numel())` here (all leaves matched), so
  this row is unaffected by the THOP parameter artifact seen in the MEW-UNet and
  state-space rows.
- A tuple return is harmless for `thop.profile` (the return value is discarded)
  but breaks a single-tensor harness contract, so the adapter is required.
- `nn.Upsample` is counted by THOP while the pyramid branch's `TF.resize` is not.

## Open items

- Resolve the licensing question before redistributing this code.
- Confirm the upstream revision/commit used and record it here.

## Integration status and benchmark command

Integrated as `model.py` in this directory and exported as `DAttUNet` from
`network/__init__.py`. By default `forward` returns only the first head as a
single `[B, n_classes, H, W]` tensor; pass `return_aux=True` to get the upstream
`(output, output2)` tuple.

```bash
cd ~/hgp/ODP-Net
CUDA_VISIBLE_DEVICES=0 python benchmark/benchmark.py \
    --model dattunet --size 256 --warmup 100 --runs 500 \
    --output benchmark/dattunet_256_gpu0.json
```

Expected: `parameters` = 11,249,564 and `thop.thop_gmacs` = 24.730 G at
256 x 256. Requires `torchvision`.
