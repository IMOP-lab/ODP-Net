# PAttUNet

## Manuscript entry

- Citation: F. Bougourzi, C. Distante, F. Dornaika, A. Taleb-Ahmed, "PDAtt-Unet:
  Pyramid dual-decoder attention Unet for Covid-19 infection segmentation from
  CT-scans", Medical Image Analysis 86, 102797 (2023)
  (`\cite{bougourzi2023pdatt}`).
- Complexity-table fingerprint (`tab:complexity`): **11.17 M** parameters,
  **16.84 GFLOPs**, 4.80 ms.
- **Both values reproduce exactly** from the unmodified upstream source at
  256 x 256 with `num_classes=2`:
  - `sum(p.numel())` = **11,173,978** (11.17 M)
  - THOP MACs = **16.839 G** (16.84 G), and here THOP's parameter count equals
    the true count because every parameter sits on a matched leaf module
    (`Conv2d`, `BatchNorm2d`, `Upsample`).
- This row is the strongest evidence for the table's measurement convention:
  at 224 x 224 the same model gives only about 12.9 G MACs, so the tabulated
  figures are **256 x 256** and are **MACs, not 2x MACs**.

## Upstream source

- Repository: <https://github.com/faresbougourzi/PDAtt-Unet> (official; the owner
  is the first author, `Architectures.py` carries a `@author: FaresBougourzi`
  header, and the README embeds the exact BibTeX of the Medical Image Analysis
  article). Verified revision: branch `main2` at
  `87a25dc8137eccc3e3b13d4444a705e300b8eec8`.
- **License: there is no LICENSE/COPYING/NOTICE file in the repository**, and
  the README says nothing about licensing, so the default is all rights
  reserved. Vendoring this code into a released benchmark or artifact carries a
  real redistribution risk; flag it before publishing.
- Model file: `Architectures.py` at the repository root. A second copy under
  `detailed train and test/Architectures.py` is byte-identical apart from one
  blank comment line, so there is one canonical definition. The file also
  contains `UNet`, `UNetplus`, `AttUNet` and `PYAttUNet` (= the full Pyramid
  Dual-Decoder PDAtt-Unet).

## Configuration and interface

```python
PAttUNet(input_channels=3, num_classes=1, deep_supervision=False)
```

- Instantiate as `PAttUNet(input_channels=3, num_classes=2)`; the default
  `num_classes=1` must be overridden.
- `deep_supervision` is accepted but **ignored** by `PAttUNet` (and by
  `DAttUNet`, `AttUNet`, `PYAttUNet`); only `UNetplus` actually uses it.
- **Variant selection is not a constructor argument or config file.** The
  training script `detailed train and test/train_test_PDAttUnet.py` uses a
  module-level string literal `modl_n` and then
  `model = getattr(networks, modl_n)()`. Set `modl_n = 'PAttUNet'` for this
  variant; the shipped default is `'PYAttUNet'`, i.e. the full PDAtt-Unet, which
  is neither of the two variants used in the manuscript. Note also that the
  root `Architectures.py` is not imported by any script — only the copy under
  `detailed train and test/` is — and that the training branch hardcodes
  `outputs, outputs2 = model(x)`, so selecting a single-tensor variant like
  `PAttUNet` there requires a code edit.
- Input size: **H and W must be divisible by 16.** Verified working at 32, 48,
  64, 96, 112, 128, 160, 192, 224, 240, 256; failing at 200, 216, 232, 250, 255.
  The cause is the pyramid branch's `TF.resize(input, size=[int(H/divsize[i]),
  int(W/divsize[i])])` with `divsize=[2,4,8,16]`, whose truncation desynchronises
  the decoder's x2 upsampling from the encoder skips. Both 224 and 256 work.
- Normalization: `nn.BatchNorm2d` exclusively (2 per `DoubleConv`, 3 per
  `Attention_block`). No GroupNorm/InstanceNorm/LayerNorm. Batch size 1 is fine
  in `eval()`; in train mode batch statistics are used instead of running stats.
- Output: a **single tensor** `[B, num_classes, H, W]` of raw logits, which
  satisfies the benchmark contract directly with `num_classes=2`.

## Dependencies

- `torch` + **`torchvision`** (`Architectures.py` does
  `import torchvision.transforms.functional as TF` and calls `TF.resize` inside
  `PAttUNet.forward`). Nothing to compile.
- Constructs fully offline; no pretrained weights are downloaded or required,
  and the repository ships no weights. The only training-side extras
  (albumentations, opencv, nibabel, medpy, SimpleITK, ...) are irrelevant to
  architecture-only benchmarking.

## Measurement caveats

- THOP is well behaved on this model: all leaves are matched types, so
  `thop.profile().params` equals `sum(p.numel())`.
- `nn.Upsample` **is** counted by THOP (11x output elements for bilinear),
  while the `TF.resize` calls in the pyramid branch are functional and are not
  counted. This is a minor undercount, and it is worth remembering that
  MEW-UNet uses `F.interpolate` (never counted) instead of `nn.Upsample`, so the
  two rows are not counted on an identical basis.

## Open items

- Resolve the licensing question before redistributing this code.
- Confirm the upstream revision/commit used and record it here.

## Integration status and benchmark command

Integrated as `model.py` in this directory and exported as `PAttUNet` from
`network/__init__.py`. The class returns a single `[B, n_classes, H, W]` logits
tensor.

```bash
cd ~/hgp/ODP-Net
CUDA_VISIBLE_DEVICES=0 python benchmark/benchmark.py \
    --model pattunet --size 256 --warmup 100 --runs 500 \
    --output benchmark/pattunet_256_gpu0.json
```

Expected: `parameters` = 11,173,978 and `thop.thop_gmacs` = 16.839 G at
256 x 256. Requires `torchvision`.
