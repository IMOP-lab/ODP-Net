# UNeXt

## Manuscript entry

- Citation: J. M. J. Valanarasu and V. M. Patel, "UNeXt: MLP-based rapid medical
  image segmentation network", MICCAI 2022, pp. 23-33
  (`\cite{valanarasu2022unext}` in the manuscript).
- Complexity-table fingerprint (`tab:complexity`): **1.47 M** parameters,
  **0.57 GFLOPs**. The table states a standardized 256 x 256 three-channel
  input.
- Parameter count is independent of the input size, so the value to reproduce
  is exactly 1.47 M. Measured THOP `thop_gmacs` (not `thop_gflops_2x`) is the
  quantity comparable to this table column.

## Upstream source

- Repository: <https://github.com/jeya-maria-jose/UNeXt-pytorch> (official; the
  README states "Official Pytorch Code base for UNeXt").
- License: MIT, Copyright (c) 2022 Jeya Maria Jose (read from the upstream
  `LICENSE` file).
- Model file: `archs.py`. Relevant classes: `UNext` (default configuration,
  `embed_dims=[128, 160, 256]`) and `UNext_S` (smaller variant, different
  `embed_dims` and halved convolutional widths).
- The manuscript's 1.47 M fingerprint matches `UNext`, not `UNext_S`.

## Benchmark interface

Upstream signature:

```python
UNext(num_classes, input_channels=3, deep_supervision=False, img_size=224,
      patch_size=16, in_chans=3, embed_dims=[128, 160, 256], ...)
```

- Instantiate as `UNext(num_classes=2)`.
- `img_size` is only fed to `OverlapPatchEmbed` metadata; its `forward`
  recomputes `H, W` from the actual feature-map shape, so the network accepts
  both 224 x 224 and 256 x 256 with no fixed-size assertion, and `forward`
  contains no hardcoded resize.
- `input_channels` is accepted but **not used**: `self.encoder1` is hardcoded as
  `nn.Conv2d(3, 16, ...)`. The benchmark is 3-channel, so this is harmless, but
  a configurable input channel count should not be expected.
- Normalization: `nn.BatchNorm2d` in the convolutional stages and
  `nn.LayerNorm` in the MLP stages. The BatchNorm path runs in eval mode and the
  LayerNorm path is batch-independent, so batch size 1 is safe.
- `forward` returns `self.final(out)`, i.e. raw logits shaped
  `[B, num_classes, H, W]`. Note that `self.soft = nn.Softmax(dim=1)` is
  constructed but never applied, so the upstream model already satisfies this
  benchmark's logits convention and needs no softmax adapter.

## Required adaptation for this benchmark

Only the imports change; the module topology is untouched.

1. The `timm` dependency is replaced with local PyTorch implementations of
   `DropPath`, `to_2tuple`, and `trunc_normal_`. No `timm` installation is
   required.
2. Drop `from mmcv.cnn import ConvModule`. `ConvModule` is never used in
   `archs.py`, and removing it avoids the upstream pin `mmcv-full==1.2.7`, which
   does not build against PyTorch 2.5.1 + CUDA 12.1.
3. Drop `from utils import *` plus the unrelated `torchvision`, `matplotlib`,
   `os` and `pdb` imports.

Keep the `torch.roll`-based channel shifting: it is a plain tensor operation, so
it is CUDA-safe under `torch.inference_mode()`.

## Open items to confirm on the workstation

- **GFLOPs resolution: resolved, it is 256 x 256, not 224 x 224.** An
  analytically THOP-faithful count of the official `UNext` gives
  0.5620 G (convolution + linear) + 0.0101 G (BatchNorm/LayerNorm) = **0.572 G
  at 256 x 256**, against 0.438 G at 224 x 224. The tabulated 0.57 therefore
  matches 256 x 256, and the earlier suspicion that this row used a 224 x 224
  number is refuted. Reproduce the same figure with a real THOP run to close
  this out.
- **Parameter count.** An exact count gives 1,471,938 for `num_classes=2`
  (1,471,921 for `num_classes=1`), consistent with the tabulated 1.47 M.
- **THOP visibility.** `shiftmlp.forward` uses `torch.roll`, `torch.chunk`,
  `torch.narrow` and `F.pad` ahead of the linear layers, so the linear-layer MACs
  are counted while the shifting itself is not (about zero MACs either way).
  Note also that this model's upsampling uses `F.interpolate`, which THOP does
  **not** count, whereas models using `nn.Upsample` are counted.

## Integration status and benchmark command

Integrated as `model.py` in this directory and exported as `UNext` from
`network/__init__.py`. The current implementation is self-contained and does
not require `timm`, `mmcv`, or any other optional package.

```bash
cd ~/hgp/ODP-Net
CUDA_VISIBLE_DEVICES=0 python benchmark/benchmark.py \
    --model unext --size 256 --warmup 100 --runs 500 \
    --output benchmark/unext_256_gpu0.json
```

Expected: `parameters` = 1,471,938, and `thop.thop_gmacs` about 0.57 at
256 x 256 (about 0.44 at 224 x 224).
