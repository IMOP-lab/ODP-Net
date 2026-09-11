# VMamba

## Manuscript entry

- Citation: Y. Liu, Y. Tian, Y. Zhao, H. Yu, L. Xie, Y. Wang, Q. Ye, J. Jiao,
  Y. Liu, "VMamba: Visual State Space Model", NeurIPS 37, 103031-103063 (2024)
  (`\cite{liu2024vmamba}`).
- Complexity-table fingerprint (`tab:complexity`): **22.04 M** parameters,
  **4.11 GFLOPs**, 8.18 ms.
- **This row cannot currently be reproduced from the official implementation.**
  The values are identical to the VM-UNet row (22.04 M / 4.11 G), and no
  official VMamba configuration yields that pair.

## Why the fingerprint does not match

Numbers recomputed from the upstream source, cross-checked against the official
`assets/performance.md` table (classification at 224 x 224, ImageNet-1K):

| Official configuration | True params | Published FLOPs |
|---|---|---|
| Vanilla-VMamba-T (v0) | 22,893,448 (22.89 M); 22.12 M without the 1000-class head | 5.6 G (older convention: 4.5 G) |
| VMamba-T `s1l8` (v2) | 30,217,960 (30.22 M) | 4.9 G |
| VMamba-T `s2l5` (v2) | 30,657,448 (30.66 M) | 4.9 G |
| VMamba-T + UperNet (ADE20K, 512 x 512) | 62 M | 949 G |
| VMamba-T + Mask R-CNN FPN (COCO) | 50 M | 271 G |

- Vanilla-VMamba-T is the closest on parameters (22.89 M ≈ the "22 M" of the
  original paper) but not on FLOPs (4.5-5.6 G at 224, not 4.11 G).
- The UperNet / Mask R-CNN adaptations are off by 2.3-2.8x in parameters and by
  two orders of magnitude in FLOPs.
- The identical 22.04 M / 4.11 G pair is exactly what **VM-UNet** reports (see
  `../vm_unet/README.md`), which is most consistent with this row being an
  accidental duplicate of the VM-UNet measurement rather than an independent
  VMamba measurement.
- This is a strong inference from the numbers, **not** a confirmed fact about
  how the table was produced. It needs an author-side check of the original
  measurement script/log before the revision states anything about it.

## Upstream source

- Repository: <https://github.com/MzeroMiko/VMamba> (official; the NeurIPS 2024
  proceedings abstract page states "Source code is available at" this URL).
- License: MIT, Copyright (c) 2024 MzeroMiko. It bundles derivations of Mamba
  (Apache-2.0) and Swin (MIT); `mamba-ssm` is Apache-2.0 and `causal-conv1d` is
  BSD-3-Clause if kernels are redistributed.
- Model file: `classification/models/vmamba.py` -> `class VSSM`, and
  `class Backbone_VSSM(VSSM)` (deletes the classifier, adds `outnorm{i}`).
  Factories in the same file: `vanilla_vmamba_tiny()`, `vmamba_tiny_s1l8()`,
  `vmamba_tiny_s2l5()`, `vmamba_small_s1l20()`, `vmamba_base_s1l20()`.

## Integration obstacle: the official backbone is a classifier

- `VSSM.forward` ends in `self.classifier(x)`, a
  `norm -> permute -> AdaptiveAvgPool2d(1) -> Flatten -> Linear` head, so it
  returns a **classification vector `[B, num_classes]`** (i.e. `[B,1000]` for
  the official ImageNet configs), not dense logits.
- `Backbone_VSSM.forward` returns a **list of four multi-scale
  `[B, C, h, w]` feature maps**, not segmentation logits.
- Therefore the benchmark contract `Model(n_channels=3, n_classes=2)` returning
  `[B, 2, H, W]` cannot be satisfied by the vanilla backbone. A segmentation
  comparison requires an explicit decoder choice (the repository pairs the
  backbone with UperNet via mmsegmentation for ADE20K, and with Mask R-CNN/FPN
  via mmdetection for COCO), and the reported parameters/FLOPs then belong to
  backbone + decoder, not to the 22.89 M / 30.22 M backbone alone.
- For an architecture-only smoke/throughput measurement, this benchmark now
  exposes the official `vmamba_tiny_s1l8` **classification** model through
  `--model vmamba`. Its output is `[B,1000]`; it is not comparable to a dense
  segmentation row. A segmentation comparison still requires an explicitly
  named decoder (for example UperNet) and should be integrated separately.

## Model details once a configuration is chosen

- Constructor: `VSSM(patch_size=4, in_chans=3, num_classes=1000,
  depths=[2,2,9,2], dims=[96,192,384,768], ssm_d_state=16, ssm_ratio=2.0, ...,
  forward_type="v2", mlp_ratio=4.0, norm_layer="LN",
  downsample_version="v2", patchembed_version="v1", posembed=False,
  imgsize=224, ...)`.
- Input size: no assertion, and `PatchMerging2D` pads odd H/W explicitly, so the
  backbone tolerates non-multiples of 32 (practical minimum H, W >= 32). All
  official configs use `posembed=False`, so there is no fixed-resolution
  positional embedding. Both 224 and 256 are safe.
- Normalization: `norm_layer` is selectable among `"ln"` (`nn.LayerNorm`),
  `"ln2d"` (`LayerNorm2d`) and `"bn"`; **official configs use `ln`/`ln2d`
  only**, so batch size 1 is safe. Note that `LayerNorm2d` subclasses
  `nn.LayerNorm`, and THOP's exact-type hook lookup therefore skips it, silently
  dropping its MACs.
- Dependencies: `timm==0.4.12`, `fvcore` (imported by `vmamba.py` at module
  import time), `einops`, optional `triton`, plus `mamba-ssm` 2.2.4 and
  `causal-conv1d`. Both `mamba-ssm` and `causal-conv1d` are **sdist-only on
  PyPI for every released version**; their `setup.py` guesses a prebuilt wheel
  for the exact (CUDA, torch, python, abi) tuple and otherwise compiles locally.
  The published 2.2.4 wheel targets `cu12torch2.2`, so on torch 2.5.1+cu121 a
  local CUDA source build is required. This is the main environment risk.
- Unlike VM-UNet, VMamba **does** ship pure-PyTorch and Triton fallbacks:
  `classification/models/csms6s.py` dispatches to `selective_scan_torch` when
  `backend == "torch"` or CUDA is unavailable, and `csm_triton.py` falls back to
  a plain-PyTorch `CrossScanF`. The PyTorch fallback is an O(L) Python loop, so
  it is unusable for latency measurement but useful for smoke-testing shapes.
- No pretrained weights are needed for an architecture-only benchmark
  (`Backbone_VSSM(pretrained=None)` skips loading).

## Measurement caveats

- **THOP undercounts Mamba models.** The selective scan is a custom autograd
  function backed by raw CUDA and is invisible to THOP's module hooks, as are
  the `torch.einsum` projections. The literature-standard counter for this
  family is `fvcore` with an explicit `prim::PythonOp.SelectiveScanCuda` JIT
  handler — which is what VMamba's own `VSSM.flops()` does — or the analytic
  `9*B*L*D*N` term. Community evidence for the THOP failure is
  state-spaces/mamba issue #303.
- **`torch.inference_mode()` is expected to fail** on the default
  `SelectiveScanCuda.apply` path because `ctx.save_for_backward(...)` is called
  on tensors created inside inference mode. Use `torch.no_grad()` for the
  latency loop. This also affects `benchmark.py`, which currently uses
  `torch.inference_mode()`.

## Open items

- Confirm from the authors' original measurement logs what the "VMamba" row was
  computed from. Until then, do not restate or defend this row in the revision.
- If the row is kept, fix the configuration explicitly and re-measure, recording
  the configuration in this file.
