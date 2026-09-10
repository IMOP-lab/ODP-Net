# CCViM

## Manuscript entry

- Citation: Y. Zhu, D. Zhang, Y. Lin, Y. Feng, J. Tang, "Merging Context
  Clustering with Visual State Space Models for Medical Image Segmentation",
  IEEE TMI (2025) (`\cite{zhu2025merging}`).
- Complexity-table fingerprint (`tab:complexity`): **23.61 M** parameters,
  **4.57 GFLOPs**, 19.95 ms.
- **The 23.61 M fingerprint cannot be reproduced from the released code.**
  Measured on the released default configuration:
  - true `sum(p.numel())` = **32,162,096 (32.16 M)**
  - THOP-reported `params` = **26,772,242 (26.77 M)**
  - **23.61 M = 26,772,242 - 3,159,624 = 23,612,618**, an exact match obtained
    by subtracting the `BiAttn` module from the THOP value. The released
    `MultiScanVSSM` contains `self.attn = BiAttn(dim)`; an earlier revision
    without it, measured with THOP, reproduces the manuscript figure exactly.
- This is the **second** row in the table with the same signature (see
  `../vm_unet/README.md`: 22.04 M is THOP-visible vs 27.43 M true). Together they
  indicate the table's `Params (M)` column is a **THOP-reported** quantity, not
  `sum(p.numel())`, and is therefore not comparable across models that hold
  parameters on non-leaf custom modules.
- The repository's own `utils.cal_params_flops(model, 256, logger)` calls
  `thop.profile` on a 256 x 256 CUDA input, which is the likely provenance of
  both the parameter and FLOPs figures.

## Upstream source

- Repository: <https://github.com/zymissy/CCViM> (official; the arXiv abstract,
  the arXiv comments field and the paper body all link it, and the owner handle
  `zymissy` matches the corresponding author's email). Default branch is
  `master`; `main` does not exist.
- License: Apache License 2.0.
- Model files:
  - `models/CCViMUNet/CCViMUNet.py` -> wrapper class **`LCVMUNet`** (the harness
    entry point)
  - `models/CCViMUNet/CCViM.py` -> factory `CCViM` returning `VSSM`, with
    `VSSBlock`, `SS2D`, `MultiScanVSSM`, `LocalCluster`, `BiAttn`,
    `PatchMerging2D`, `PatchExpand2D`, `Final_PatchExpand2D`
  - `models/CCViMUNet/mamba/multi_mamba.py` -> `MultiScan`, `LocalCluster`
- Name collision warning: an unrelated 2026 MDPI watermarking paper also calls
  itself "CCViM". Do not confuse the two when searching.

## Configuration

The repository ships **one segmentation architecture** (no tiny/small/base
family); the released configs differ only in `num_classes` (1 for ISIC17/18,
9 for Synapse) and input size.

```python
LCVMUNet(input_channels=3, num_classes=1,
         depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1],
         drop_path_rate=0.2, load_ckpt_path=None)
```

- Use `LCVMUNet(input_channels=3, num_classes=2, load_ckpt_path=None)` and never
  call `load_from()` (the training scripts `torch.load(load_ckpt_path)` and fail
  without the checkpoint file).
- `LCVMUNet` does not expose `dims`, `d_state` or `win_size`; call the `CCViM`
  factory directly if those need overriding.
- Note the separate `CCViM-nuclei/` subtree: it has a different wrapper (two
  decoders, an `OrderedDict` output, ~48.2 M parameters by analytic estimate)
  and is not the architecture behind the table row.

## Benchmark interface

- Output: `LCVMUNet.forward` returns raw logits `[B, num_classes, H, W]`, and
  applies `torch.sigmoid` **only when `num_classes == 1`**. Use `num_classes=2`
  (the ISIC config value of 1 would return probabilities instead of logits).
  There is no deep supervision and no multi-scale output list.
- Input size: **H and W must be divisible by 32.** `PatchMerging2D` has an
  internal odd-size pad, but `LocalCluster` factors the feature map with
  `rearrange("b e (Wg w) (Hg h) -> (b Wg Hg) e w h", ...)` using
  `win_size=[8,4,2,1]` at the H/4 stage, which requires exact divisibility.
  Both 224 and 256 satisfy this. `MultiScanVSSM` sets its `token_size` from the
  input on every forward, and there is no hardcoded resize.
- Normalization: **`nn.LayerNorm` only** in the default path (`norm_layer="LN"`;
  a `"bn"` option exists but is unused) — batch size 1 is safe.
- Call `model.eval()` to disable `DropPath(drop_path_rate=0.2)`.

## Dependencies: CUDA kernel required before the model can even be imported

- `models/CCViMUNet/CCViM.py` performs a **hard top-level import**:
  `try: from selective_scan_vmamba_pt202 import selective_scan_cuda_core` /
  `except: import selective_scan_cuda`. The `except` branch is unguarded, so
  without a built `selective_scan_cuda` the model cannot be imported or
  constructed at all. There is no working pure-PyTorch fallback (the
  `selective_scan_ref` import is commented out).
- Good news: the `SSMODE="mamba_ssm"` path calls
  `selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias,
  delta_softplus)`, which matches the `mamba-ssm` kernel signature, so a
  CUDA-enabled `mamba-ssm` is sufficient and the VMamba kernels are optional.
- Selective scan is **CUDA-only**; there is no CPU forward path.
- The repository pins `torch==1.13.0+cu117`, `triton==2.0.0`,
  `causal_conv1d==1.0.0`, `mamba_ssm==1.0.1`, `timm==0.4.12`. For
  torch 2.5.1+cu121: those wheels do not exist; current `mamba-ssm` no longer
  builds the kernel by default and needs
  `MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm --no-build-isolation`.
  Do **not** install the pinned `triton==2.0.0` alongside torch 2.5 (it needs
  triton 3.x), and `timm==0.4.12` is unnecessary. `torch.cuda.amp.custom_fwd` /
  `custom_bwd` used in `CCViM.py` are deprecated on torch >= 2.4.
- No mmdetection/mmsegmentation; dependencies are torch, timm, einops plus the
  CUDA kernels.
- Construction is otherwise offline; no weights are auto-downloaded.

## Measurement caveats

- Report `sum(p.numel() for p in model.parameters())` as the parameter count,
  **not** THOP's `params` field, which silently drops the `nn.Parameter`s held
  directly by the non-leaf `SS2D` (5,389,824) and `LocalCluster` (30).
- THOP also undercounts MACs for the same structural reason: the selective scan
  runs inside a custom autograd `Function` with no module wrapper, and the
  `x_proj`/`dt_proj` projections, the `LocalCluster` similarity matmuls,
  `scatter_`, `flip` and `rearrange` are all invisible to module hooks.
- The authors' fvcore-based `selective_scan_flop_jit` rules and `VSSM.flops()`
  exist in the file but are **dead code** (the `fvcore` import is commented
  out), so they cannot be used as-is.
- `torch.inference_mode()` is risky for the same reason as other Mamba models:
  the custom `SelectiveScan` function calls `ctx.save_for_backward(...)` on
  tensors created in inference mode. Test before adopting, or use
  `torch.no_grad()`. Note this affects `benchmark.py`, which currently uses
  `torch.inference_mode()`.
- `thop` master still imports `distutils`, so it needs Python <= 3.11 or a
  patched thop.

## Open items

- Confirm the upstream revision/commit used for the benchmark and record it.
- Decide how to report this row: the true 32.16 M, or 23.61 M labelled as the
  authors' THOP-reported value on a `BiAttn`-free revision. This is a manuscript
  decision.
- The 4.57 GFLOPs figure could not be traced to a primary source; treat it as
  unverified until re-measured.
