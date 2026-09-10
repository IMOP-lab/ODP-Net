# VM-UNet

## Manuscript entry

- Citation: J. Ruan, J. Li, S. Xiang, "VM-UNet: Vision Mamba UNet for medical
  image segmentation", ACM TOMM (2024) (`\cite{ruan2024vm}`).
- Complexity-table fingerprint (`tab:complexity`): **22.04 M** parameters,
  **4.11 GFLOPs**, 8.12 ms.
- **The 22.04 M fingerprint is a THOP artifact, not a true parameter count.**
  Recomputed from the upstream source at the paper configuration:
  - true `sum(p.numel())` = **27,427,561 (27.43 M)**; with `num_classes=2`
    instead of the paper's 1, 27,427,586.
  - parameters visible to THOP's leaf-module hooks = **22,037,737 (22.04 M)**,
    matching the manuscript exactly.
  - the 5,389,824-parameter gap is SS2D's own `nn.Parameter` set
    (`x_proj_weight`, `dt_projs_weight`, `dt_projs_bias`, `A_logs`, `Ds`), which
    is attached to a non-leaf module and is therefore missed by THOP.
- Consequence for this benchmark: `parameters_million` and
  `thop.thop_params` will disagree for this model by design. Both are recorded;
  the table column reproduces `thop.thop_params`.

## Upstream source

- Repository: <https://github.com/JCruan519/VM-UNet> (official; the arXiv
  abstract of 2402.02491 links to it, and the repository owner handle
  `JCruan519` is the first author).
- License: Apache License 2.0.
- Model files:
  - `models/vmunet/vmunet.py` -> `class VMUNet(nn.Module)`
  - `models/vmunet/vmamba.py` -> `class VSSM`, `class SS2D`, `class VSSBlock`,
    `class VSSLayer`, `class VSSLayer_up`, `PatchEmbed2D`, `PatchMerging2D`,
    `PatchExpand2D`, `Final_PatchExpand2D`

## Configuration: the code defaults are NOT the paper model

This is the single most important integration detail.

- Constructor:
  `VMUNet(input_channels=3, num_classes=1, depths=[2,2,9,2], depths_decoder=[2,9,2,2], drop_path_rate=0.2, load_ckpt_path=None)`
- `configs/config_setting.py` sets the configuration actually used for the paper:
  `depths=[2,2,2,2]`, `depths_decoder=[2,2,2,1]`, `num_classes=1`,
  `input_channels=3`, `drop_path_rate=0.2`, `input_size_h/w = 256`.
- Instantiate with the paper configuration, otherwise the parameter count is
  wrong by a large margin (code defaults give 44.27 M, not 27.43 M):

```python
VMUNet(input_channels=3, num_classes=2,
       depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1], drop_path_rate=0.2)
```

## Benchmark interface

- Output: `VMUNet.forward` repeats 1-channel input to 3 and then returns
  `logits`; it applies `torch.sigmoid` **only when `num_classes == 1`**. With
  `num_classes=2` it returns raw logits `[B, 2, H, W]`, which satisfies this
  benchmark's contract directly.
- Input size: no explicit assertion, but the constraint is structural.
  `PatchMerging2D.forward` prints `Warning, x.shape ... is not match even` and
  then truncates when H or W is odd, and the decoder skip connections are
  size-exact additions (`x = layer_up(x + skip_list[-inx])`), so a non-multiple
  of 32 raises a shape error. **Require H, W == 0 (mod 32).** Both 224 and 256
  satisfy this, so the 224/256 question can be measured on this model without
  changing the architecture.
- Normalization: `nn.LayerNorm` only; no BatchNorm or InstanceNorm anywhere, so
  batch size 1 is safe.
- SS2D hard-casts the scan to fp32 and asserts on dtype, so autocast will not
  speed it up and will trip the dtype assertions.

## Dependencies and the CUDA kernel requirement

- Pins from the upstream README: python 3.8, `torch==1.13.0`,
  `timm==0.4.12`, `triton==2.0.0`, `causal_conv1d==1.0.0`, `mamba_ssm==1.0.1`,
  plus `thop`, `einops`, `medpy`, `SimpleITK`.
- `vmamba.py` imports the selective scan inside a bare
  `try: ... except: pass`. If the compiled `selective_scan_cuda` is unavailable,
  the import failure is swallowed and the forward then dies with
  `NameError: selective_scan_fn`. **VM-UNet cannot run on its default path
  without the compiled CUDA selective-scan kernel**; there is no wired-in
  Triton or pure-PyTorch fallback (unlike VMamba).
- `mamba_ssm` and `causal_conv1d` are sdist-only on PyPI for every released
  version, and no official wheel targets torch 2.5. Their `setup.py` guesses a
  prebuilt wheel for the exact (CUDA, torch, python, abi) tuple and otherwise
  compiles locally, so on torch 2.5.1+cu121 expect a source build against the
  local nvcc. This is the main environment risk for this baseline.
- Model construction itself is offline; `load_ckpt_path=None` by default.
  The upstream `train.py` initialises from the VMamba-Small ImageNet-1k EMA
  checkpoint, distributed via Baidu Netdisk / Google Drive rather than GitHub
  Releases. No weights are needed for an architecture-only benchmark.

## THOP caveats specific to this model

- MACs are also undercounted: the selective scans and the `x_proj`/`dt`
  `torch.einsum` projections are invisible to THOP's module hooks, so 4.11 G
  understates the real cost roughly two-fold (reported figure ~4.11 G vs an
  approximate 8.3 G; the scan term alone is ~2.96 G MACs at 256 x 256).
- For Mamba-family models the literature-standard counter is **fvcore** with an
  explicit `prim::PythonOp.SelectiveScanCuda` JIT handler, or the analytic
  `9*B*L*D*N` formula; VMamba's own repository uses fvcore for this reason.
  Keep THOP only if the whole table must share one counter, and label it.
- `torch.inference_mode()` is expected to fail on this model:
  `SelectiveScanFn.forward` calls `ctx.save_for_backward(...)` on tensors created
  inside inference mode, which PyTorch rejects. **Use `torch.no_grad()` for the
  latency and memory loops.** This currently affects `benchmark.py`, which uses
  `torch.inference_mode()` in `cuda_latency`, `peak_memory` and `thop_counts`.

## Open items to confirm on the workstation

- Confirm the exact upstream revision/commit used, and record it here.
- Verify that the reported true parameter count is 27,427,586 with
  `num_classes=2` at the paper configuration.
- Decide how the revision reports this row: the true 27.43 M, or 22.04 M
  labelled as the authors' THOP-reported value. This is a manuscript decision,
  not a benchmark decision.
