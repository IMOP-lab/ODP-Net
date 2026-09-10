# Baseline model organization

Each subdirectory contains one baseline model used in the manuscript's
complexity comparison. Keep the upstream implementation and source notice
inside the corresponding model directory.

Every integrated model should expose a constructor and forward interface
compatible with:

```python
model = Model(n_channels=3, n_classes=2, bilinear=False)
logits = model(x)  # x: [B, 3, H, W], logits: [B, 2, H, W]
```

Current model groups:

- `unet/`, `segnet/`, `enet/`, `r2unet/`: classic convolutional baselines
- `mewunet/`: frequency-domain multi-axis representation baseline
- `unext/`, `mewunet/`, `pattunet/`, `dattunet/`: modern CNN/U-Net variants
- `polyp_pvt/`, `mdvit/`: Transformer or hybrid attention models
- `vm_unet/`, `vmamba/`, `ccvim/`: state-space or visual state-space models

## Manuscript complexity-table cross-check

The values below are the manuscript's `tab:complexity` entries (stated as a
256 x 256 three-channel benchmarking condition). "True params" is
`sum(p.numel())`, which is independent of the input size; "THOP params" is the
`params` field returned by `thop.profile`. Each model's own README records the
evidence. The only rows not yet verified are SegNet and ENet, which are already
integrated and can be measured directly.

| Model | Paper Params (M) | Paper GFLOPs | True params | Status |
|---|---|---|---|---|
| U-Net | 31.04 | 54.74 | 31.0377 M (measured) | matches exactly |
| SegNet | 29.44 | 40.17 | pending measurement | to verify |
| ENet | 0.36 | 0.59 | pending measurement | to verify |
| R2U-Net | 9.78 | 38.38 | 39.09 M full width; 9.78 M half width | matches a half-width variant only |
| UNeXt | 1.47 | 0.57 | 1.4719 M | matches exactly (both columns, at 256^2) |
| MEWUNet | 138.89 | 40.52 | 140.27 M (THOP reports 138.89 M) | reproduces THOP only; GFLOPs cell unresolved |
| PAttUNet | 11.17 | 16.84 | 11.1739 M | matches exactly (both columns) |
| DAttUNet | 11.25 | 24.73 | 11.2496 M | matches exactly (returns a tuple) |
| PolypPVT | 25.11 | 5.30 | 25.1076 M | matches exactly; GFLOPs is a rescaled 352^2 value |
| MDViT | 38.78 | 9.31 | 28.5 M per paper; 34.98 M as built | row unreproducible |
| VM-UNet | 22.04 | 4.11 | 27.43 M (THOP reports 22.04 M) | reproduces THOP only |
| VMamba | 22.04 | 4.11 | no official config matches | row unreproducible |
| CCViM | 23.61 | 4.57 | 32.16 M (THOP reports 26.77 M) | matches THOP on an earlier revision |
| ODP-Net | 139.51 | 313.03 | 207.34 M (measured) | gap unexplained |

### What the cross-check has established so far

1. **The `Params (M)` column is a THOP-reported quantity, not
   `sum(p.numel())`.** Three independent rows reproduce their table value only
   through THOP:
   - VM-UNet: 22.04 M visible vs 27.43 M true (SS2D's own `nn.Parameter`s)
   - CCViM: 23.61 M vs 26.77 M (THOP) vs 32.16 M true
   - MEWUNet: 138.89 M vs 140.27 M true, and the difference is exactly the two
     THOP blind spots — `nn.GroupNorm` has no counting rule at all (544,320
     parameters) and the `MEW` module holds 829,440 parameters as raw
     `nn.Parameter` on a custom module: 140,268,294 - 544,320 - 829,440 =
     138,894,534
   THOP's `profile()` accumulates parameters only through hooks registered for
   module types it knows, so unmatched custom modules and every `nn.GroupNorm`
   are silently dropped. The CCViM repository's own
   `utils.cal_params_flops(model, 256, logger)` is a `thop.profile` call at
   256 x 256, which corroborates the provenance. Re-measuring with
   `sum(p.numel())` will change several rows.
2. **This does not by itself explain ODP-Net.** Every ODP-Net parameter belongs
   to a leaf module (there are no free-standing `nn.Parameter` objects in
   `network/odpnet/`), so THOP would report approximately 207.28 M for the
   current model, not 139.51 M. The ODP-Net gap therefore points to a different
   architecture revision rather than to a counting artifact.
3. **`GFLOPs` is a THOP `gmacs` figure** (one operation per
   multiply-accumulate) **measured at 256 x 256**. Two independent confirmations:
   U-Net's 41.9118 GMACs at 224 x 224 scales to 54.74 at 256 x 256, and
   PAttUNet / DAttUNet reproduce both of their columns exactly at 256 x 256
   (11,173,978 / 16.839 G and 11,249,564 / 24.730 G); at 224 x 224 they give
   only about 12.9 G and 18.7 G. Use `thop_gmacs`, never `thop_gflops_2x`, for
   this column.
4. **The counter is not applied consistently across rows, which silently
   favours some models.** THOP counts `nn.Upsample` (as 11x output elements for
   bilinear) but has no hook for functional `F.interpolate`; it counts
   `BatchNorm` (as 4x input elements) but has no rule for `GroupNorm`. The PDAtt
   models use `nn.Upsample` and are counted, MEW-UNet uses `F.interpolate` and
   is not, and MEW-UNet's 36 GroupNorm-bearing blocks contribute zero. Any
   revision that keeps a THOP-based column should state these asymmetries.
5. **Not every GFLOPs entry is a measurement at 256 x 256.** Polyp-PVT's 5.30 G
   equals the widely cited 352 x 352 value (10.005 G, a third-party number; the
   authors publish none) rescaled by (256/352)^2 = 0.5289, while an independent
   THOP count at 256 x 256 gives about 5.77 G. Polyp-PVT is canonically trained
   and benchmarked at 352 x 352, so this entry mixes resolutions. (UNeXt's 0.57 G
   was checked separately and does correspond to 256 x 256, so that row is
   consistent.)
6. **Four rows do not reproduce against the cited architecture or any
   configuration of the official code:**
   - MDViT 38.78 M: the paper states 28.5 M, and the released code as
     constructed gives 34.98 M
   - VMamba 22.04 M / 4.11 G: identical to the VM-UNet row, and no official
     VMamba configuration produces that pair
   - MEWUNet 40.52 G: THOP gives 39.51 G, and adding GroupNorm MACs gives
     40.07 G; no configuration reproduces 40.52 G
   - R2U-Net 9.78 M / 38.38 G: these are the values of a **half-width (k=0.5)**
     R2U-Net, not of the cited Alom 2019 architecture, which has 39.09 M
     parameters (and about 152.5 G THOP MACs at 256 x 256). R2U-Net also has no
     author-released implementation at all.
   Note that the first three are counting/config discrepancies; the R2U-Net row
   is an architecture-identity discrepancy of a different kind.
7. **THOP is not a valid counter for the Mamba/SSM rows.** The selective scan is
   a custom CUDA autograd function with no module wrapper, and the SSM
   projections use `torch.einsum`, so both are invisible to module hooks. The
   literature-standard counter for this family is `fvcore` with a
   `prim::PythonOp.SelectiveScanCuda` JIT handler, or the analytic
   `9*B*L*D*N` term. If some rows keep THOP and others move to `fvcore`, the
   table is no longer internally consistent and that must be stated explicitly.
8. **Three upstream sources restrict redistribution.** Polyp-PVT and PDAtt-Unet
   have no LICENSE file at all (Polyp-PVT's README grants research/education use
   only; PDAtt-Unet grants nothing explicitly, so the default is all rights
   reserved), and the MDViT repository has no LICENSE file while vendoring
   dual-licensed (GPL-3.0 or commercial) ETRI MPViT code. Check all three before
   this benchmark code is redistributed. MEW-UNet (Apache-2.0), VMamba (MIT),
   VM-UNet (Apache-2.0), CCViM (Apache-2.0) and UNeXt (MIT) are unambiguous.
