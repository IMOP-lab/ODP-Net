# R2U-Net

The benchmark adapter is now available as `network.baselines.r2unet.model`.
It follows the public `navamikairanda/R2U-Net` implementation and removes
training/dataset code so that the architecture can be profiled offline.

The command-line benchmark uses `base_channels=32` by default.  This is the
half-width configuration (32/64/128/256/512) and gives approximately 9.78M
parameters, matching the current manuscript complexity row.  The upstream
default is full width (64/128/256/512/1024), which is available by constructing
`R2UNet(base_channels=64)` directly and has approximately 39.09M parameters.

## Manuscript entry

- Citation: M. Z. Alom, C. Yakopcic, M. Hasan, T. M. Taha, V. K. Asari, "Recurrent
  residual U-Net for medical image segmentation", Journal of Medical Imaging 6,
  014006 (2019) (`\cite{alom2019recurrent}`).
- Complexity-table fingerprint (`tab:complexity`): **9.78 M** parameters,
  **38.38 GFLOPs**, 6.75 ms.
- **The fingerprint does not correspond to the architecture the citation
  describes.** The canonical R2U-Net (channel widths 64/128/256/512/1024, `t=2`)
  has **39,091,458 parameters** (3 input channels, 2 classes), matching the
  39.091 M figure cited in the literature. The tabulated 9.78 M corresponds
  instead to a **half-width (`k=0.5`) variant** with widths 32/64/128/256/512:
  - 9,781,762 parameters (Dual Cross-Attention port style)
  - 9,778,818 parameters (LeeJunHyun port style)
  - the two half-width candidates are numerically indistinguishable in both
    parameter count and convolution MACs
- The GFLOPs entry corroborates the half-width reading: an analytically
  THOP-faithful count of the half-width model at 256 x 256 gives 38.128 G
  (convolution + linear) + 0.112 G (BatchNorm) + 0.043 G (`nn.Upsample`) =
  **38.28 G**, against the tabulated 38.38 G (+0.26%). The full-width model would
  be about **152.5 G** at 256 x 256.
- The only published source found that reports exactly 9.78 M for R2U-Net is
  Ates et al., "Dual Cross-Attention" (arXiv:2303.17696, Table 1). That work
  trains at 224 x 224, so the tabulated 38.38 G was probably not copied from it
  but measured independently at 256 x 256.
- **Decision required:** either disclose that a half-width (0.5x) R2U-Net was
  measured, or replace the row with canonical figures (about 39.09 M
  parameters, about 152.5 G THOP MACs at 256 x 256). This is a manuscript
  decision, not a benchmark decision.

## Upstream source

**There is no author-released implementation.** The arXiv page (1802.06955)
carries no code link, and code aggregators list only third-party ports, none
owned by the authors. Treat "no official code release" as the finding. (The
Alom et al. PDF could not be retrieved, so "the paper itself contains no code
link" is unverified; "no official repository is discoverable" is medium
confidence.)

Candidate implementations:

| Repository | License | Relevant interface | Result |
|---|---|---|---|
| <https://github.com/gorkemcanates/Dual-Cross-Attention> | MIT (c) 2023 Gorkem Can Ates | `R2Unet(attention=False, n=1, in_features=3, out_features=3, k=0.5, input_size=(512,512), patch_size=8, ..., device='cuda')` | **reproduces the fingerprint**: 9,781,762 params, 38.28 G at 256 x 256 |
| <https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets> | MIT (c) 2019 Malav Bateriwala | `R2U_Net(img_ch=3, output_ch=1, t=2)`, internal `n1 = 64` hardcoded | canonical width: 39,091,458 params |
| <https://github.com/LeeJunHyun/Image_Segmentation> | **no LICENSE file -> all rights reserved** | `R2U_Net(img_ch=3, output_ch=1, t=2)` | canonical width: 39,091,458 params |
| <https://github.com/yingkaisha/keras-unet-collection> | MIT (c) 2020 Yingkai Sha | Keras/TensorFlow reference | canonical width |

Notes:
- `LeeJunHyun/Image_Segmentation` is unlicensed and its README states the
  repository is no longer updated, so it should **not** be vendored.
- `bigmb/.../Models.py` diverges semantically from the paper: its
  `Recurrent_block.forward` computes `conv(x + x)` rather than the residual
  accumulation `conv(x + x1)`. Shapes and MACs are identical, so it is usable for
  a complexity benchmark but not for reproducing the method.
- The DCA repository is the only candidate that reproduces the tabulated
  parameter count.

## Configuration and interface

- Full-width ports: `R2U_Net(img_ch=3, output_ch=2)`; the argument names are
  `img_ch` / `output_ch`, so a wrapper is needed for the benchmark's
  `n_channels` / `n_classes` contract.
- Half-width (DCA): `R2Unet(in_features=3, out_features=2, k=0.5, device='cpu')`.
  The default `out_features=3` must be overridden.
- Input size: nothing is hardcoded, but 4 `MaxPool2d(2,2)` stages, 4
  `nn.Upsample(scale_factor=2)` stages and `torch.cat` skip connections require
  **H and W divisible by 16**. Both 224 and 256 are fine. The DCA port's
  `input_size` / `patch_size` arguments only matter when `attention=True`.
- Normalization: `nn.BatchNorm2d` only, inside every convolutional block
  including the re-used recurrent convolution. The same BatchNorm module
  therefore runs `t+1 = 3` times per recurrent block, so batch size 1 must run in
  `eval()` to use running statistics.
- Output: a single logits tensor `[B, output_ch, H, W]` from a final 1x1
  convolution. No softmax, no multi-scale outputs, no deep supervision.
- Dependencies: **torch only** for the bigmb and LeeJunHyun ports. The DCA port
  additionally needs its own `model/utils/dca.py` and `main_blocks.py`. Nothing
  compiles CUDA kernels, and no pretrained weights are required or downloaded:
  the model builds fully offline.

## Specific hazard in the DCA port

`R2Unet` is constructed with `device='cuda'` by default and then calls
`torch.cuda.set_enabled_lms(True)`. That API exists **only in IBM's LMS-patched
PyTorch** (<https://github.com/IBM/pytorch-large-model-support>), so on stock
torch 2.5.1 construction raises `AttributeError`. Pass `device='cpu'` or delete
that line. (Documentation-level evidence; not executed here.)

## Measurement caveats

- The model is THOP-friendly: only `Conv2d`, `BatchNorm2d` and `nn.Upsample`
  are parameterized or counted, with no attention, `einsum`, FFT or custom
  autograd.
- The Python recurrence re-applies the same `nn.Conv2d` module `t+1` times per
  forward, and THOP's forward hooks fire on each application, so the recurrent
  MACs are counted correctly rather than once.
- THOP counts `nn.Upsample` but not `F.interpolate`, which matters when
  comparing this row against MEW-UNet and VM-UNet.

## Open items

- Establish which implementation and which width produced the tabulated row.
- Decide between disclosing the half-width configuration and replacing the row
  with canonical R2U-Net figures.
- Pick a redistributable source (MIT): the DCA port for the half-width variant,
  or the bigmb port for canonical width.
