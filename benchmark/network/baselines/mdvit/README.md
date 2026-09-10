# MDViT

## Manuscript entry

- Citation: S. Du, N. Bayasi, G. Hamarneh, R. Garbi, "MDViT: Multi-domain vision
  transformer for small medical image segmentation datasets", MICCAI 2023,
  pp. 448-458 (`\cite{du2023mdvit}`).
- Complexity-table fingerprint (`tab:complexity`): **38.78 M** parameters,
  **9.31 GFLOPs**, 14.79 ms.
- **The 38.78 M fingerprint does not reproduce from the official code under any
  configuration found.** The MDViT paper (Table 2) states **28.5 M** for MDViT
  and **27.8 M** for BASE, and says parameters are reported "at inference time"
  (the domain peers are discarded after training). An independent enumeration
  from the released source reproduces those exactly:
  - universal network (BASE + domain adaptation) = **28.537 M**
  - BASE (`adapt_method=None`) = **27.753 M**
  - as actually constructed by the repository (4 domains, `decoder_name='MLPFM'`,
    `adapt_method='Sup'`, i.e. 4 peers at about 1.610 M each) = **34.976 M**
  - other variants: MLP peers about 32.7 M, `MDViT_DSN` about 35.0 M
  - **38.78 M matches none of these.** This row needs an author-side check
    before it can be stated or defended.
- The paper has **no FLOPs column**. An independent THOP-style count at
  256 x 256 gives about **8.34 G** with peers idle and about **13.07 G** with one
  peer active, versus the tabulated 9.31 G, which matches neither.

## Upstream source

- Repository: <https://github.com/siyi-wind/MDViT> (official; the owner handle
  `siyi-wind` is the first author and the arXiv abstract links it).
- **License: there is no LICENSE or COPYING file** and the README has no license
  section. Worse, the repository vendors third-party code:
  `Models/Transformer/mpvit.py` carries an ETRI copyright header declaring dual
  licensing (GPL-3.0 or commercial) with reference to a LICENSE file that is
  **absent** from this repository, and it also bundles Swin-Unet / UTNet /
  TransFuse / DeiT / DeepLabV3+ snippets. Licensing is ambiguous: flag before
  redistributing anything from this directory.
- Model files:
  - `Models/Transformer/mdvit.py` -> `class MDViT`, `class MDViT_DSN`
  - `Models/Decoders.py` -> `UnetDecodingBlockTransformer`, `MLPDecoderFM`,
    `MLPDecoder`, `DeepLabV3Decoder`
  - `Models/Transformer/mpvit.py` -> `ConvPosEnc`, `ConvRelPosEnc`, `Conv2d_BN`,
    `Mlp`

## Configuration and interface

Signature:

```python
MDViT(img_size=512, in_chans=3, num_stages=4, num_layers=[2,2,2,2],
      embed_dims=[64,128,320,512], mlp_ratios=[8,8,4,4], num_heads=[8,8,8,8],
      qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
      drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
      conv_norm=nn.BatchNorm2d, adapt_method=None, num_domains=4,
      decoder_name='MLPFM', **kwargs)
```

- The authors construct it (`multi_train_MDViT.py`) as
  `MDViT(img_size=config.data.img_size, drop_rate=0.1, drop_path_rate=0.1,
  conv_norm=nn.BatchNorm2d, adapt_method=..., num_domains=K,
  decoder_name='MLPFM')` with `K = 4`, and `Configs/multi_train_local.yml` sets
  `img_size: 256`. There is **no `num_classes` argument anywhere**.
- Input size: no assertions. `img_size` feeds only `seq_length`, which is unused
  inside `FactorAtt_ConvRelPosEnc_Sup`; positions come from convolutional
  position encoding, so the input size is free. Multiples of 32 are recommended
  (stem /4, then three stride-2 patch embeddings).
- Normalization: LayerNorm (`eps=1e-6`) in every transformer block, plus
  **`nn.BatchNorm2d`** in the stem, the patch embeddings, the bridge, each
  decoder `conv_after` and each peer's MLP `linear_fuse`. `MDViT_DSN` replaces
  each BatchNorm with a per-domain `nn.ModuleList` of which only one member
  executes. Batch size 1 is fine in `eval()`.
- **Output is a Python list `[out, aux_out]`**: `out` is `[B, 1, H, W]`
  full-resolution raw logits, and `aux_out` is the peer branch selected by a
  **string** domain index `d` in `'0'..'3'`, otherwise `None`. The class count is
  hardcoded to 1. A wrapper is required, for example `model(x, d='0')[0]` plus a
  2-channel head (+65 parameters, so the parameter fingerprint is unaffected).
  With `out_feat=True` the forward returns a dict instead.

## Dependencies and adaptation

- Builds **fully offline**: no `timm.create_model`, no `pretrained=True`, no
  checkpoint load in the model code; weights come from in-code `trunc_normal_`
  initialisation. The only hardcoded checkpoint path is an author-local path in
  `Configs/multi_train_local.yml`, used only for `test.only_test`.
- Dependencies: torch + timm + einops + numpy. No mmcv/mmseg,
  no segmentation-models-pytorch, no CUDA kernel compilation.
- Gotcha: `Models/Decoders.py` begins with `from turtle import forward`, which
  **imports tkinter** and raises `ImportError` on Python builds without tkinter
  (common in slim conda/Docker images). Remove that import. `mdvit.py` and
  `Decoders.py` also `sys.path.append` an author-local path at import time
  (harmless).

## Measurement caveats

- THOP systematically undercounts this model: the core `torch.einsum` in the
  factorized attention has no hook, and the pervasive `F.interpolate` calls
  (including two full-resolution upsamples) and all softmax operations are not
  counted.
- The forward signature needs the domain argument, so a default
  `thop.profile(model, inputs=(x,))` profiles only the universal network. Use
  `thop.profile(model, inputs=(x, None, '0'))` to include one peer, and note that
  only 1 of the 4 peers executes, so MACs depend on the arbitrarily chosen
  domain. All 4 domain-BatchNorm members are counted in the parameter total
  (correct) while only one executes.
- `UnetDecodingBlockTransformer.forward` wraps its MHSA call in a
  `try/except` that can silently swallow shape or keyword errors, so a
  mis-specified wrapper may fail quietly rather than loudly.

## Open items

- Establish which configuration and which tool produced 38.78 M / 9.31 G, or
  re-measure. The paper's own numbers are 28.5 M / 27.8 M with no FLOPs column.
- Resolve the licensing question before redistributing this code.
- Confirm the upstream revision/commit used and record it here.
