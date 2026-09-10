# U-Net baseline source

The U-Net implementation in `model.py` and `parts.py` is adapted from
[milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).

- Upstream files: `unet/unet_model.py` and `unet/unet_parts.py`
- Upstream license: GNU GPL-3.0
- Adaptation: package-local imports and benchmark-only integration; the model
  topology and default `bilinear=False` setting are unchanged.

If this benchmark code is redistributed, retain the upstream copyright and
GPL-3.0 license notice.
