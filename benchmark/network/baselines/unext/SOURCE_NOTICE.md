# UNeXt baseline source

Adapted from:

<https://github.com/jeya-maria-jose/UNeXt-pytorch>

Upstream model file:

<https://github.com/jeya-maria-jose/UNeXt-pytorch/blob/main/archs.py>

Upstream license: MIT License, Copyright (c) 2022 Jeya Maria Jose. Retain the
upstream copyright and license notice when redistributing this adapted
implementation.

Adaptations, relative to upstream:

- the constructor exposes `n_channels` / `n_classes`; upstream accepted
  `input_channels` but ignored it and hardcoded a 3-channel first convolution;
- unused imports were dropped (`mmcv.cnn.ConvModule`, `utils`, `torchvision`,
  `matplotlib`, `os`, `pdb`);
- `timm` is imported via `timm.layers` with a `timm.models.layers` fallback;
- the unused `UNext_S` variant is not included.
