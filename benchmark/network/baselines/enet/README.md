# ENet

The implementation in `model.py` is adapted from
[davidtvs/PyTorch-ENet](https://github.com/davidtvs/PyTorch-ENet), specifically
`models/enet.py`.

The upstream repository is an MIT-licensed PyTorch implementation ported from
the authors' Torch ENet implementation. The benchmark adapter keeps the ENet
topology and returns logits with shape `[B, n_classes, H, W]`.
