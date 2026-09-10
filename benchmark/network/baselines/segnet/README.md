# SegNet

The implementation in `model.py` is adapted from
[vinceecws/SegNet_PyTorch](https://github.com/vinceecws/SegNet_PyTorch),
specifically the `Pavements/SegNet.py` architecture.

Required benchmark interface:

```python
model = SegNet(n_channels=3, n_classes=2)
logits = model(x)
```

The upstream model applies a final softmax. This benchmark adapter returns
logits so that all models expose the same segmentation interface; THOP,
parameter, and tensor-shape measurements are otherwise architecture-based.
Set `apply_softmax=True` only when reproducing the upstream probability-output
behavior.

Record the exact upstream revision and license if the source is redistributed.
