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
- `unext/`, `mewunet/`, `pattunet/`, `dattunet/`: modern CNN/U-Net variants
- `polyp_pvt/`, `mdvit/`: Transformer or hybrid attention models
- `vm_unet/`, `vmamba/`, `ccvim/`: state-space or visual state-space models
