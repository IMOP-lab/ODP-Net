# Benchmark model layout

```text
network/
├── odpnet/                 # ODP-Net and its private implementation modules
└── baselines/              # comparison models
    ├── unet/               # integrated
    ├── segnet/             # integrated
    ├── enet/               # integrated
    └── r2unet/             # integrated R2U-Net (half-width benchmark default)
```

Each model package should expose a PyTorch module with the same contract:

```python
model = Model(n_channels=3, n_classes=2, bilinear=False)
logits = model(x)  # x: [B, 3, H, W], logits: [B, 2, H, W]
```

To add a model:

1. Put its source files in a new directory under `network/baselines/`.
2. Add an `__init__.py` that exports the model class.
3. Export that class from `network/__init__.py`.
4. Add one factory branch and one `--model` choice in `benchmark.py`.

Keep the upstream source URL and license in the model's package directory.
