"""Model implementations used by the local benchmark scripts.

The four core models (ODP-Net, U-Net, SegNet, ENet) are imported eagerly. The
baselines that need optional third-party packages (``timm`` for UNeXt and
Polyp-PVT, ``torchvision`` for the PDAtt models) are resolved lazily through the
module ``__getattr__`` below, so that a missing optional dependency cannot break
benchmarking of the models that do not need it.
"""

from .odpnet.main import ODPNet, DP_CoNet
from .baselines.enet.model import ENet
from .baselines.segnet.model import SegNet
from .baselines.unet.model import UNet
from .baselines.r2unet.model import R2UNet

# name -> (module path relative to this package, attribute name)
_LAZY_MODELS = {
    "UNext": ("baselines.unext.model", "UNext"),
    "PAttUNet": ("baselines.pattunet.model", "PAttUNet"),
    "DAttUNet": ("baselines.dattunet.model", "DAttUNet"),
    "PolypPVT": ("baselines.polyp_pvt.model", "PolypPVT"),
    "MEWUNet": ("baselines.mewunet.model", "MEWUNet"),
}

__all__ = [
    "ODPNet",
    "DP_CoNet",
    "UNet",
    "R2UNet",
    "MEWUNet",
    "SegNet",
    "ENet",
    "UNext",
    "PAttUNet",
    "DAttUNet",
    "PolypPVT",
]


def __getattr__(name):
    """Import optional-dependency baselines on first access."""
    if name in _LAZY_MODELS:
        from importlib import import_module

        module_name, attribute = _LAZY_MODELS[name]
        module = import_module(f".{module_name}", __name__)
        value = getattr(module, attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
