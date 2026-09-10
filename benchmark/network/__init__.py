"""Model implementations used by the local benchmark scripts."""

from .odpnet.main import ODPNet, DP_CoNet
from .baselines.enet.model import ENet
from .baselines.segnet.model import SegNet
from .baselines.unet.model import UNet

__all__ = ["ODPNet", "DP_CoNet", "UNet", "SegNet", "ENet"]
