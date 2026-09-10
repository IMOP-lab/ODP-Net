"""Compatibility entry point for the ODP-Net model.

The importable implementation is kept in ``network/main.py`` because a
hyphenated filename cannot be imported with normal Python syntax.  This file
is retained to mirror the manuscript-oriented folder layout requested for the
benchmark package.
"""

from .main import (  # noqa: F401
    DP_CoNet,
    DifferentialOperatorPriorsAttention,
    FrequencyDomainPropagatorAttention,
    GloballyModulatedDiffusion,
    ODPNet,
)

__all__ = [
    "ODPNet",
    "DP_CoNet",
    "GloballyModulatedDiffusion",
    "FrequencyDomainPropagatorAttention",
    "DifferentialOperatorPriorsAttention",
]
