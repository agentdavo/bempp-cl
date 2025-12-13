"""
Computation utilities for visualization.

This package contains data/metric helpers that are independent of any plotting backend.
"""

from bempp_audio.viz.data.metrics import (
    beamwidth_vs_frequency,
    directivity_index_vs_frequency,
)
from bempp_audio.viz.data.normalization import (
    normalize_spl_map_by_angle,
    symmetric_polar_map,
)
from bempp_audio.viz.data.polar_data import PolarMapData

__all__ = [
    "PolarMapData",
    "beamwidth_vs_frequency",
    "directivity_index_vs_frequency",
    "normalize_spl_map_by_angle",
    "symmetric_polar_map",
]

