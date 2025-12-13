"""
Matplotlib plot implementations, split by topic.

These modules provide plot implementations for use by panels and the report builder.
"""

from bempp_audio.viz.plots.basic import (
    spl_response,
    phase_response,
    bode_plot,
    group_delay,
    pressure_field_2d,
)
from bempp_audio.viz.plots.directivity import (
    polar_directivity,
    polar_directivity_multi,
    directivity_balloon_3d,
    directivity_metrics,
)
from bempp_audio.viz.plots.power import power_response

__all__ = [
    "spl_response",
    "phase_response",
    "bode_plot",
    "group_delay",
    "pressure_field_2d",
    "polar_directivity",
    "polar_directivity_multi",
    "directivity_balloon_3d",
    "directivity_metrics",
    "power_response",
]
