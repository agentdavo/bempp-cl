"""
Power-related matplotlib plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    plt = None

from bempp_audio.viz.style import setup_style as _setup_style

if TYPE_CHECKING:
    from bempp_audio.results import FrequencyResponse


def power_response(
    response: "FrequencyResponse",
    ax=None,
    **kwargs,
):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        _setup_style()
        _, ax = plt.subplots(figsize=(10, 6))

    freqs, swl = response.sound_power_level_vs_freq()
    ax.semilogx(freqs, swl, **kwargs)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Sound Power Level (dB re 1 pW)")
    ax.set_title("Radiated Power")
    ax.grid(True, which="both", alpha=0.3)

    return ax

