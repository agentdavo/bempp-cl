from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    plt = None

from bempp_audio.viz.style import DEFAULT_PLOT_CONFIG, setup_freq_axis as _setup_freq_axis


def render_isobars(
    ax,
    *,
    frequencies_hz: np.ndarray,
    angles_symmetric_deg: np.ndarray,
    spl_symmetric_db: np.ndarray,
    levels_db: list[float] = None,
    linewidth: float = 1.0,
    color: str = "k",
) -> object:
    """
    Render isobar contours (dB-down angles) vs frequency.

    Inputs are typically from a normalized symmetric polar map where 0 dB is the
    reference angle response.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    freqs = np.asarray(frequencies_hz, dtype=float)
    angles = np.asarray(angles_symmetric_deg, dtype=float)
    spl = np.asarray(spl_symmetric_db, dtype=float)

    if levels_db is None:
        levels_db = [-3.0, -6.0, -10.0, -20.0]
    # Matplotlib requires strictly increasing contour levels.
    # Default "dB down" conventions are often specified in descending order
    # (e.g. [-3, -6, -10, -20]), so sort defensively.
    levels = sorted({float(l) for l in levels_db})
    if len(levels) < 2:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Isobars require 2+ contour levels", ha="center", va="center")
        return ax

    if freqs.size < 2 or angles.size < 2:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Isobars require a frequency sweep", ha="center", va="center")
        return ax

    F, A = np.meshgrid(freqs, angles)
    # Filter to levels that fall within the data range; otherwise matplotlib can raise.
    with np.errstate(all="ignore"):
        zmin = float(np.nanmin(spl))
        zmax = float(np.nanmax(spl))
    if not (np.isfinite(zmin) and np.isfinite(zmax)):
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Isobars unavailable (non-finite polar data)", ha="center", va="center")
        return ax
    if not (zmax > zmin):
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Isobars unavailable (flat polar data)", ha="center", va="center")
        return ax
    levels_in = [l for l in levels if zmin <= float(l) <= zmax]
    if len(levels_in) < 2:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Isobars unavailable (levels out of range)", ha="center", va="center")
        return ax

    cs = ax.contour(F, A, spl, levels=levels_in, colors=color, linewidths=float(linewidth))
    ax.clabel(cs, fmt=lambda v: f"{v:.0f} dB", fontsize=DEFAULT_PLOT_CONFIG.font_size_tick)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Isobar Contours")
    _setup_freq_axis(ax)
    ax.grid(True, which="both", alpha=0.25)
    return ax
