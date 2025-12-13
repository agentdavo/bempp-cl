from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    plt = None

from bempp_audio.viz.style import DEFAULT_PLOT_CONFIG, setup_freq_axis as _setup_freq_axis


def render_normalized_offaxis_family(
    ax,
    response: "FrequencyResponse",
    *,
    polar: "PolarMapData",
    spl_ref_db: np.ndarray,
    norm_angle_used_deg: float,
    angles_deg: list[float],
    offset_db: float = 3.0,
    show_progress: bool = True,
    logger: Optional[object] = None,
) -> object:
    """
    Render a normalized off-axis family (waterfall-style) plot.

    Each curve is normalized to the chosen reference angle and vertically offset
    for readability.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    freqs = np.asarray(response.frequencies, dtype=float)
    spl_ref = np.asarray(spl_ref_db, dtype=float)

    spl_curves = polar.spl_curves_db(
        response,
        angles_deg=[float(a) for a in angles_deg],
        show_progress=bool(show_progress),
        logger=logger,
    )

    offset = float(offset_db)
    for j, angle in enumerate(angles_deg):
        curve = spl_curves[:, j] - spl_ref
        y = curve + (j * offset)
        ax.semilogx(freqs, y, linewidth=1.5, label=f"{float(angle):g}°")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"dB (norm {float(norm_angle_used_deg):g}°) + offset")
    ax.set_title("Normalized Off-Axis Family")
    _setup_freq_axis(ax)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=DEFAULT_PLOT_CONFIG.font_size_tick, ncol=2)
    return ax

