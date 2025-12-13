from __future__ import annotations

import numpy as np

try:
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

from bempp_audio.viz.data.metrics import beamwidth_vs_frequency, directivity_index_vs_frequency
from bempp_audio.viz.style import DEFAULT_PLOT_CONFIG, setup_freq_axis as _setup_freq_axis


def render_di_panel(
    ax,
    response: "FrequencyResponse",
    *,
    di_angles_deg: list[float] = None,
    level_db: float = -6.0,
    plane: str = "xz",
) -> object:
    """
    Render directivity index curves and beamwidth (twin axis).
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    freqs = np.asarray(response.frequencies, dtype=float)
    if di_angles_deg is None:
        di_angles_deg = [0.0, 10.0, 20.0]

    di_colors = ["k", "#404040", "#808080"]
    di_styles = ["-", "--", "--"]
    di_widths = [2.5, 1.5, 1.5]

    for angle, color, style, width in zip(di_angles_deg, di_colors, di_styles, di_widths):
        _, di = directivity_index_vs_frequency(response, angle_deg=float(angle))
        label = "on-axis" if float(angle) == 0.0 else f"{float(angle):g}°"
        ax.semilogx(freqs, di, color=color, linestyle=style, linewidth=width, label=label)

    ax_bw = ax.twinx()
    _, bw = beamwidth_vs_frequency(response, level_db=float(level_db), plane=plane)
    ax_bw.semilogx(freqs, bw, "b:", linewidth=1.5, alpha=0.7, label=f"{level_db:.0f}dB BW")
    ax_bw.set_ylabel("Beamwidth (degrees)", color="b")
    ax_bw.tick_params(axis="y", labelcolor="b", labelsize=DEFAULT_PLOT_CONFIG.font_size_tick)
    ax_bw.set_ylim([0, 180])

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Directivity Index (dB)")
    ax.set_title("Directivity Index")
    _setup_freq_axis(ax)

    ax.set_ylim([0, 24])
    ax.set_yticks([0, 3, 6, 9, 12, 15, 18, 24])
    ax.set_yticklabels(["0", "3", "6", "9", "12", "15", "18", "24"])

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=DEFAULT_PLOT_CONFIG.font_size_tick)
    ax_bw.legend(loc="upper right", fontsize=DEFAULT_PLOT_CONFIG.font_size_tick)
    return ax
