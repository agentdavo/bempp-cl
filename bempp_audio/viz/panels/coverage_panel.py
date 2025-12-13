from __future__ import annotations

import numpy as np

try:
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

from bempp_audio.viz.data.metrics import beamwidth_vs_frequency
from bempp_audio.viz.style import setup_freq_axis as _setup_freq_axis


def render_coverage_uniformity(
    ax,
    *,
    frequencies_hz: np.ndarray,
    angles_deg: np.ndarray,
    spl_map_db: np.ndarray,
    coverage_angle_deg: float = 60.0,
    metric: str = "std",
) -> object:
    """
    Render a simple coverage-uniformity metric vs frequency.

    Parameters
    ----------
    spl_map_db:
        Typically a normalized (0 dB reference) map with shape (n_angles, n_freqs).
    coverage_angle_deg:
        Half-angle defining the coverage cone (±coverage_angle_deg).
    metric:
        "std" for standard deviation, or "ptp" for peak-to-peak within the cone.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    freqs = np.asarray(frequencies_hz, dtype=float)
    angles = np.asarray(angles_deg, dtype=float)
    spl = np.asarray(spl_map_db, dtype=float)

    if spl.shape != (angles.size, freqs.size):
        raise ValueError(f"Expected spl_map_db shape {(angles.size, freqs.size)}; got {spl.shape}")

    mask = np.abs(angles) <= float(coverage_angle_deg)
    if not np.any(mask):
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No angles within coverage cone", ha="center", va="center")
        return ax

    slice_ = spl[mask, :]
    metric = str(metric).lower().strip()
    if metric == "std":
        y = np.std(slice_, axis=0)
        ylabel = "Std Dev (dB)"
        title = f"Coverage Uniformity (±{float(coverage_angle_deg):g}°)"
    elif metric == "ptp":
        y = np.ptp(slice_, axis=0)
        ylabel = "Peak-to-Peak (dB)"
        title = f"Coverage Uniformity P-P (±{float(coverage_angle_deg):g}°)"
    else:
        raise ValueError("metric must be 'std' or 'ptp'")

    ax.semilogx(freqs, y, "k-", linewidth=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _setup_freq_axis(ax)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def render_pattern_control(
    ax,
    response: "FrequencyResponse",
    *,
    target_beamwidth_deg: float = 90.0,
    tolerance_deg: float = 10.0,
    level_db: float = -6.0,
    plane: str = "xz",
) -> object:
    """
    Render a simple pattern-control bandwidth chart using beamwidth targets.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    freqs, bw = beamwidth_vs_frequency(response, level_db=float(level_db), plane=plane)
    bw = np.asarray(bw, dtype=float)

    target = float(target_beamwidth_deg)
    tol = float(tolerance_deg)

    ax.semilogx(freqs, bw, "k-", linewidth=2, label=f"{level_db:.0f} dB BW")
    ax.axhline(target, color="b", linestyle="--", linewidth=1.5, label="Target")
    ax.fill_between(freqs, target - tol, target + tol, color="b", alpha=0.12, label="Tolerance")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Beamwidth (degrees)")
    ax.set_title("Pattern Control Bandwidth")
    _setup_freq_axis(ax)
    ax.set_ylim([0, 180])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    return ax

