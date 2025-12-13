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


def _sound_power_spl_equivalent_db(
    response: "FrequencyResponse",
    *,
    distance_m: float,
    ref_pa: float = 20e-6,
) -> np.ndarray:
    """
    SPL-equivalent from radiated power, for comparison to SPL curves at `distance_m`.
    """
    from bempp_audio.baffles import InfiniteBaffle

    freqs = np.asarray(response.frequencies, dtype=float)
    spl_power = np.zeros_like(freqs, dtype=float)
    for i, result in enumerate(response.results):
        power_w = float(result.radiated_power())
        if power_w <= 0:
            spl_power[i] = -np.inf
            continue
        solid_angle = 2.0 * np.pi if isinstance(getattr(result, "baffle", None), InfiniteBaffle) else 4.0 * np.pi
        p_rms_avg = np.sqrt((power_w * float(result.rho) * float(result.c)) / (solid_angle * float(distance_m) ** 2))
        spl_power[i] = 20.0 * np.log10(np.maximum(p_rms_avg, 1e-20) / float(ref_pa))
    return spl_power


def render_spl_panel(
    ax,
    response: "FrequencyResponse",
    *,
    polar: "PolarMapData",
    spl_ref_db: np.ndarray,
    norm_angle_used_deg: float,
    angles_deg: list[float],
    distance_m: float,
    ref_pa: float = 20e-6,
    show_progress: bool = True,
    logger: Optional[object] = None,
) -> object:
    """
    Render normalized SPL curves at multiple angles plus SPL-equivalent sound power.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    freqs = np.asarray(response.frequencies, dtype=float)
    spl_ref = np.asarray(spl_ref_db, dtype=float)

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(angles_deg)))
    spl_curves = polar.spl_curves_db(
        response,
        angles_deg=[float(a) for a in angles_deg],
        show_progress=bool(show_progress),
        logger=logger,
    )

    for j, (angle, color) in enumerate(zip(angles_deg, colors)):
        spl_curve_norm = spl_curves[:, j] - spl_ref
        ax.semilogx(freqs, spl_curve_norm, color=color, label=f"{float(angle):g}°", linewidth=1.5)

    spl_power = _sound_power_spl_equivalent_db(response, distance_m=float(distance_m), ref_pa=float(ref_pa))
    ax.semilogx(
        freqs,
        spl_power - spl_ref,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Sound power (eq)",
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"dB (normalized to {norm_angle_used_deg:.0f}°)")
    ax.set_title(f"Normalized SPL + Sound Power (ref {norm_angle_used_deg:.0f}°)")
    _setup_freq_axis(ax)

    ax.set_ylim([-24, 3])
    ax.set_yticks([3, 0, -3, -6, -9, -12, -15, -18, -21, -24])
    ax.set_yticklabels(["3", "0", "-3", "-6", "-9", "-12", "-15", "-18", "-21", "-24"])

    ax.legend(loc="lower left", fontsize=DEFAULT_PLOT_CONFIG.font_size_tick, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    return ax
