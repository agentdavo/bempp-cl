from __future__ import annotations

import numpy as np

try:
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

from bempp_audio.viz.style import DEFAULT_PLOT_CONFIG, setup_freq_axis as _setup_freq_axis


def render_driver_impedance_panel(ax, response: "FrequencyResponse") -> object:
    """Render driver electrical impedance (magnitude + phase)."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    freqs = np.asarray(response.frequencies, dtype=float)

    z_data = response.driver_impedance
    if z_data is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Driver metrics not available", ha="center", va="center")
        return ax

    z_re, z_im = z_data
    z_re = np.asarray(z_re, dtype=float)
    z_im = np.asarray(z_im, dtype=float)
    z_mag = np.sqrt(z_re**2 + z_im**2)
    z_phase = np.degrees(np.arctan2(z_im, z_re))

    ax.semilogx(freqs, z_mag, "k-", linewidth=2, label="|Z|")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Impedance (Ω)")
    ax.set_title("Driver Electrical Impedance")
    _setup_freq_axis(ax)
    ax.set_ylim([0, 40])
    ax.set_yticks([0, 8, 16, 24, 32, 40])
    ax.grid(True, which="both", alpha=0.3)

    ax_phase = ax.twinx()
    ax_phase.semilogx(freqs, z_phase, "k--", linewidth=1.5, label="Phase")
    ax_phase.set_ylabel("Phase (°)")
    ax_phase.set_ylim([-90, 90])
    ax_phase.set_yticks([-90, -45, 0, 45, 90])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_phase.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        fontsize=DEFAULT_PLOT_CONFIG.font_size_tick,
    )
    return ax


def render_driver_excursion_panel(
    ax,
    response: "FrequencyResponse",
    *,
    distance_m: float,
    spl_on_axis_db: np.ndarray,
) -> object:
    """
    Render driver peak excursion.

    Notes
    -----
    - `spl_on_axis_db` should be the absolute SPL at 0° for the same `distance_m`
      (used for the optional "excursion needed for 110 dB" overlay).
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    freqs = np.asarray(response.frequencies, dtype=float)

    x_mm = response.driver_excursion_mm
    if x_mm is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Driver metrics not available", ha="center", va="center")
        return ax

    x_mm = np.asarray(x_mm, dtype=float)
    ax.semilogx(freqs, x_mm, "k-", linewidth=2, label="2.83V RMS")

    # Calculate excursion required for 110 dB at `distance_m` (best-effort)
    try:
        spl_on_axis = np.asarray(spl_on_axis_db, dtype=float)
        target_spl = 110.0
        scale_factor = 10 ** ((target_spl - spl_on_axis) / 20)
        x_for_target = x_mm * scale_factor
        label = f"{target_spl:.0f} dB @ {float(distance_m):g} m"
        ax.semilogx(freqs, x_for_target, "k--", linewidth=1.5, label=label)
    except Exception:
        pass

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Peak Excursion (mm)")
    ax.set_title("Driver Excursion")
    _setup_freq_axis(ax)
    ax.legend(loc="upper right", fontsize=DEFAULT_PLOT_CONFIG.font_size_tick)

    x_max = float(np.max(x_mm)) * 1.2 if np.size(x_mm) and float(np.max(x_mm)) > 0 else 1.0
    ax.set_ylim([0, x_max])
    ax.grid(True, which="both", alpha=0.3)
    return ax


def render_driver_panel(
    ax_impedance,
    ax_excursion,
    response: "FrequencyResponse",
    *,
    distance_m: float,
    spl_on_axis_db: np.ndarray,
) -> tuple[object, object]:
    """Backward-compatible two-axis driver panel used by `summary_overview()`."""
    render_driver_impedance_panel(ax_impedance, response)
    render_driver_excursion_panel(
        ax_excursion,
        response,
        distance_m=float(distance_m),
        spl_on_axis_db=np.asarray(spl_on_axis_db, dtype=float),
    )
    return ax_impedance, ax_excursion
