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


def render_polar_panel(
    ax,
    *,
    frequencies_hz: np.ndarray,
    angles_symmetric_deg: np.ndarray,
    spl_symmetric_db: np.ndarray,
    max_angle_deg: float,
    norm_angle_used_deg: float,
    cmap: str = "jet",
) -> object:
    """
    Render a normalized polar contour panel.

    Parameters
    ----------
    ax:
        Matplotlib axes.
    frequencies_hz:
        Frequencies (n_f,).
    angles_symmetric_deg:
        Symmetric angle grid (-max..max), (n_a_sym,).
    spl_symmetric_db:
        Normalized SPL map (n_a_sym, n_f).
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    freqs = np.asarray(frequencies_hz, dtype=float)
    theta_sym = np.asarray(angles_symmetric_deg, dtype=float)
    spl_sym = np.asarray(spl_symmetric_db, dtype=float)

    if freqs.size >= 2:
        F, T = np.meshgrid(freqs, theta_sym)
        polar_levels = np.asarray(np.arange(-28, 8, 2), dtype=float)
        polar_levels = np.unique(polar_levels)
        polar_levels = polar_levels[np.isfinite(polar_levels)]
        polar_levels.sort()

        cs = ax.contourf(
            F,
            T,
            spl_sym,
            levels=polar_levels,
            cmap=cmap,
            vmin=-28,
            vmax=6,
            extend="both",
        )

        # Only draw contour lines if at least two requested levels fall inside the data range.
        with np.errstate(all="ignore"):
            zmin = float(np.nanmin(spl_sym))
            zmax = float(np.nanmax(spl_sym))
        contour_levels = np.asarray([-24, -18, -12, -6, 0, 6], dtype=float)
        contour_levels = np.unique(contour_levels)
        contour_levels = contour_levels[np.isfinite(contour_levels)]
        contour_levels.sort()
        if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
            in_range = contour_levels[(contour_levels >= zmin) & (contour_levels <= zmax)]
            if in_range.size >= 2:
                ax.contour(F, T, spl_sym, levels=in_range, colors="k", linewidths=0.5)
        cbar = plt.colorbar(cs, ax=ax, label="dB (normalized)", ticks=np.arange(-28, 8, 4))
        cbar.ax.tick_params(labelsize=DEFAULT_PLOT_CONFIG.font_size_tick)

        ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title(f"Polar Map (normalized to {norm_angle_used_deg:.0f}°)")
        _setup_freq_axis(ax)
        ax.set_ylim([-float(max_angle_deg), float(max_angle_deg)])
        return ax

    # Single frequency: show polar response vs angle
    spl_single = spl_sym[:, 0] if spl_sym.ndim == 2 and spl_sym.shape[1] else spl_sym
    ax.plot(theta_sym, spl_single, "b-", linewidth=2)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    ax.axhline(y=-6, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel(f"dB (normalized to {norm_angle_used_deg:.0f}°)")
    ax.set_title(f"Polar Response @ {freqs[0]:.0f} Hz" if freqs.size else "Polar Response")
    ax.set_xlim([-float(max_angle_deg), float(max_angle_deg)])
    ax.set_ylim([-30, 6])
    ax.grid(True, alpha=0.3)
    return ax
