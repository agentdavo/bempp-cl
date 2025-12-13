from __future__ import annotations

import numpy as np

try:
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

from bempp_audio.viz.style import DEFAULT_PLOT_CONFIG, setup_freq_axis as _setup_freq_axis


def render_impedance_panel(
    ax,
    response: "FrequencyResponse",
    *,
    show_progress: bool = True,
    normalize_to_rhoc: bool = True,
    y_min: float = 0.0,
    y_max: float = 2.0,
) -> object:
    """
    Render radiation impedance (specific) vs frequency.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    from bempp_audio.progress import ProgressTracker

    freqs = np.asarray(response.frequencies, dtype=float)
    z_re: list[float] = []
    z_im: list[float] = []
    z0_spec = np.nan
    if bool(normalize_to_rhoc):
        try:
            if response.results:
                rho = float(getattr(response.results[0], "rho", np.nan))
                c = float(getattr(response.results[0], "c", np.nan))
                if np.isfinite(rho) and np.isfinite(c) and rho > 0 and c > 0:
                    z0_spec = float(rho * c)  # Pa·s/m
        except Exception:
            z0_spec = np.nan

    with ProgressTracker(
        total=len(freqs),
        desc="Computing impedance",
        unit="freq",
        disable=not bool(show_progress),
    ) as pbar:
        for result in response.results:
            z = result.radiation_impedance()
            z_spec = z.specific()  # Pa·s/m
            zr = float(np.real(z_spec))
            zi = float(np.imag(z_spec))
            if np.isfinite(z0_spec) and float(z0_spec) > 0:
                zr /= float(z0_spec)
                zi /= float(z0_spec)
            z_re.append(zr)
            z_im.append(zi)
            pbar.update(item=f"{result.frequency:.0f} Hz")

    ax.semilogx(freqs, z_re, "k-", linewidth=2, label="Re(Z)")
    ax.semilogx(freqs, z_im, "k--", linewidth=1.5, label="Im(Z)")

    ax.set_xlabel("Frequency (Hz)")
    if np.isfinite(z0_spec) and float(z0_spec) > 0:
        ax.set_ylabel("Specific Impedance (Z / ρc)")
        ax.set_ylim(float(y_min), float(y_max))
        ax.axhline(1.0, color="#AA0000", linestyle="--", linewidth=1.0, label="ρc reference")
    else:
        ax.set_ylabel("Specific Impedance (Pa·s/m)")
    ax.set_title("Radiation Impedance")
    _setup_freq_axis(ax)
    ax.legend(loc="best", fontsize=DEFAULT_PLOT_CONFIG.font_size_tick)
    ax.grid(True, which="both", alpha=0.3)
    return ax
