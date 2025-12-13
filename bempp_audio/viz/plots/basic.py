"""
Basic matplotlib plots for acoustic results.
"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    plt = None

from bempp_audio.viz.style import (
    DEFAULT_PLOT_CONFIG,
    setup_style as _setup_style,
)

if TYPE_CHECKING:
    from bempp_audio.results import FrequencyResponse, RadiationResult


def spl_response(
    response: "FrequencyResponse",
    point: np.ndarray,
    ax=None,
    label: str = "BEM",
    measured: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    **kwargs,
):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        _setup_style()
        _, ax = plt.subplots(figsize=(10, 6))

    freqs, spl = response.spl_at_point(point)
    ax.semilogx(freqs, spl, label=label, **kwargs)

    if measured is not None:
        mfreqs, mspl = measured
        ax.semilogx(mfreqs, mspl, "--", label="Measured", alpha=0.7)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPL (dB)")
    ax.set_title("Frequency Response")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    ax.set_xlim([DEFAULT_PLOT_CONFIG.freq_min_hz, DEFAULT_PLOT_CONFIG.freq_max_hz])

    return ax


def phase_response(
    response: "FrequencyResponse",
    point: np.ndarray,
    ax=None,
    label: str = "BEM",
    unwrap: bool = True,
    **kwargs,
):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        _setup_style()
        _, ax = plt.subplots(figsize=(10, 6))

    freqs, phase = response.phase_at_point(point, unwrap=unwrap)
    ax.semilogx(freqs, np.degrees(phase), label=label, **kwargs)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (degrees)")
    ax.set_title("Phase Response")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()

    return ax


def bode_plot(
    response: "FrequencyResponse",
    point: np.ndarray,
    fig=None,
    **kwargs,
):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    if fig is None:
        _setup_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    else:
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

    spl_response(response, point, ax=ax1, **kwargs)
    ax1.set_xlabel("")

    phase_response(response, point, ax=ax2, **kwargs)

    fig.tight_layout()
    return fig


def group_delay(
    response: "FrequencyResponse",
    point: np.ndarray,
    ax=None,
    **kwargs,
):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        _setup_style()
        _, ax = plt.subplots(figsize=(10, 6))

    freqs, gd = response.group_delay(point)
    ax.semilogx(freqs, gd * 1000, **kwargs)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Group Delay (ms)")
    ax.set_title("Group Delay")
    ax.grid(True, which="both", ls="-", alpha=0.3)

    return ax


def pressure_field_2d(
    result: "RadiationResult",
    plane: str = "xz",
    extent: float = 1.0,
    n_points: int = 100,
    ax=None,
    quantity: str = "spl",
    **kwargs,
):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    from bempp_audio.results.pressure_field import PressureField

    pf = PressureField(result)
    C1, C2, pressure = pf.on_plane(plane, extent, n_points)

    if quantity == "spl":
        data = 20 * np.log10(np.abs(pressure) / (np.sqrt(2) * 20e-6))
        label = "SPL (dB)"
        cmap = "viridis"
    elif quantity == "magnitude":
        data = np.abs(pressure)
        label = "Pressure (Pa)"
        cmap = "viridis"
    elif quantity == "phase":
        data = np.degrees(np.angle(pressure))
        label = "Phase (deg)"
        cmap = "hsv"
    else:
        raise ValueError(f"Unknown quantity: {quantity}")

    if ax is None:
        _setup_style()
        _, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(
        data.T,
        origin="lower",
        extent=[-extent, extent, -extent, extent],
        cmap=cmap,
        **kwargs,
    )
    plt.colorbar(im, ax=ax, label=label)

    if plane == "xz":
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
    elif plane == "xy":
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    else:
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("Z (m)")

    ax.set_title(f"Pressure Field at {result.frequency:.0f} Hz")
    return ax
