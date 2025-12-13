"""
Directivity-focused matplotlib plots.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mcolors = None

from bempp_audio.viz.style import setup_style as _setup_style
from bempp_audio.viz.data.metrics import beamwidth_vs_frequency, directivity_index_vs_frequency

if TYPE_CHECKING:
    from bempp_audio.results import FrequencyResponse, RadiationResult, DirectivityBalloon


def polar_directivity(
    result: "RadiationResult",
    ax=None,
    plane: str = "xz",
    normalize: bool = True,
    **kwargs,
):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        _setup_style()
        _, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    dp = result.directivity()
    theta, pattern = dp.polar_2d(plane=plane)

    mag = np.maximum(np.abs(pattern), 1e-20)
    mag_db = 20 * np.log10(mag)
    if normalize:
        mag_db -= mag_db.max()

    mag_db = np.maximum(mag_db, -40)
    ax.plot(theta, mag_db + 40, **kwargs)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([0, 40])
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_yticklabels(["-40", "-30", "-20", "-10", "0"])
    ax.set_title(f"Directivity at {result.frequency:.0f} Hz")

    return ax


def polar_directivity_multi(
    response: "FrequencyResponse",
    frequencies: List[float],
    ax=None,
    plane: str = "xz",
    **kwargs,
):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        _setup_style()
        _, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))

    for freq, color in zip(frequencies, colors):
        idx = int(np.argmin(np.abs(response.frequencies - float(freq))))
        result = response.results[idx]

        dp = result.directivity()
        theta, pattern = dp.polar_2d(plane=plane)

        mag = np.maximum(np.abs(pattern), 1e-20)
        mag_db = 20 * np.log10(mag)
        mag_db -= mag_db.max()
        mag_db = np.maximum(mag_db, -40)

        ax.plot(theta, mag_db + 40, color=color, label=f"{freq:.0f} Hz", **kwargs)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([0, 40])
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_yticklabels(["-40", "-30", "-20", "-10", "0"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title("Directivity")

    return ax


def directivity_balloon_3d(
    balloon: "DirectivityBalloon",
    backend: str = "plotly",
    **kwargs,
):
    X, Y, Z = balloon.to_cartesian()
    mag_db = balloon.magnitude_db(normalize=True)

    if backend == "plotly":
        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError("plotly required for 3D balloon") from e

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    surfacecolor=mag_db,
                    colorscale="RdBu_r",
                    cmin=-30,
                    cmax=0,
                    colorbar=dict(title="dB"),
                )
            ]
        )

        fig.update_layout(
            title=f"Directivity Balloon at {balloon.frequency:.0f} Hz",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
            ),
        )
        return fig

    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    _setup_style()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    norm = mcolors.Normalize(vmin=-30, vmax=0)
    colors = plt.cm.RdBu_r(norm(mag_db))

    ax.plot_surface(X, Y, Z, facecolors=colors, shade=False)
    ax.set_title(f"Directivity Balloon at {balloon.frequency:.0f} Hz")

    return fig


def directivity_metrics(
    response: "FrequencyResponse",
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

    freqs, di = directivity_index_vs_frequency(response, angle_deg=0.0)
    ax1.semilogx(freqs, di, **kwargs)
    ax1.set_ylabel("DI (dB)")
    ax1.set_title("Directivity Metrics")
    ax1.grid(True, which="both", alpha=0.3)

    freqs, bw = beamwidth_vs_frequency(response, level_db=-6.0, plane="xz")
    ax2.semilogx(freqs, bw, **kwargs)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Beamwidth (degrees)")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    return fig

