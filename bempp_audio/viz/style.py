"""
Shared plot styling for bempp_audio visualizations.

This module centralizes matplotlib defaults and common axis formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Mapping, Sequence
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    ticker = None


@dataclass(frozen=True)
class CEAStyle:
    """ANSI/CEA-2034A plotting constants."""

    width_in: float = 12.0
    height_in: float = 14.0
    dpi: int = 200

    freq_min_hz: float = 200.0
    freq_max_hz: float = 20000.0

    spl_min_db: float = -45.0
    spl_max_db: float = 5.0
    spl_step_db: float = 5.0

    di_min_db: float = 0.0
    di_max_db: float = 15.0
    di_step_db: float = 5.0

    freq_ticks_hz: Sequence[float] = (200, 500, 1000, 2000, 5000, 10000, 20000)
    freq_tick_labels: Sequence[str] = ("200", "500", "1k", "2k", "5k", "10k", "20k")

    colors: Mapping[str, str] = field(
        default_factory=lambda: {
            "on_axis": "#000000",
            "listening_window": "#0066CC",
            "early_reflections": "#00CC66",
            "sound_power": "#CC0000",
            "directivity": "#6600CC",
            "early_di": "#FF6600",
            "grid": "#CCCCCC",
        }
    )

    linewidth: float = 1.5
    linewidth_thin: float = 1.0


@dataclass(frozen=True)
class PlotConfig:
    """
    Global plotting configuration for bempp_audio visualizations.

    Most modules should treat this as the single source of truth for plot styling.
    """

    figsize: tuple[float, float] = (19.2, 10.8)  # 1920x1080 at 100 DPI
    dpi: int = 100

    font_family: str = "Courier New"
    font_size_tick: int = 10
    font_size_label: int = 11
    font_size_title: int = 12

    freq_ticks_hz: Sequence[float] = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
    freq_tick_labels: Sequence[str] = ("20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k")

    freq_min_hz: float = 200.0
    freq_max_hz: float = 20000.0

    CEA: ClassVar[CEAStyle] = CEAStyle()


DEFAULT_PLOT_CONFIG = PlotConfig()

# Backwards-compatible constants (prefer `DEFAULT_PLOT_CONFIG`).
DEFAULT_FIGSIZE = DEFAULT_PLOT_CONFIG.figsize
DEFAULT_DPI = DEFAULT_PLOT_CONFIG.dpi
FONT_FAMILY = DEFAULT_PLOT_CONFIG.font_family
FONT_SIZE_TICK = DEFAULT_PLOT_CONFIG.font_size_tick
FONT_SIZE_LABEL = DEFAULT_PLOT_CONFIG.font_size_label
FONT_SIZE_TITLE = DEFAULT_PLOT_CONFIG.font_size_title
FREQ_TICKS = list(DEFAULT_PLOT_CONFIG.freq_ticks_hz)
FREQ_LABELS = list(DEFAULT_PLOT_CONFIG.freq_tick_labels)
DEFAULT_FREQ_MIN = DEFAULT_PLOT_CONFIG.freq_min_hz
DEFAULT_FREQ_MAX = DEFAULT_PLOT_CONFIG.freq_max_hz


def setup_style(config: PlotConfig = DEFAULT_PLOT_CONFIG) -> None:
    """Configure matplotlib style for acoustic plots."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.rcParams.update(
        {
            "font.family": "monospace",
            "font.monospace": [config.font_family, "DejaVu Sans Mono", "Courier"],
            "font.size": config.font_size_tick,
            "axes.labelsize": config.font_size_label,
            "axes.titlesize": config.font_size_title,
            "xtick.labelsize": config.font_size_tick,
            "ytick.labelsize": config.font_size_tick,
            "legend.fontsize": config.font_size_tick,
            "figure.titlesize": config.font_size_title + 2,
        }
    )


def setup_freq_axis(
    ax,
    f_min: float = None,
    f_max: float = None,
    *,
    config: PlotConfig = DEFAULT_PLOT_CONFIG,
) -> None:
    """
    Configure frequency axis with standard audio ticks.

    Uses DEFAULT_FREQ_MIN (200 Hz) and DEFAULT_FREQ_MAX (20 kHz) when
    f_min/f_max are not specified.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    # Use defaults if not specified
    if f_min is None:
        f_min = config.freq_min_hz
    if f_max is None:
        f_max = config.freq_max_hz

    visible_ticks = []
    visible_labels = []
    for t, l in zip(config.freq_ticks_hz, config.freq_tick_labels):
        if f_min * 0.9 <= t <= f_max * 1.1:
            visible_ticks.append(t)
            visible_labels.append(l)

    if visible_ticks:
        ax.set_xticks(visible_ticks)
        ax.set_xticklabels(visible_labels)

    # Handle single-frequency case: expand limits slightly
    if f_min == f_max:
        ax.set_xlim([f_min * 0.5, f_max * 2.0])
    else:
        ax.set_xlim([f_min, f_max])

    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
