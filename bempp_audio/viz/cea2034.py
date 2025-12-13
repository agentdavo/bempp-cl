"""
ANSI/CEA-2034A Standard Loudspeaker Measurement Chart Generator.

This module generates publication-quality charts following the ANSI/CEA-2034A
standard for loudspeaker measurements. The standard defines a comprehensive
set of acoustic measurements including:

- On-axis frequency response
- Listening window (±10° horizontal and vertical average)
- Early reflections (floor, ceiling, side walls, front and rear walls)
- Sound power (total hemispherical radiated power)
- Sound Power Directivity Index (SPDI)
- Early Reflections Directivity Index (ERDI)

The chart format follows the Spinorama convention:
- X-axis: 200 Hz to 20 kHz (logarithmic)
- Y-axis (SPL): +5 dB to -45 dB (5 dB increments)
- Y-axis (DI): 0 dB to 15 dB (5 dB increments)
- Resolution: 200 DPI for publication quality
- Normalized to on-axis response at a reference frequency

Usage Example
-------------
>>> from bempp_audio import Loudspeaker
>>> from bempp_audio.viz import CEA2034Chart, plot_cea2034
>>>
>>> # Solve for loudspeaker response
>>> response = (Loudspeaker()
...     .circular_piston(radius=0.05)
...     .infinite_baffle()
...     .frequency_range(200, 20000, num=100)
...     .polar_angles(0, 90, num=19)
...     .solve())
>>>
>>> # Generate standard Spinorama chart
>>> plot_cea2034(response, chart_type='spinorama', save='spinorama.png')
>>>
>>> # Generate full CEA-2034A report
>>> chart = CEA2034Chart(response, measurement_distance=1.0)
>>> chart.plot_full_report(save='cea2034_report.png')

References
----------
ANSI/CEA-2034-A (2015): Standard Method of Measurement for In-Home
Loudspeakers

Implementation Notes
--------------------
This implementation currently uses a simplified 2D polar sampling model:
- Polar responses are sampled in a single plane derived from `mesh.axis`
  (horizontal plane approximation).
- Full multi-plane CEA-2034A acquisition (horizontal + vertical + listening
  window definitions, etc.) is not implemented yet; outputs are approximations.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    gridspec = None
    MultipleLocator = None
    FuncFormatter = None

if TYPE_CHECKING:
    from bempp_audio.results import FrequencyResponse

from bempp_audio.viz.style import PlotConfig

# =============================================================================
# CEA-2034A Chart Configuration
# =============================================================================

_CEA = PlotConfig.CEA

CEA_WIDTH = _CEA.width_in
CEA_HEIGHT = _CEA.height_in
CEA_DPI = _CEA.dpi

CEA_FREQ_MIN = _CEA.freq_min_hz
CEA_FREQ_MAX = _CEA.freq_max_hz

CEA_SPL_MIN = _CEA.spl_min_db
CEA_SPL_MAX = _CEA.spl_max_db
CEA_SPL_STEP = _CEA.spl_step_db

CEA_DI_MIN = _CEA.di_min_db
CEA_DI_MAX = _CEA.di_max_db
CEA_DI_STEP = _CEA.di_step_db

CEA_FREQ_TICKS = list(_CEA.freq_ticks_hz)
CEA_FREQ_LABELS = list(_CEA.freq_tick_labels)

# Standard measurement angles (degrees)
# Horizontal angles for directivity
HORIZONTAL_ANGLES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
VERTICAL_ANGLES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# Common spinorama-style angle sets (degrees from on-axis).
# These defaults aim to be useful and "multi-plane" (horizontal + vertical),
# but they are not a claim of strict ANSI/CEA-2034A compliance.
LISTENING_WINDOW_H = [-30, -20, -10, 0, 10, 20, 30]
LISTENING_WINDOW_V = [-10, 10]

CEA_COLORS = dict(_CEA.colors)
CEA_LINEWIDTH = _CEA.linewidth
CEA_LINEWIDTH_THIN = _CEA.linewidth_thin


@dataclass
class CEA2034Data:
    """
    Container for CEA-2034A measurement data.
    
    All SPL values are in dB, normalized to the reference.
    Frequencies are in Hz.
    """
    frequencies: np.ndarray
    
    # Main curves
    on_axis: np.ndarray
    listening_window: np.ndarray
    early_reflections: np.ndarray
    sound_power: np.ndarray
    
    # Directivity indices
    directivity_index: np.ndarray
    early_reflections_di: np.ndarray
    
    # Individual reflection components (optional, for detailed plots)
    floor_bounce: Optional[np.ndarray] = None
    ceiling_bounce: Optional[np.ndarray] = None
    front_wall_bounce: Optional[np.ndarray] = None
    side_wall_bounce: Optional[np.ndarray] = None
    rear_wall_bounce: Optional[np.ndarray] = None
    
    # Normalization info
    reference_frequency: float = 1000.0
    reference_level: float = 0.0


class CEA2034Calculator:
    """
    Calculate CEA-2034-style (spinorama) measurements from radiation results.
    
    This class processes directivity data to compute the standard
    curves popularized by ANSI/CEA-2034A / "spinorama" workflows.

    Notes
    -----
    This implementation is designed to be practical and multi-plane
    (horizontal + vertical), but it does not claim strict ANSI/CEA-2034A
    compliance. If you need compliance, review/adjust the angle sets and
    early-reflections weights to match your standard/measurement process.
    
    Parameters
    ----------
    response : FrequencyResponse
        Frequency response with polar directivity data.
    measurement_distance : float
        Distance in meters for SPL calculation (default 1.0m).
    reference_frequency : float
        Frequency for normalization (default 1000 Hz).
    normalize_to_angle : float
        Angle for normalization reference (default 0° = on-axis).
    """
    
    def __init__(
        self,
        response: "FrequencyResponse",
        measurement_distance: float = 1.0,
        reference_frequency: float = 1000.0,
        normalize_to_angle: float = 0.0,
    ):
        self.response = response
        self.distance = measurement_distance
        self.ref_freq = reference_frequency
        self.norm_angle = normalize_to_angle
    
    def compute(self) -> CEA2034Data:
        """
        Compute all CEA-2034A standard measurements.
        
        Returns
        -------
        CEA2034Data
            Complete set of CEA-2034A curves.
        """
        # Extract frequencies
        freqs = np.array([r.frequency for r in self.response.results])
        
        # Compute multi-plane polar sweeps at required angles.
        sweeps = self._compute_polar_sweeps()
        
        # Compute standard curves
        horizontal = sweeps["horizontal"]
        vertical = sweeps["vertical"]

        on_axis = horizontal.by_angle()[0.0]  # 0° horizontal
        listening_window = self._compute_listening_window(horizontal, vertical)
        early_reflections = self._compute_early_reflections(horizontal, vertical)
        sound_power = self._compute_sound_power(freqs)
        
        # Compute directivity indices
        # Directivity Index (DI) = On-axis SPL - Sound Power SPL
        # Positive DI means speaker is more directional (on-axis louder than average)
        # DI = 0 dB: omnidirectional (on-axis = sound power)
        # DI = 6 dB: typical for many speakers at high frequencies
        # DI = 10-15 dB: highly directional (horns, waveguides)
        directivity_index = on_axis - sound_power  # SPDI (Sound Power DI)
        erdi = on_axis - early_reflections         # ERDI (Early Reflections DI)
        
        # Normalize all curves
        ref_idx = np.argmin(np.abs(freqs - self.ref_freq))
        ref_level = on_axis[ref_idx]
        
        on_axis -= ref_level
        listening_window -= ref_level
        early_reflections -= ref_level
        sound_power -= ref_level
        # DI curves are already differential, no normalization needed
        
        return CEA2034Data(
            frequencies=freqs,
            on_axis=on_axis,
            listening_window=listening_window,
            early_reflections=early_reflections,
            sound_power=sound_power,
            directivity_index=directivity_index,
            early_reflections_di=erdi,
            reference_frequency=self.ref_freq,
            reference_level=ref_level,
        )
    
    def _compute_polar_sweeps(self) -> Dict[str, "PolarSweep"]:
        """
        Compute horizontal + vertical polar sweeps at the angles needed by the calculator.

        Returns
        -------
        dict
            `{"horizontal": PolarSweep, "vertical": PolarSweep}`
        """
        from bempp_audio.results.polar import PolarSweep

        # Angles needed for the standard curves, plus enough coverage for the
        # contour plot and typical "spinorama" use.
        angles_h = np.arange(-180.0, 181.0, 10.0)
        angles_v = np.arange(-90.0, 91.0, 10.0)

        horizontal = self.response.polar_sweep(
            angles_deg=angles_h.tolist(),
            distance_m=float(self.distance),
            plane="horizontal",
        )
        vertical = self.response.polar_sweep(
            angles_deg=angles_v.tolist(),
            distance_m=float(self.distance),
            plane="vertical",
        )

        return {"horizontal": horizontal, "vertical": vertical}

    def _compute_listening_window(self, horizontal: "PolarSweep", vertical: "PolarSweep") -> np.ndarray:
        """
        Listening Window: Average of ±10° horizontal and vertical.
        
        CEA-2034A defines this as the average of:
        - On-axis (0°)
        - ±10° horizontal
        - ±10° vertical
        """
        by_h = horizontal.by_angle()
        by_v = vertical.by_angle()

        curves = []
        for a in LISTENING_WINDOW_H:
            if float(a) in by_h:
                curves.append(by_h[float(a)])
        for a in LISTENING_WINDOW_V:
            if float(a) in by_v:
                curves.append(by_v[float(a)])

        if not curves:
            return by_h.get(0.0, np.zeros_like(horizontal.frequencies_hz, dtype=float))

        # Average in linear mean-square pressure (energy), convert back to dB SPL.
        # SPL(dB) = 10*log10(p_rms^2 / p_ref^2), so average uses 10**(SPL/10).
        p2_rel = np.mean([10 ** (spl / 10) for spl in curves], axis=0)
        return 10 * np.log10(np.maximum(p2_rel, 1e-20))

    def _compute_early_reflections(self, horizontal: "PolarSweep", vertical: "PolarSweep") -> np.ndarray:
        """
        Early Reflections: Weighted average of first reflections.
        
        This implementation uses both horizontal and vertical planes and applies
        weights to approximate the energy arriving from first reflections.

        Note: The exact angle sets/weights differ between published conventions.
        Treat these defaults as a useful starting point, not a standards claim.
        """
        ref_pa = 20e-6

        by_h = horizontal.by_angle()
        by_v = vertical.by_angle()

        groups = [
            ("floor", "vertical", [-40, -30, -20], 0.20),
            ("ceiling", "vertical", [40, 50, 60], 0.20),
            ("front", "horizontal", [-20, -10, 0, 10, 20], 0.25),
            ("side", "horizontal", [-60, -50, -40, 40, 50, 60], 0.25),
            ("rear", "horizontal", [140, 150, 160, 170, 180, -140, -150, -160, -170], 0.10),
        ]

        total_p2 = None
        total_w = 0.0

        for _, plane, angles, weight in groups:
            by = by_h if plane == "horizontal" else by_v
            curves = [by.get(float(a)) for a in angles if float(a) in by]
            if not curves:
                continue

            # Curves are in dB SPL; convert to linear p_rms and average energy.
            p_rms = np.stack([10 ** (spl / 20) for spl in curves], axis=0) * ref_pa
            p2 = np.mean(p_rms**2, axis=0)

            if total_p2 is None:
                total_p2 = weight * p2
            else:
                total_p2 = total_p2 + (weight * p2)
            total_w += weight

        if total_p2 is None or total_w <= 0:
            return by_h.get(0.0, np.zeros_like(horizontal.frequencies_hz, dtype=float))

        p_rms_eq = np.sqrt(total_p2 / total_w)
        return 20 * np.log10(np.maximum(p_rms_eq, 1e-20) / ref_pa)
    
    def _compute_sound_power(self, freqs: np.ndarray) -> np.ndarray:
        """
        Sound Power (pressure-equivalent): derive average p_rms from radiated power.

        Uses the far-field relation between total power and average intensity to
        compute an equivalent average RMS pressure at the configured distance.
        """
        from bempp_audio.baffles import InfiniteBaffle

        ref_pa = 20e-6
        distance = float(self.distance)
        spl = np.zeros_like(freqs, dtype=float)

        for i, result in enumerate(self.response.results):
            power_w = float(result.radiated_power())
            if power_w <= 0:
                spl[i] = -np.inf
                continue

            solid_angle = 2.0 * np.pi if isinstance(getattr(result, "baffle", None), InfiniteBaffle) else 4.0 * np.pi
            p_rms_avg = np.sqrt((power_w * float(result.rho) * float(result.c)) / (solid_angle * distance**2))
            spl[i] = 20 * np.log10(np.maximum(p_rms_avg, 1e-20) / ref_pa)

        return spl


class CEA2034Chart:
    """
    Generate CEA-2034-style charts (Spinorama).
    
    Creates publication-quality charts following the CEA-2034A format
    with proper frequency range (200 Hz - 20 kHz), SPL range (+5 to -45 dB),
    directivity index range (0 to 15 dB), and 200 DPI resolution.
    
    Chart Types Available
    ---------------------
    1. **Spinorama** (single panel):
       - On-axis response
       - Listening window
       - Early reflections
       - Sound power
       
    2. **Standard** (3-panel):
       - Panel 1: All frequency response curves
       - Panel 2: Directivity indices (SPDI, ERDI)
       - Panel 3: Early reflections detail
       
    3. **Full Report** (5-panel):
       - All standard panels plus sound power comparison and polar map
    
    Axis Specifications
    -------------------
    - X-axis: 200 Hz to 20 kHz (logarithmic scale)
    - Y-axis (SPL): +5 dB to -45 dB (5 dB increments)
    - Y-axis (DI): 0 dB to 15 dB (5 dB increments)
    - Resolution: 200 DPI
    
    Examples
    --------
    >>> from bempp_audio.viz import CEA2034Chart
    >>> chart = CEA2034Chart(response, measurement_distance=1.0)
    >>> 
    >>> # Generate Spinorama (most popular format)
    >>> chart.plot_spinorama(save="spinorama.png")
    >>> 
    >>> # Generate 3-panel standard chart
    >>> chart.plot_standard(save="cea2034_standard.png")
    >>> 
    >>> # Generate full 5-panel report
    >>> chart.plot_full_report(save="cea2034_report.png")
    """
    
    def __init__(
        self,
        response: "FrequencyResponse",
        measurement_distance: float = 1.0,
        reference_frequency: float = 1000.0,
        normalize_to_angle: float = 0.0,
    ):
        """
        Initialize CEA-2034A chart generator.
        
        Parameters
        ----------
        response : FrequencyResponse
            Acoustic response data.
        measurement_distance : float
            SPL measurement distance in meters.
        reference_frequency : float
            Normalization frequency in Hz.
        normalize_to_angle : float
            Normalization angle in degrees.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for CEA-2034A charts")
        
        self.response = response
        self.distance = measurement_distance
        self.ref_freq = reference_frequency
        self.norm_angle = normalize_to_angle
        
        # Compute CEA-2034A data
        calc = CEA2034Calculator(response, measurement_distance, reference_frequency, normalize_to_angle)
        self.data = calc.compute()
    
    def plot_standard(
        self,
        save: Optional[str] = None,
        title: str = "ANSI/CEA-2034A Loudspeaker Measurements",
        show: bool = False,
    ):
        """
        Generate the standard CEA-2034A 3-panel chart.
        
        Panel 1: Frequency Response Curves
        Panel 2: Directivity Index
        Panel 3: Early Reflections Curve
        
        Parameters
        ----------
        save : str, optional
            Filename to save chart (PNG format at 200 DPI).
        title : str
            Chart title.
        show : bool
            If True, display chart interactively.
        """
        if not MATPLOTLIB_AVAILABLE or plt is None or gridspec is None:
            raise ImportError("matplotlib required for plotting")
        
        fig = plt.figure(figsize=(CEA_WIDTH, CEA_HEIGHT), dpi=CEA_DPI)
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
        
        # Panel 1: Frequency Response
        ax1 = fig.add_subplot(gs[0])
        self._plot_frequency_response(ax1)
        
        # Panel 2: Directivity Index
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        self._plot_directivity_index(ax2)
        
        # Panel 3: Early Reflections
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        self._plot_early_reflections(ax3)
        
        # Overall title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        if save:
            plt.savefig(save, dpi=CEA_DPI, bbox_inches='tight', facecolor='white')
            from bempp_audio.progress import get_logger

            get_logger().info(f"CEA-2034A chart saved: {save}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_spinorama(
        self,
        save: Optional[str] = None,
        title: str = "Spinorama - CEA-2034A Measurements",
        show: bool = False,
    ):
        """
        Generate the popular "Spinorama" single-panel chart.
        
        Shows all main curves on one plot for quick assessment.
        
        Parameters
        ----------
        save : str, optional
            Filename to save chart.
        title : str
            Chart title.
        show : bool
            If True, display chart.
        """
        if not MATPLOTLIB_AVAILABLE or plt is None:
            raise ImportError("matplotlib required for plotting")
        
        fig, ax = plt.subplots(figsize=(CEA_WIDTH, 8), dpi=CEA_DPI)
        
        # Plot all main curves
        freqs = self.data.frequencies
        
        ax.plot(freqs, self.data.on_axis, 
                color=CEA_COLORS['on_axis'], linewidth=CEA_LINEWIDTH,
                label='On-Axis', zorder=5)
        
        ax.plot(freqs, self.data.listening_window,
                color=CEA_COLORS['listening_window'], linewidth=CEA_LINEWIDTH,
                label='Listening Window', linestyle='--', zorder=4)
        
        ax.plot(freqs, self.data.early_reflections,
                color=CEA_COLORS['early_reflections'], linewidth=CEA_LINEWIDTH,
                label='Early Reflections', linestyle='--', zorder=3)
        
        ax.plot(freqs, self.data.sound_power,
                color=CEA_COLORS['sound_power'], linewidth=CEA_LINEWIDTH,
                label='Sound Power', linestyle=':', zorder=2)
        
        # Configure axes
        self._setup_freq_axis(ax)
        self._setup_spl_axis(ax)
        
        ax.set_ylabel('Relative SPL (dB)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
        ax.grid(True, which='both', color=CEA_COLORS['grid'], 
                linewidth=0.5, alpha=0.7)
        
        if save:
            plt.savefig(save, dpi=CEA_DPI, bbox_inches='tight', facecolor='white')
            from bempp_audio.progress import get_logger

            get_logger().info(f"Spinorama chart saved: {save}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_full_report(
        self,
        save: Optional[str] = None,
        title: str = "Complete CEA-2034A Loudspeaker Report",
        show: bool = False,
    ):
        """
        Generate a comprehensive 5-panel report.
        
        Includes frequency response, directivity, reflections,
        sound power comparison, and horizontal polar map.
        
        Parameters
        ----------
        save : str, optional
            Filename to save report.
        title : str
            Report title.
        show : bool
            If True, display report.
        """
        if not MATPLOTLIB_AVAILABLE or plt is None or gridspec is None:
            raise ImportError("matplotlib required for plotting")
        
        fig = plt.figure(figsize=(CEA_WIDTH, 16), dpi=CEA_DPI)
        gs = gridspec.GridSpec(5, 1, height_ratios=[2, 1, 1, 1, 1.5], hspace=0.35)
        
        # Panel 1: Frequency Response
        ax1 = fig.add_subplot(gs[0])
        self._plot_frequency_response(ax1)
        ax1.set_title('Frequency Response Curves', fontsize=12, fontweight='bold', pad=10)
        
        # Panel 2: Directivity Index
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        self._plot_directivity_index(ax2)
        ax2.set_title('Directivity Index', fontsize=12, fontweight='bold', pad=10)
        
        # Panel 3: Sound Power Comparison
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        self._plot_sound_power_comparison(ax3)
        ax3.set_title('Sound Power vs On-Axis', fontsize=12, fontweight='bold', pad=10)
        
        # Panel 4: Early Reflections Detail
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        self._plot_early_reflections(ax4)
        ax4.set_title('Early Reflections', fontsize=12, fontweight='bold', pad=10)
        
        # Panel 5: Horizontal Polar Map (placeholder)
        ax5 = fig.add_subplot(gs[4], sharex=ax1)
        self._plot_horizontal_contour(ax5)
        ax5.set_title('Horizontal Directivity (normalized)', fontsize=12, fontweight='bold', pad=10)
        
        # Overall title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.997)
        
        if save:
            plt.savefig(save, dpi=CEA_DPI, bbox_inches='tight', facecolor='white')
            from bempp_audio.progress import get_logger

            get_logger().info(f"Full CEA-2034A report saved: {save}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    # -------------------------------------------------------------------------
    # Internal plotting methods
    # -------------------------------------------------------------------------
    
    def _plot_frequency_response(self, ax):
        """Plot main frequency response curves."""
        freqs = self.data.frequencies
        
        ax.plot(freqs, self.data.on_axis,
                color=CEA_COLORS['on_axis'], linewidth=CEA_LINEWIDTH,
                label='On-Axis', zorder=5)
        
        ax.plot(freqs, self.data.listening_window,
                color=CEA_COLORS['listening_window'], linewidth=CEA_LINEWIDTH,
                label='Listening Window', linestyle='--', zorder=4)
        
        ax.plot(freqs, self.data.early_reflections,
                color=CEA_COLORS['early_reflections'], linewidth=CEA_LINEWIDTH,
                label='Early Reflections', linestyle='-.', zorder=3)
        
        ax.plot(freqs, self.data.sound_power,
                color=CEA_COLORS['sound_power'], linewidth=CEA_LINEWIDTH,
                label='Sound Power', linestyle=':', zorder=2)
        
        self._setup_freq_axis(ax)
        self._setup_spl_axis(ax)
        ax.set_ylabel('Relative SPL (dB)', fontsize=11, fontweight='bold')
        ax.legend(loc='lower left', framealpha=0.95, fontsize=9)
        ax.grid(True, which='both', color=CEA_COLORS['grid'], 
                linewidth=0.5, alpha=0.6)
    
    def _plot_directivity_index(self, ax):
        """Plot directivity index curves."""
        if MultipleLocator is None:
            return
        
        freqs = self.data.frequencies
        
        ax.plot(freqs, self.data.directivity_index,
                color=CEA_COLORS['directivity'], linewidth=CEA_LINEWIDTH,
                label='Sound Power DI (SPDI)', zorder=4)
        
        ax.plot(freqs, self.data.early_reflections_di,
                color=CEA_COLORS['early_di'], linewidth=CEA_LINEWIDTH,
                label='Early Reflections DI (ERDI)', linestyle='--', zorder=3)
        
        # Zero reference line
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
        
        self._setup_freq_axis(ax)
        ax.set_ylabel('Directivity Index (dB)', fontsize=11, fontweight='bold')
        
        # CEA-2034A standard: DI range 0 to 15 dB (positive values only)
        ax.set_ylim(0, 15)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        
        ax.legend(loc='upper left', framealpha=0.95, fontsize=9)
        ax.grid(True, which='both', color=CEA_COLORS['grid'],
                linewidth=0.5, alpha=0.6)
    
    def _plot_early_reflections(self, ax):
        """Plot early reflections detail."""
        freqs = self.data.frequencies
        
        ax.plot(freqs, self.data.on_axis,
                color=CEA_COLORS['on_axis'], linewidth=CEA_LINEWIDTH,
                label='On-Axis', zorder=5)
        
        ax.plot(freqs, self.data.early_reflections,
                color=CEA_COLORS['early_reflections'], linewidth=CEA_LINEWIDTH,
                label='Early Reflections', zorder=4)
        
        # Fill between to show difference
        ax.fill_between(freqs, self.data.on_axis, self.data.early_reflections,
                        color=CEA_COLORS['early_reflections'], alpha=0.15)
        
        self._setup_freq_axis(ax)
        self._setup_spl_axis(ax)
        ax.set_ylabel('Relative SPL (dB)', fontsize=11, fontweight='bold')
        ax.legend(loc='lower left', framealpha=0.95, fontsize=9)
        ax.grid(True, which='both', color=CEA_COLORS['grid'],
                linewidth=0.5, alpha=0.6)
    
    def _plot_sound_power_comparison(self, ax):
        """Plot sound power vs on-axis comparison."""
        freqs = self.data.frequencies
        
        ax.plot(freqs, self.data.on_axis,
                color=CEA_COLORS['on_axis'], linewidth=CEA_LINEWIDTH,
                label='On-Axis', zorder=5)
        
        ax.plot(freqs, self.data.sound_power,
                color=CEA_COLORS['sound_power'], linewidth=CEA_LINEWIDTH,
                label='Sound Power', zorder=4)
        
        # Fill between
        ax.fill_between(freqs, self.data.on_axis, self.data.sound_power,
                        color=CEA_COLORS['sound_power'], alpha=0.15)
        
        self._setup_freq_axis(ax)
        self._setup_spl_axis(ax)
        ax.set_ylabel('Relative SPL (dB)', fontsize=11, fontweight='bold')
        ax.legend(loc='lower left', framealpha=0.95, fontsize=9)
        ax.grid(True, which='both', color=CEA_COLORS['grid'],
                linewidth=0.5, alpha=0.6)
    
    def _plot_horizontal_contour(self, ax):
        """Plot horizontal directivity contour map (2D plane approximation)."""
        freqs = self.data.frequencies

        # Use a denser angle grid than the standard curves.
        angles = np.arange(0.0, 91.0, 5.0)
        sweep = self.response.polar_sweep(angles_deg=angles.tolist(), distance_m=float(self.distance), plane="horizontal")
        spl = sweep.spl_db()

        # Normalize to the same reference as the other panels.
        spl -= float(self.data.reference_level)

        # pcolormesh expects increasing coordinates; use angle (0..90).
        X, Y = np.meshgrid(freqs, angles, indexing="xy")
        Z = spl.T  # (n_angles, n_freq)

        im = ax.pcolormesh(
            X,
            Y,
            Z,
            shading="auto",
            cmap="viridis",
            vmin=CEA_SPL_MIN,
            vmax=CEA_SPL_MAX,
        )
        ax.set_ylim(0, 90)
        ax.set_ylabel("Angle (degrees)", fontsize=11, fontweight="bold")
        self._setup_freq_axis(ax, no_ylabel=True)
        ax.grid(False)

        try:
            cbar = plt.colorbar(im, ax=ax, pad=0.01)
            cbar.set_label("Relative SPL (dB)", fontsize=10)
        except Exception:
            pass
    
    def _setup_freq_axis(self, ax, no_ylabel: bool = False):
        """Configure frequency axis to CEA-2034A standard."""
        ax.set_xlim(CEA_FREQ_MIN, CEA_FREQ_MAX)
        ax.set_xscale('log')
        ax.set_xticks(CEA_FREQ_TICKS)
        ax.set_xticklabels(CEA_FREQ_LABELS)
        if not no_ylabel:
            ax.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
    
    def _setup_spl_axis(self, ax):
        """Configure SPL axis to CEA-2034A standard."""
        if MultipleLocator is None or FuncFormatter is None:
            return
        
        ax.set_ylim(CEA_SPL_MIN, CEA_SPL_MAX)
        ax.yaxis.set_major_locator(MultipleLocator(CEA_SPL_STEP))
        
        # Format y-axis labels with explicit sign
        def format_db(x, pos):
            if x >= 0:
                return f'+{int(x)}'
            else:
                return f'{int(x)}'
        ax.yaxis.set_major_formatter(FuncFormatter(format_db))


# =============================================================================
# Convenience function
# =============================================================================

def plot_cea2034(
    response: "FrequencyResponse",
    chart_type: str = "spinorama",
    save: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
) -> None:
    """
    Convenience function to generate CEA-2034A charts.
    
    Parameters
    ----------
    response : FrequencyResponse
        Acoustic response data.
    chart_type : str
        Chart type: 'spinorama', 'standard', or 'full_report'.
    save : str, optional
        Filename to save chart.
    title : str, optional
        Custom chart title.
    **kwargs
        Additional arguments passed to CEA2034Chart.
    
    Examples
    --------
    >>> from bempp_audio.viz import plot_cea2034
    >>> plot_cea2034(response, chart_type='spinorama', save='spinorama.png')
    >>> plot_cea2034(response, chart_type='full_report', save='report.png')
    """
    chart = CEA2034Chart(response, **kwargs)
    
    if chart_type == 'spinorama':
        chart.plot_spinorama(save=save, title=title or "Spinorama - CEA-2034A", show=False)
    elif chart_type == 'standard':
        chart.plot_standard(save=save, title=title or "CEA-2034A Standard", show=False)
    elif chart_type == 'full_report':
        chart.plot_full_report(save=save, title=title or "CEA-2034A Full Report", show=False)
    else:
        raise ValueError(f"Unknown chart_type: {chart_type}. Use 'spinorama', 'standard', or 'full_report'.")
