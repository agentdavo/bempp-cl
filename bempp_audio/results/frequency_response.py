"""
Frequency response container for multi-frequency results.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, TYPE_CHECKING, Dict
import numpy as np

from bempp_audio.progress import ProgressTracker
from bempp_audio.api.types import Plane2DLike, PolarPlaneLike, normalize_polar_plane

if TYPE_CHECKING:
    from bempp_audio.results.radiation_result import RadiationResult


class FrequencyResponse:
    """
    Container for frequency-dependent acoustic results.

    Manages results from multiple frequencies and provides methods
    for computing SPL, phase, group delay, and directivity vs frequency.

    Examples
    --------
    >>> response = solver.solve_frequencies(frequencies, velocity)
    >>> freqs, spl = response.spl_at_point(point)
    >>> response.plot_spl()
    """

    def __init__(self):
        self._results: List["RadiationResult"] = []
        self._frequencies: List[float] = []
        self._sorted: bool = True
        # Driver metrics (populated when using compression driver model)
        self._driver_impedance: Dict[float, Tuple[float, float]] = {}  # f -> (Re, Im) in ohms
        self._driver_excursion_mm: Dict[float, float] = {}  # f -> peak excursion in mm
        self._driver_diaphragm_area: Optional[float] = None  # Sd in m²
        # Waveguide metadata (populated when solving waveguide geometry)
        self._waveguide_metadata: Optional[object] = None  # WaveguideMetadata

    def add(self, result: "RadiationResult"):
        """Add a result at a new frequency."""
        f = float(result.frequency)
        if self._frequencies and f < float(self._frequencies[-1]):
            self._sorted = False
        self._results.append(result)
        self._frequencies.append(f)

    def _ensure_sorted(self) -> None:
        """Ensure internal results are sorted by frequency."""
        if self._sorted:
            return
        pairs = list(zip(self._frequencies, self._results))
        pairs.sort(key=lambda x: x[0])
        self._frequencies = [p[0] for p in pairs]
        self._results = [p[1] for p in pairs]
        self._sorted = True

    @property
    def frequencies(self) -> np.ndarray:
        """Array of frequencies in Hz."""
        self._ensure_sorted()
        return np.array(self._frequencies)

    @property
    def results(self) -> List["RadiationResult"]:
        """List of RadiationResult objects."""
        self._ensure_sorted()
        return self._results

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, index) -> "RadiationResult":
        self._ensure_sorted()
        return self._results[index]

    # =========================================================================
    # Driver metrics (compression driver mode)
    # =========================================================================

    def add_driver_metrics(
        self,
        frequency: float,
        impedance_re: float,
        impedance_im: float,
        excursion_mm: float,
    ):
        """Add driver metrics for a frequency point."""
        f = float(frequency)
        self._driver_impedance[f] = (float(impedance_re), float(impedance_im))
        self._driver_excursion_mm[f] = float(excursion_mm)

    def set_driver_diaphragm_area(self, area_m2: float):
        """Set the driver diaphragm area (for reference)."""
        self._driver_diaphragm_area = area_m2

    @property
    def has_driver_metrics(self) -> bool:
        """True if driver metrics are available."""
        return bool(self._driver_impedance)

    @property
    def driver_impedance(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Driver electrical impedance vs frequency.

        Returns
        -------
        tuple or None
            (impedance_re, impedance_im) arrays in ohms, or None if not available.
        """
        if not self._driver_impedance:
            return None
        freqs = self.frequencies
        re = np.array([self._driver_impedance.get(float(f), (np.nan, np.nan))[0] for f in freqs], dtype=float)
        im = np.array([self._driver_impedance.get(float(f), (np.nan, np.nan))[1] for f in freqs], dtype=float)
        return re, im

    @property
    def driver_excursion_mm(self) -> Optional[np.ndarray]:
        """
        Peak driver excursion vs frequency in mm.

        Returns
        -------
        np.ndarray or None
            Peak excursion in mm, or None if not available.
        """
        if not self._driver_excursion_mm:
            return None
        freqs = self.frequencies
        return np.array([self._driver_excursion_mm.get(float(f), np.nan) for f in freqs], dtype=float)

    # =========================================================================
    # Waveguide metadata
    # =========================================================================

    @property
    def waveguide_metadata(self) -> Optional[object]:
        """Waveguide metadata (geometry, profile, etc.) if available."""
        return self._waveguide_metadata

    def set_waveguide_metadata(self, metadata: object):
        """Set the waveguide metadata for plotting."""
        self._waveguide_metadata = metadata

    @property
    def has_waveguide(self) -> bool:
        """True if waveguide metadata is available."""
        return self._waveguide_metadata is not None

    def spl_at_point(
        self,
        point: np.ndarray,
        ref: float = 20e-6,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SPL vs frequency at a single point.

        Parameters
        ----------
        point : np.ndarray
            Position (3,).
        ref : float
            Reference pressure.
        show_progress : bool
            If True, show progress bar. Default True.

        Returns
        -------
        tuple
            (frequencies, spl_db)
        """
        self._ensure_sorted()
        point = np.asarray(point).reshape(3, 1)

        spl = []
        with ProgressTracker(
            total=len(self._results),
            desc="Computing SPL",
            unit="freq",
            disable=not show_progress
        ) as pbar:
            for result in self._results:
                pressure = result.pressure_at(point)
                p_rms = np.abs(pressure[0]) / np.sqrt(2)
                spl_val = 20 * np.log10(max(p_rms, 1e-20) / ref)
                spl.append(spl_val)
                pbar.update(item=f"{result.frequency:.0f} Hz")

        return self.frequencies, np.array(spl)

    def phase_at_point(
        self,
        point: np.ndarray,
        unwrap: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute phase vs frequency at a single point.

        Parameters
        ----------
        point : np.ndarray
            Position (3,).
        unwrap : bool
            If True, unwrap phase to avoid discontinuities.

        Returns
        -------
        tuple
            (frequencies, phase_rad)
        """
        self._ensure_sorted()
        point = np.asarray(point).reshape(3, 1)

        phase = []
        for result in self._results:
            pressure = result.pressure_at(point)
            phase.append(np.angle(pressure[0]))

        phase = np.array(phase)
        if unwrap:
            phase = np.unwrap(phase)

        return self.frequencies, phase

    def group_delay(
        self,
        point: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute group delay vs frequency.

        Group delay = -d(phase)/d(omega)

        Parameters
        ----------
        point : np.ndarray
            Position (3,).

        Returns
        -------
        tuple
            (frequencies, group_delay_seconds)
        """
        self._ensure_sorted()
        freqs, phase = self.phase_at_point(point, unwrap=True)
        omega = 2 * np.pi * freqs

        # Numerical derivative
        gd = -np.gradient(phase, omega)

        return freqs, gd

    def on_axis_response(
        self,
        distance: float = 1.0,
        ref: float = 20e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute on-axis SPL vs frequency.

        Parameters
        ----------
        distance : float
            Distance from radiator center.
        ref : float
            Reference pressure.

        Returns
        -------
        tuple
            (frequencies, spl_db)
        """
        self._ensure_sorted()
        # Get axis from first result
        if not self._results:
            return np.array([]), np.array([])

        from bempp_audio.acoustic_reference import AcousticReference

        reference = self._results[0].reference or AcousticReference.from_mesh(self._results[0].mesh)
        point = reference.point_on_axis(distance_m=float(distance))

        return self.spl_at_point(point, ref)

    def directivity_index_vs_freq(
        self,
        angle: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Directivity Index vs frequency at a given angle.

        The DI at angle θ represents the SPL at that angle relative to
        the average SPL over a sphere, in dB.

        Parameters
        ----------
        angle : float
            Angle from axis in degrees (default 0 = on-axis).

        Returns
        -------
        tuple
            (frequencies, DI_db)
        """
        self._ensure_sorted()
        di = []
        for result in self._results:
            dp = result.directivity()
            if angle == 0.0:
                # Standard on-axis DI
                di.append(dp.directivity_index())
            else:
                # DI at angle: SPL(θ) - SPL_avg
                # This shows how the response at angle θ compares to average
                di_on_axis = dp.directivity_index()
                # Get normalized response at this angle
                theta_rad = np.radians(angle)
                spl_norm = dp.spl_at_angle(theta_rad)  # Normalized to on-axis
                # DI at angle = DI_on_axis + SPL_norm (which is negative)
                di.append(di_on_axis + spl_norm)

        return self.frequencies, np.array(di)

    def beamwidth_vs_freq(
        self,
        level_db: float = -6.0,
        plane: Plane2DLike = "xz",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute beamwidth vs frequency.

        Parameters
        ----------
        level_db : float
            Level below peak (negative dB).
        plane : str
            Plane for measurement.

        Returns
        -------
        tuple
            (frequencies, beamwidth_degrees)
        """
        self._ensure_sorted()
        bw = []
        for result in self._results:
            dp = result.directivity()
            bw.append(dp.beamwidth(level_db, plane))

        return self.frequencies, np.array(bw)

    def radiated_power_vs_freq(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute radiated power vs frequency.

        Returns
        -------
        tuple
            (frequencies, power_watts)
        """
        self._ensure_sorted()
        power = [r.radiated_power() for r in self._results]
        return self.frequencies, np.array(power)

    def sound_power_level_vs_freq(
        self,
        ref: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sound power level vs frequency.

        Parameters
        ----------
        ref : float
            Reference power (default 1 pW).

        Returns
        -------
        tuple
            (frequencies, SWL_db)
        """
        self._ensure_sorted()
        swl = [r.sound_power_level(ref) for r in self._results]
        return self.frequencies, np.array(swl)

    def polar_sweep(
        self,
        *,
        angles_deg: List[float],
        distance_m: float = 1.0,
        plane: PolarPlaneLike = "horizontal",
        show_progress: bool = False,
    ) -> "PolarSweep":
        """
        Compute a 2D polar sweep using field evaluation at a fixed distance.

        Conventions
        ----------
        - Angles are degrees from the mesh axis (0° = on-axis).
        - `plane="horizontal"` prefers global +x as lateral direction.
        - `plane="vertical"` prefers global +y as lateral direction.
        """
        self._ensure_sorted()
        from bempp_audio.results.polar import compute_polar_sweep

        return compute_polar_sweep(
            self,
            angles_deg=angles_deg,
            distance_m=distance_m,
            plane=normalize_polar_plane(plane),  # type: ignore[arg-type]
            show_progress=bool(show_progress),
        )

    def interpolate_at(
        self,
        frequency: float,
        point: np.ndarray,
    ) -> complex:
        """
        Interpolate pressure at an arbitrary frequency.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.
        point : np.ndarray
            Position (3,).

        Returns
        -------
        complex
            Interpolated pressure.
        """
        self._ensure_sorted()
        from scipy.interpolate import interp1d

        point = np.asarray(point).reshape(3, 1)

        # Get pressure at all frequencies
        pressures = []
        for result in self._results:
            p = result.pressure_at(point)
            pressures.append(p[0])

        pressures = np.array(pressures)

        # Interpolate magnitude and phase separately
        mags = np.abs(pressures)
        phases = np.unwrap(np.angle(pressures))

        mag_interp = interp1d(
            self._frequencies, mags, kind="cubic", fill_value="extrapolate"
        )
        phase_interp = interp1d(
            self._frequencies, phases, kind="cubic", fill_value="extrapolate"
        )

        mag = mag_interp(frequency)
        phase = phase_interp(frequency)

        return mag * np.exp(1j * phase)

    def save_summary(
        self,
        filename: str = "acoustic_summary.png",
        title: Optional[str] = None,
        distance: float = 1.0,
        angles: Optional[List[float]] = None,
        max_angle: float = 90.0,
        normalize_angle: float = 0.0,
        include_driver: Optional[bool] = None,
        include_waveguide: Optional[bool] = None,
    ) -> None:
        """
        Save summary plot (adaptive panels).
        
        Generates a comprehensive overview of the acoustic response:
        - Top-left: SPL vs frequency at multiple angles
        - Top-right: Polar directivity map
        - Bottom-left: Radiation impedance (real and imaginary parts)
        - Bottom-right: Directivity index vs frequency
        - Optional (driver mode): electrical impedance and excursion panels
        - Optional (waveguide mode): cross-section panel
        
        Parameters
        ----------
        filename : str
            Output filename (default: "acoustic_summary.png").
        title : str, optional
            Plot title (auto-generated from geometry if None).
        distance : float
            Measurement distance in meters (default 1.0).
        angles : list of float, optional
            SPL curve angles in degrees (default: [0, 15, 30, 45, 60, 90]).
        max_angle : float
            Maximum polar angle in degrees (default 90).
        normalize_angle : float
            Normalization angle in degrees (default 0 = on-axis).
            
        Examples
        --------
        >>> # Simple summary with defaults
        >>> response.save_summary("my_speaker.png")
        
        >>> # Custom configuration
        >>> response.save_summary(
        ...     filename="waveguide.png",
        ...     title="Exponential Waveguide",
        ...     distance=1.0,
        ...     angles=[0, 10, 20, 30],
        ...     normalize_angle=10
        ... )
        """
        from bempp_audio.viz.reporting import save_summary

        save_summary(
            self,
            filename=filename,
            title=title,
            distance=distance,
            angles=angles,
            max_angle=max_angle,
            normalize_angle=normalize_angle,
            include_driver=include_driver,
            include_waveguide=include_waveguide,
        )
    
    def save_cea2034(
        self,
        filename_prefix: str = "cea2034",
        measurement_distance: float = 1.0,
        normalize_to_angle: float = 0.0,
        reference_frequency: float = 1000.0,
        charts: str = 'all',
        title: Optional[str] = None,
    ) -> None:
        """
        Save CEA-2034A industry-standard spinorama charts.
        
        Generates ANSI/CEA-2034A compliant loudspeaker measurement charts
        used throughout the audio industry. Charts include:
        - On-axis response
        - Listening window (average 0-30°)
        - Early reflections (floor, ceiling, side walls)
        - Sound power (full sphere average)
        - Directivity indices (SPDI, ERDI)
        
        Parameters
        ----------
        filename_prefix : str
            Filename prefix (default: "cea2034").
            Generates: "{prefix}_spinorama.png", "{prefix}_standard.png", etc.
        measurement_distance : float
            Measurement distance in meters (default 1.0).
        normalize_to_angle : float
            Normalization angle in degrees (default 0 = on-axis).
            Some manufacturers normalize to 10° for horn-loaded speakers.
        reference_frequency : float
            Reference frequency for normalization in Hz (default 1000).
        charts : str
            Which charts to generate:
            - 'spinorama': Single-panel standard chart
            - 'standard': 3-panel chart (spinorama + reflections + DI)
            - 'full': 5-panel comprehensive report
            - 'all': Generate all chart types (default)
        title : str, optional
            Chart title (auto-generated if None).
            
        Examples
        --------
        >>> # Generate all CEA-2034A charts with defaults
        >>> response.save_cea2034(filename_prefix="my_speaker")
        
        >>> # Only spinorama chart, normalized to 10° off-axis
        >>> response.save_cea2034(
        ...     filename_prefix="waveguide",
        ...     charts='spinorama',
        ...     normalize_to_angle=10.0
        ... )
        """
        from bempp_audio.viz.reporting import save_cea2034

        save_cea2034(
            self,
            filename_prefix=filename_prefix,
            measurement_distance=measurement_distance,
            normalize_to_angle=normalize_to_angle,
            reference_frequency=reference_frequency,
            charts=charts,
            title=title,
        )
    
    def save_all_plots(
        self,
        prefix: str = "acoustic",
        title: Optional[str] = None,
        distance: float = 1.0,
        max_angle: float = 90.0,
        normalize_angle: float = 0.0,
    ) -> None:
        """
        Save all standard plots (4-panel summary + CEA-2034A charts).
        
        Convenience method that generates a complete set of visualization
        plots for acoustic analysis. Creates:
        
        - {prefix}_summary.png: 4-panel overview (SPL, polar, impedance, DI)
        - {prefix}_spinorama.png: CEA-2034A single-panel chart
        - {prefix}_standard.png: CEA-2034A 3-panel chart
        - {prefix}_full.png: CEA-2034A 5-panel comprehensive report
        
        Parameters
        ----------
        prefix : str
            Filename prefix for all plots (default: "acoustic").
        title : str, optional
            Plot title (auto-generated if None).
        distance : float
            Measurement distance in meters (default 1.0).
        max_angle : float
            Maximum polar angle for plots in degrees (default 90).
        normalize_angle : float
            Normalization angle in degrees (default 0 = on-axis).
            
        Examples
        --------
        >>> # Generate all plots with default settings
        >>> response.save_all_plots()
        
        >>> # Custom configuration for waveguide
        >>> response.save_all_plots(
        ...     prefix="waveguide",
        ...     title="Exponential Waveguide (25mm throat)",
        ...     distance=1.0,
        ...     max_angle=90,
        ...     normalize_angle=10
        ... )
        
        Notes
        -----
        This method replaces 40+ lines of manual plotting code with a single
        function call. All generated plots follow industry standards and are
        suitable for professional documentation.
        """
        from bempp_audio.viz.reporting import save_all_plots

        save_all_plots(
            self,
            prefix=prefix,
            title=title,
            distance=distance,
            max_angle=max_angle,
            normalize_angle=normalize_angle,
        )

    @classmethod
    def from_results(cls, results: List["RadiationResult"]) -> "FrequencyResponse":
        """Create FrequencyResponse from list of results."""
        response = cls()
        for result in results:
            response.add(result)
        return response

    def __repr__(self) -> str:
        if not self._frequencies:
            return "FrequencyResponse(empty)"
        return (
            f"FrequencyResponse({len(self._frequencies)} points, "
            f"{min(self._frequencies):.0f}-{max(self._frequencies):.0f}Hz)"
        )
