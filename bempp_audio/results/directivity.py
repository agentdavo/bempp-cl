"""
Directivity pattern computation and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING, Optional
import numpy as np

from bempp_audio._optional import optional_import

bempp, BEMPP_AVAILABLE = optional_import("bempp_cl.api")
from bempp_audio.baffles import InfiniteBaffle
from bempp_audio.results._eval_utils import cache_key_for_array, mask_directions_front_hemisphere
from bempp_audio.api.types import Plane2DLike, normalize_plane_2d

if TYPE_CHECKING:
    from bempp_audio.results.radiation_result import RadiationResult


@dataclass
class DirectivityBalloon:
    """
    Container for 3D directivity balloon data.

    Attributes
    ----------
    theta : np.ndarray
        Polar angle grid (0 to π).
    phi : np.ndarray
        Azimuthal angle grid (0 to 2π).
    pattern : np.ndarray
        Complex far-field pattern, shape matches theta/phi grids.
    frequency : float
        Frequency in Hz.
    """

    theta: np.ndarray
    phi: np.ndarray
    pattern: np.ndarray
    frequency: float

    def magnitude_db(self, normalize: bool = True) -> np.ndarray:
        """
        Get pattern magnitude in dB.

        Parameters
        ----------
        normalize : bool
            If True, normalize to 0 dB at maximum.

        Returns
        -------
        np.ndarray
            Magnitude in dB.
        """
        mag = np.abs(self.pattern)
        mag = np.maximum(mag, 1e-20)  # Avoid log(0)

        db = 20 * np.log10(mag)
        if normalize:
            db -= db.max()
        return db

    def directivity_index(self, *, solid_angle: float = 4.0 * np.pi) -> float:
        """
        Compute Directivity Index (DI).

        DI = 10 * log10(max_intensity / average_intensity)

        Notes
        -----
        `average_intensity` is computed as the intensity integrated over the
        sampled angular region divided by `solid_angle`. For a full-space
        average use `solid_angle=4π`. For an infinite-baffle (hemispherical)
        convention use `solid_angle=2π`.

        Returns
        -------
        float
            Directivity Index in dB.
        """
        intensity = np.abs(self.pattern) ** 2

        # Average over sphere (weighted by sin(theta)) using trapezoidal integration.
        if self.theta.ndim == 2 and self.phi.ndim == 2:
            theta = self.theta
            phi = self.phi
        else:
            theta_1d = np.asarray(self.theta, dtype=float).ravel()
            phi_1d = np.asarray(self.phi, dtype=float).ravel()
            if theta_1d.size < 2 or phi_1d.size < 2:
                raise ValueError("theta and phi must have at least 2 samples each")
            theta, phi = np.meshgrid(theta_1d, phi_1d, indexing="ij")
            if intensity.shape != theta.shape:
                intensity = np.asarray(intensity).reshape(theta.shape)

        sin_theta = np.sin(theta)
        integrand = intensity * sin_theta
        # NumPy < 2.0 uses `trapz` (no `trapezoid` alias).
        integral_phi = np.trapz(integrand, x=phi[0, :], axis=1)
        integral = float(np.trapz(integral_phi, x=theta[:, 0], axis=0))
        if float(solid_angle) <= 0:
            raise ValueError("solid_angle must be positive")
        avg_intensity = integral / float(solid_angle)

        max_intensity = float(np.max(intensity))
        if avg_intensity <= 0:
            return np.inf
        return 10.0 * np.log10(max_intensity / avg_intensity)

    def to_cartesian(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert to Cartesian coordinates for 3D plotting.

        Returns magnitude-scaled coordinates suitable for balloon plots.

        Returns
        -------
        tuple
            (X, Y, Z) coordinate arrays.
        """
        r = np.abs(self.pattern)
        r = r / r.max()  # Normalize

        X = r * np.sin(self.theta) * np.cos(self.phi)
        Y = r * np.sin(self.theta) * np.sin(self.phi)
        Z = r * np.cos(self.theta)

        return X, Y, Z


class DirectivityPattern:
    """
    Compute and analyze directivity patterns.

    Provides methods for computing 2D polar patterns, 3D balloons,
    and derived metrics like Directivity Index and beamwidth.

    Parameters
    ----------
    result : RadiationResult
        The radiation solution.

    Examples
    --------
    >>> dp = DirectivityPattern(result)
    >>> theta, pattern = dp.polar_2d(plane='xz')
    >>> di = dp.directivity_index()
    """

    def __init__(self, result: "RadiationResult"):
        self.result = result
        self.space = result.surface_pressure.space
        self.k = result.wavenumber
        self._ff_slp = None
        self._ff_dlp = None
        self._last_directions_key = None

    def far_field_at(self, directions: np.ndarray) -> np.ndarray:
        """
        Evaluate far-field pattern at given directions.

        Parameters
        ----------
        directions : np.ndarray
            Unit direction vectors, shape (3, n_directions).

        Returns
        -------
        np.ndarray
            Far-field pattern values.
        """
        if not BEMPP_AVAILABLE:
            raise ImportError("bempp_cl is required for far-field evaluation")

        directions = np.asarray(directions)
        if directions.ndim == 1:
            directions = directions.reshape(3, 1)

        # Infinite-baffle post-processing: suppress rear half-space.
        # This is a pragmatic approximation layered on top of a full-space solve.
        if isinstance(getattr(self.result, "baffle", None), InfiniteBaffle):
            # For far-field, use direction z component to decide hemisphere.
            # (Direction vectors are unitless; the baffle plane is assumed to be z=0.)
            mask, directions_eval = mask_directions_front_hemisphere(directions)
            out = np.zeros(directions.shape[1], dtype=complex)
            if not np.any(mask):
                return out
        else:
            directions_eval = directions

        # Normalize directions
        norms = np.linalg.norm(directions_eval, axis=0, keepdims=True)
        directions_eval = directions_eval / norms

        # Neumann data from prescribed normal velocity: ∂p/∂n = -iωρ vₙ
        neumann_gf = self.result.neumann_grid_function(self.space)

        # Far-field representation (consistent with PressureField.at_points):
        # p∞(d) = S∞(∂p/∂n) - D∞(p)
        directions_key = cache_key_for_array(directions_eval)
        if self._last_directions_key == directions_key and self._ff_slp is not None and self._ff_dlp is not None:
            ff_slp = self._ff_slp
            ff_dlp = self._ff_dlp
        else:
            ff_slp = bempp.operators.far_field.helmholtz.single_layer(self.space, directions_eval, self.k)
            ff_dlp = bempp.operators.far_field.helmholtz.double_layer(self.space, directions_eval, self.k)
            self._ff_slp = ff_slp
            self._ff_dlp = ff_dlp
            self._last_directions_key = directions_key

        pattern = (ff_slp @ neumann_gf) - (ff_dlp @ self.result.surface_pressure)
        pattern = pattern.ravel()
        if isinstance(getattr(self.result, "baffle", None), InfiniteBaffle):
            out[mask] = pattern
            return out
        return pattern

    def polar_2d(
        self,
        n_angles: int = 360,
        plane: Plane2DLike = "xz",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 2D polar directivity in a specified plane.

        Parameters
        ----------
        n_angles : int
            Number of angles in the pattern.
        plane : str
            Coordinate plane to sample:
            - `'xz'`: cut in the x–z plane (contains +z axis).
            - `'yz'`: cut in the y–z plane (contains +z axis).
            - `'xy'`: azimuthal cut in the x–y plane (z=0, does not include +z axis).

        Returns
        -------
        tuple
            (theta, pattern) where theta is angles in radians
            and pattern is complex far-field values.
        """
        theta = np.linspace(0, 2 * np.pi, n_angles)

        plane_norm = normalize_plane_2d(plane)
        if plane_norm == "xz":
            directions = np.array([
                np.sin(theta),
                np.zeros_like(theta),
                np.cos(theta),
            ])
        elif plane_norm == "xy":
            directions = np.array([
                np.cos(theta),
                np.sin(theta),
                np.zeros_like(theta),
            ])
        elif plane_norm == "yz":
            directions = np.array([
                np.zeros_like(theta),
                np.sin(theta),
                np.cos(theta),
            ])
        else:
            raise ValueError(f"Unknown plane: {plane}")

        pattern = self.far_field_at(directions)
        return theta, pattern

    def balloon_3d(
        self,
        n_theta: int = 37,
        n_phi: int = 73,
    ) -> DirectivityBalloon:
        """
        Compute full 3D directivity balloon.

        Parameters
        ----------
        n_theta : int
            Number of polar angles (0 to π).
        n_phi : int
            Number of azimuthal angles (0 to 2π).

        Returns
        -------
        DirectivityBalloon
            3D directivity data container.
        """
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

        directions = np.array([
            np.sin(THETA.ravel()) * np.cos(PHI.ravel()),
            np.sin(THETA.ravel()) * np.sin(PHI.ravel()),
            np.cos(THETA.ravel()),
        ])

        pattern = self.far_field_at(directions)
        pattern = pattern.reshape(THETA.shape)

        return DirectivityBalloon(
            theta=THETA,
            phi=PHI,
            pattern=pattern,
            frequency=self.result.frequency,
        )

    def directivity_index(self) -> float:
        """
        Compute Directivity Index (DI).

        Returns
        -------
        float
            DI in dB.
        """
        balloon = self.balloon_3d()
        solid_angle = 2.0 * np.pi if isinstance(getattr(self.result, "baffle", None), InfiniteBaffle) else 4.0 * np.pi
        return balloon.directivity_index(solid_angle=solid_angle)

    def spl_at_angle(self, theta: float, plane: Plane2DLike = "xz") -> float:
        """
        Get normalized SPL at a specific angle (relative to on-axis).

        Parameters
        ----------
        theta : float
            Angle from axis in radians.
        plane : str
            Plane for measurement. For consistency with the rest of the library,
            `'xz'` and `'yz'` are interpreted as deflections away from +z within
            those planes. `'xy'` is treated as an alias for `'xz'`.

        Returns
        -------
        float
            SPL normalized to on-axis (dB, typically negative for off-axis).
        """
        # Build direction vectors for on-axis and at angle (relative to +z).
        plane_norm = normalize_plane_2d(plane)
        if plane_norm in ("xz", "xy"):
            lateral = np.array([[1.0], [0.0], [0.0]])
        elif plane_norm == "yz":
            lateral = np.array([[0.0], [1.0], [0.0]])
        else:
            raise ValueError(f"Unknown plane: {plane}")

        dir_on_axis = np.array([[0.0], [0.0], [1.0]])
        dir_at_angle = (np.cos(theta) * dir_on_axis) + (np.sin(theta) * lateral)

        # Get far-field at both angles
        p_on_axis = self.far_field_at(dir_on_axis)[0]
        p_at_angle = self.far_field_at(dir_at_angle)[0]

        # Compute normalized SPL (in dB)
        ratio = np.abs(p_at_angle) / np.abs(p_on_axis) if np.abs(p_on_axis) > 1e-20 else 0
        if ratio > 1e-20:
            return 20 * np.log10(ratio)
        else:
            return -60.0  # Floor at -60 dB

    def beamwidth(
        self,
        level_db: float = -6.0,
        plane: Plane2DLike = "xz",
    ) -> float:
        """
        Compute beamwidth at a given level below maximum.

        Parameters
        ----------
        level_db : float
            Level below peak (negative dB).
        plane : str
            Plane for measurement.

        Returns
        -------
        float
            Beamwidth in degrees.
        """
        theta, pattern = self.polar_2d(n_angles=361, plane=plane)
        # `pattern` can contain exact zeros (e.g., infinite-baffle rear suppression).
        # Clamp to avoid -inf and noisy runtime warnings; callers can still apply
        # their own display floors.
        mag = np.maximum(np.abs(pattern), 1e-20)
        mag_db = 20 * np.log10(mag)
        mag_db -= mag_db.max()  # Normalize to 0 dB

        # Find first crossing of level on each side
        # Assume maximum is near theta=0 (forward direction)
        n = len(theta)

        # Find peak index
        peak_idx = np.argmax(np.abs(pattern))

        # Search forward from peak
        forward_angle = None
        for i in range(peak_idx, min(peak_idx + n // 2, n)):
            if mag_db[i] < level_db:
                forward_angle = theta[i]
                break

        # Search backward from peak
        backward_angle = None
        for i in range(peak_idx, max(peak_idx - n // 2, -1), -1):
            if mag_db[i] < level_db:
                backward_angle = theta[i]
                break

        if forward_angle is None or backward_angle is None:
            return 360.0  # Very wide pattern

        beamwidth_rad = abs(forward_angle - backward_angle)
        return np.degrees(beamwidth_rad)

    def __repr__(self) -> str:
        return f"DirectivityPattern(f={self.result.frequency:.1f}Hz)"


# =============================================================================
# Frequency Sweep Directivity Metrics
# =============================================================================


@dataclass
class DirectivitySweepMetrics:
    """Metrics for evaluating directivity behavior across frequency.

    These metrics help assess waveguide design quality for "smooth"
    directivity targets:
    - Low DI ripple (smooth transition vs frequency)
    - Monotonic beamwidth narrowing with frequency
    - Consistent coverage angle

    Attributes
    ----------
    frequencies : np.ndarray
        Frequency points (Hz).
    di_values : np.ndarray
        Directivity Index at each frequency (dB).
    beamwidth_values : np.ndarray
        Beamwidth at each frequency (degrees).
    """

    frequencies: np.ndarray
    di_values: np.ndarray
    beamwidth_values: np.ndarray

    @property
    def di_ripple_db(self) -> float:
        """Compute DI ripple (peak-to-peak variation).

        Lower is better. Good waveguides have DI ripple < 2 dB.

        Returns
        -------
        float
            Peak-to-peak DI variation in dB.
        """
        return float(np.max(self.di_values) - np.min(self.di_values))

    @property
    def di_std_db(self) -> float:
        """Standard deviation of DI values.

        Returns
        -------
        float
            Standard deviation in dB.
        """
        return float(np.std(self.di_values))

    @property
    def beamwidth_monotonicity(self) -> float:
        """Measure how monotonically beamwidth decreases with frequency.

        Returns a value in [0, 1]:
        - 1.0 = perfectly monotonic (beamwidth always decreases)
        - 0.0 = highly non-monotonic

        Computed as fraction of frequency steps where beamwidth decreases.

        Returns
        -------
        float
            Monotonicity score in [0, 1].
        """
        if len(self.beamwidth_values) < 2:
            return 1.0

        diffs = np.diff(self.beamwidth_values)
        n_decreasing = np.sum(diffs <= 0)
        return float(n_decreasing / len(diffs))

    @property
    def beamwidth_ripple_deg(self) -> float:
        """Compute beamwidth ripple (peak-to-peak variation).

        Returns
        -------
        float
            Peak-to-peak beamwidth variation in degrees.
        """
        return float(np.max(self.beamwidth_values) - np.min(self.beamwidth_values))

    @property
    def coverage_angle_mean_deg(self) -> float:
        """Mean coverage angle (beamwidth) across frequency.

        Returns
        -------
        float
            Mean beamwidth in degrees.
        """
        return float(np.mean(self.beamwidth_values))

    def is_smooth(
        self,
        max_di_ripple_db: float = 3.0,
        min_monotonicity: float = 0.7,
    ) -> bool:
        """Check if directivity meets smoothness criteria.

        Parameters
        ----------
        max_di_ripple_db : float
            Maximum allowed DI ripple (default 3 dB).
        min_monotonicity : float
            Minimum beamwidth monotonicity score (default 0.7).

        Returns
        -------
        bool
            True if directivity is smooth.
        """
        return (
            self.di_ripple_db <= max_di_ripple_db
            and self.beamwidth_monotonicity >= min_monotonicity
        )

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            f"Directivity Sweep Metrics ({len(self.frequencies)} frequencies)",
            f"  Frequency range: {self.frequencies[0]:.0f} - {self.frequencies[-1]:.0f} Hz",
            f"  DI range: {np.min(self.di_values):.1f} - {np.max(self.di_values):.1f} dB",
            f"  DI ripple: {self.di_ripple_db:.1f} dB",
            f"  DI std: {self.di_std_db:.1f} dB",
            f"  Beamwidth range: {np.min(self.beamwidth_values):.0f} - {np.max(self.beamwidth_values):.0f} deg",
            f"  Beamwidth monotonicity: {self.beamwidth_monotonicity:.1%}",
            f"  Mean coverage: {self.coverage_angle_mean_deg:.0f} deg",
        ]
        return "\n".join(lines)


def compute_directivity_sweep_metrics(
    results: list,
    beamwidth_level_db: float = -6.0,
    plane: Plane2DLike = "xz",
) -> DirectivitySweepMetrics:
    """Compute directivity metrics from a frequency sweep of RadiationResults.

    Parameters
    ----------
    results : list of RadiationResult
        Results from frequency sweep, must have .frequency attribute.
    beamwidth_level_db : float
        Level for beamwidth calculation (default -6 dB).
    plane : str
        Plane for beamwidth calculation ('xz', 'yz', 'xy').

    Returns
    -------
    DirectivitySweepMetrics
        Computed metrics.

    Examples
    --------
    >>> from bempp_audio.results import compute_directivity_sweep_metrics
    >>> metrics = compute_directivity_sweep_metrics(frequency_response.results)
    >>> print(metrics.summary())
    >>> if metrics.is_smooth():
    ...     print("Directivity meets smoothness targets")
    """
    if not results:
        raise ValueError("results list cannot be empty")

    # Sort by frequency
    sorted_results = sorted(results, key=lambda r: r.frequency)

    frequencies = np.array([r.frequency for r in sorted_results])
    di_values = np.zeros(len(sorted_results))
    beamwidth_values = np.zeros(len(sorted_results))

    for i, result in enumerate(sorted_results):
        dp = DirectivityPattern(result)
        di_values[i] = dp.directivity_index()
        beamwidth_values[i] = dp.beamwidth(level_db=beamwidth_level_db, plane=plane)

    return DirectivitySweepMetrics(
        frequencies=frequencies,
        di_values=di_values,
        beamwidth_values=beamwidth_values,
    )
