"""
Elastic dome integration with compression driver lumped network.

This module bridges shell FEM velocity distributions with the lumped-element
electro-acoustic network model. It enables:
- Replacing rigid piston assumptions with FEM-derived velocity profiles
- Computing frequency-dependent effective area from modal analysis
- Identifying break-up frequency thresholds

The key insight is that the lumped network assumes uniform piston velocity
(v_piston * S_d = volume velocity), but an elastic dome has spatially varying
velocity. We define an "effective area" that preserves volume velocity:

    S_eff(f) = U(f) / v_vc(f)

where U(f) is the actual volume velocity and v_vc(f) is the voice coil velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import numpy as np


@dataclass
class ElasticDomeProfile:
    """
    Frequency-dependent elastic dome velocity profile.

    This captures the deviation from rigid piston behavior as a function
    of frequency, derived from shell FEM analysis.

    Attributes
    ----------
    frequencies_hz : np.ndarray
        Analysis frequencies.
    effective_area_m2 : np.ndarray
        Frequency-dependent effective radiating area.
    volume_velocity_ratio : np.ndarray
        Ratio of actual volume velocity to rigid piston volume velocity.
        Equal to S_eff / S_d (effective area / diaphragm area).
    velocity_distributions : Optional[dict]
        Full velocity distribution at each frequency (for detailed analysis).
    mode_contributions : Optional[dict]
        Modal contribution data at each frequency.
    """
    frequencies_hz: np.ndarray
    effective_area_m2: np.ndarray
    volume_velocity_ratio: np.ndarray
    physical_area_m2: float
    velocity_distributions: Optional[dict] = None
    mode_contributions: Optional[dict] = None

    @classmethod
    def from_rigid_piston(
        cls,
        diaphragm_area_m2: float,
        frequencies_hz: np.ndarray,
    ) -> "ElasticDomeProfile":
        """Create a profile for a perfectly rigid piston (unity ratio)."""
        n = len(frequencies_hz)
        return cls(
            frequencies_hz=np.asarray(frequencies_hz),
            effective_area_m2=np.full(n, diaphragm_area_m2),
            volume_velocity_ratio=np.ones(n),
            physical_area_m2=diaphragm_area_m2,
        )

    @classmethod
    def from_fem_results(
        cls,
        frequencies_hz: np.ndarray,
        velocity_distributions: dict[float, np.ndarray],
        element_areas: np.ndarray,
        voice_coil_velocity: np.ndarray,
        diaphragm_area_m2: float,
    ) -> "ElasticDomeProfile":
        """
        Create profile from FEM velocity distributions.

        Parameters
        ----------
        frequencies_hz : np.ndarray
            Analysis frequencies [Hz].
        velocity_distributions : dict
            Normal velocity at each element, keyed by frequency.
        element_areas : np.ndarray
            Area of each mesh element [m²].
        voice_coil_velocity : np.ndarray
            Voice coil velocity at each frequency [m/s].
        diaphragm_area_m2 : float
            Physical diaphragm area [m²].

        Returns
        -------
        ElasticDomeProfile
            Profile capturing frequency-dependent behavior.
        """
        effective_areas = []
        ratios = []

        for i, freq in enumerate(frequencies_hz):
            v_dist = velocity_distributions[float(freq)]
            v_vc = voice_coil_velocity[i]

            # Volume velocity from actual distribution
            u_actual = np.sum(v_dist * element_areas)

            # Volume velocity from rigid piston
            u_rigid = v_vc * diaphragm_area_m2

            # Effective area
            if np.abs(v_vc) > 1e-30:
                s_eff = np.abs(u_actual / v_vc)
            else:
                s_eff = diaphragm_area_m2

            ratio = np.abs(u_actual) / (np.abs(u_rigid) + 1e-30)

            effective_areas.append(s_eff)
            ratios.append(ratio)

        return cls(
            frequencies_hz=np.asarray(frequencies_hz),
            effective_area_m2=np.asarray(effective_areas),
            volume_velocity_ratio=np.asarray(ratios),
            physical_area_m2=diaphragm_area_m2,
            velocity_distributions=velocity_distributions,
        )

    def interpolate_ratio(self, frequency_hz: float) -> float:
        """Interpolate volume velocity ratio at arbitrary frequency."""
        return float(np.interp(frequency_hz, self.frequencies_hz, self.volume_velocity_ratio))

    def interpolate_effective_area(self, frequency_hz: float) -> float:
        """Interpolate effective area at arbitrary frequency."""
        return float(np.interp(frequency_hz, self.frequencies_hz, self.effective_area_m2))

    def breakup_frequencies(self, threshold_db: float = -3.0) -> np.ndarray:
        """
        Find frequencies where ratio drops below threshold.

        Parameters
        ----------
        threshold_db : float
            Threshold in dB (e.g., -3 dB for half-power point).

        Returns
        -------
        np.ndarray
            Frequencies where breakup begins (ratio crosses threshold).
        """
        threshold_linear = 10 ** (threshold_db / 20)
        crossings = []

        ratio = self.volume_velocity_ratio
        for i in range(len(ratio) - 1):
            if ratio[i] >= threshold_linear and ratio[i+1] < threshold_linear:
                # Linear interpolation to find crossing point
                f_cross = np.interp(
                    threshold_linear,
                    [ratio[i+1], ratio[i]],
                    [self.frequencies_hz[i+1], self.frequencies_hz[i]],
                )
                crossings.append(f_cross)

        return np.array(crossings)


class ElasticDomeNetworkAdapter:
    """
    Adapts CompressionDriverNetwork to use elastic dome velocity profiles.

    This wrapper intercepts volume velocity calculations and applies
    the frequency-dependent effective area correction.

    Parameters
    ----------
    network : CompressionDriverNetwork
        Base lumped-element network.
    dome_profile : ElasticDomeProfile
        Frequency-dependent dome behavior from FEM.

    Examples
    --------
    >>> from bempp_audio.driver.network import CompressionDriverNetwork
    >>> base_network = CompressionDriverNetwork(system_config)
    >>> dome_profile = ElasticDomeProfile.from_fem_results(...)
    >>> adapted = ElasticDomeNetworkAdapter(base_network, dome_profile)
    >>> u = adapted.solve_volume_velocity(1000)  # Accounts for dome elasticity
    """

    def __init__(
        self,
        network: "CompressionDriverNetwork",
        dome_profile: ElasticDomeProfile,
    ):
        self._network = network
        self._profile = dome_profile

        # Store original Sd for comparison
        self._original_sd = network.diaphragm_area_m2

    @property
    def network(self) -> "CompressionDriverNetwork":
        """Underlying lumped network."""
        return self._network

    @property
    def profile(self) -> ElasticDomeProfile:
        """Elastic dome profile."""
        return self._profile

    def solve_volume_velocity(
        self,
        frequency_hz: float,
        excitation=None,
        z_external: complex = np.inf + 0j,
    ) -> complex:
        """
        Solve for volume velocity with elastic dome correction.

        The correction is applied as a multiplicative factor:
            U_elastic = U_rigid * ratio(f)

        Parameters
        ----------
        frequency_hz : float
            Frequency [Hz].
        excitation : CompressionDriverExcitation, optional
            Electrical excitation.
        z_external : complex
            External acoustic load impedance.

        Returns
        -------
        complex
            Volume velocity [m³/s] accounting for dome elasticity.
        """
        # Get rigid piston volume velocity from base network
        if excitation is not None:
            u_rigid = self._network.solve_volume_velocity(
                frequency_hz, excitation, z_external
            )
        else:
            u_rigid = self._network.solve_volume_velocity(
                frequency_hz, z_external=z_external
            )

        # Apply elastic dome correction
        ratio = self._profile.interpolate_ratio(frequency_hz)
        return u_rigid * ratio

    def solve_with_metrics(
        self,
        frequency_hz: float,
        excitation=None,
        z_external: complex = np.inf + 0j,
    ) -> dict:
        """
        Solve with full metrics including elastic dome corrections.

        Returns
        -------
        dict
            Extended metrics dictionary with additional keys:
            - 'volume_velocity_rigid': rigid piston volume velocity
            - 'effective_area_m2': frequency-dependent effective area
            - 'velocity_ratio': elastic/rigid ratio
        """
        # Get base metrics
        if excitation is not None:
            metrics = self._network.solve_with_metrics(
                frequency_hz, excitation, z_external
            )
        else:
            metrics = self._network.solve_with_metrics(
                frequency_hz, z_external=z_external
            )

        # Store rigid piston result
        u_rigid = metrics['volume_velocity']

        # Apply elastic correction
        ratio = self._profile.interpolate_ratio(frequency_hz)
        s_eff = self._profile.interpolate_effective_area(frequency_hz)

        metrics['volume_velocity_rigid'] = u_rigid
        metrics['volume_velocity'] = u_rigid * ratio
        metrics['effective_area_m2'] = s_eff
        metrics['velocity_ratio'] = ratio
        metrics['physical_area_m2'] = self._profile.physical_area_m2

        return metrics

    def frequency_sweep(
        self,
        frequencies_hz: np.ndarray,
        excitation=None,
        z_external_func: Optional[Callable[[float], complex]] = None,
    ) -> dict:
        """
        Run frequency sweep with elastic dome corrections.

        Parameters
        ----------
        frequencies_hz : np.ndarray
            Analysis frequencies [Hz].
        excitation : CompressionDriverExcitation, optional
            Electrical excitation.
        z_external_func : callable, optional
            Function f(freq) -> z_external. If None, uses open circuit.

        Returns
        -------
        dict
            Sweep results with keys:
            - 'frequencies': analysis frequencies
            - 'volume_velocity': complex volume velocity at each frequency
            - 'volume_velocity_rigid': rigid piston reference
            - 'velocity_ratio': elastic/rigid ratio
            - 'effective_area': effective area at each frequency
        """
        results = {
            'frequencies': np.asarray(frequencies_hz),
            'volume_velocity': [],
            'volume_velocity_rigid': [],
            'velocity_ratio': [],
            'effective_area': [],
        }

        for freq in frequencies_hz:
            z_ext = z_external_func(freq) if z_external_func else np.inf + 0j
            metrics = self.solve_with_metrics(freq, excitation, z_ext)

            results['volume_velocity'].append(metrics['volume_velocity'])
            results['volume_velocity_rigid'].append(metrics['volume_velocity_rigid'])
            results['velocity_ratio'].append(metrics['velocity_ratio'])
            results['effective_area'].append(metrics['effective_area_m2'])

        # Convert to arrays
        for key in ['volume_velocity', 'volume_velocity_rigid', 'velocity_ratio', 'effective_area']:
            results[key] = np.array(results[key])

        return results


@dataclass
class RigidVsElasticComparison:
    """
    Comparison between rigid piston and elastic dome predictions.

    Attributes
    ----------
    frequencies_hz : np.ndarray
        Analysis frequencies.
    spl_rigid_db : np.ndarray
        SPL prediction from rigid piston model.
    spl_elastic_db : np.ndarray
        SPL prediction from elastic dome model.
    spl_difference_db : np.ndarray
        Deviation (elastic - rigid) in dB.
    breakup_frequency_hz : Optional[float]
        First frequency where deviation exceeds threshold.
    threshold_db : float
        Threshold used for breakup detection.
    """
    frequencies_hz: np.ndarray
    spl_rigid_db: np.ndarray
    spl_elastic_db: np.ndarray
    spl_difference_db: np.ndarray
    breakup_frequency_hz: Optional[float]
    threshold_db: float

    @classmethod
    def from_sweep_results(
        cls,
        rigid_results: dict,
        elastic_results: dict,
        threshold_db: float = 3.0,
    ) -> "RigidVsElasticComparison":
        """
        Create comparison from frequency sweep results.

        Parameters
        ----------
        rigid_results : dict
            Results from rigid piston sweep.
        elastic_results : dict
            Results from elastic dome sweep.
        threshold_db : float
            Deviation threshold for breakup detection [dB].
        """
        freqs = rigid_results['frequencies']

        # Compute SPL from volume velocity magnitude
        # SPL ∝ 20 log10(|U|)
        u_rigid = np.abs(rigid_results['volume_velocity'])
        u_elastic = np.abs(elastic_results['volume_velocity'])

        # Reference to first frequency rigid value for relative comparison
        u_ref = u_rigid[0] if u_rigid[0] > 0 else 1.0

        spl_rigid = 20 * np.log10(u_rigid / u_ref + 1e-30)
        spl_elastic = 20 * np.log10(u_elastic / u_ref + 1e-30)
        spl_diff = spl_elastic - spl_rigid

        # Find first breakup frequency
        breakup_idx = np.where(np.abs(spl_diff) > threshold_db)[0]
        breakup_freq = freqs[breakup_idx[0]] if len(breakup_idx) > 0 else None

        return cls(
            frequencies_hz=freqs,
            spl_rigid_db=spl_rigid,
            spl_elastic_db=spl_elastic,
            spl_difference_db=spl_diff,
            breakup_frequency_hz=breakup_freq,
            threshold_db=threshold_db,
        )


def create_validation_workflow(
    network: "CompressionDriverNetwork",
    dome_mesh,
    fem_solver: Optional[Callable] = None,
    frequencies_hz: Optional[np.ndarray] = None,
    *,
    c: float = 343.0,
    rho: float = 1.225,
) -> dict:
    """
    Create a validation workflow comparing rigid vs elastic dome.

    This function sets up the complete workflow for:
    1. Running frequency sweep with rigid piston assumption
    2. Computing FEM velocity distributions (if solver provided)
    3. Running frequency sweep with elastic dome corrections
    4. Comparing results to identify breakup frequency

    Parameters
    ----------
    network : CompressionDriverNetwork
        Lumped-element network for the driver.
    dome_mesh : DomeMesh or bempp_cl.api.Grid
        Dome surface mesh.
    fem_solver : callable, optional
        Function f(frequency_hz) -> velocity_distribution.
        If None, uses approximate first-mode profile.
    frequencies_hz : np.ndarray, optional
        Analysis frequencies. Default: log-spaced 100 Hz to 20 kHz.
    c : float
        Speed of sound [m/s].
    rho : float
        Air density [kg/m³].

    Returns
    -------
    dict
        Validation workflow results:
        - 'rigid_sweep': rigid piston frequency sweep
        - 'elastic_sweep': elastic dome frequency sweep
        - 'comparison': RigidVsElasticComparison object
        - 'profile': ElasticDomeProfile
        - 'breakup_frequency_hz': detected breakup frequency
    """
    if frequencies_hz is None:
        frequencies_hz = np.logspace(np.log10(100), np.log10(20000), 50)

    # Get mesh data
    if hasattr(dome_mesh, 'to_bempp_grid'):
        bempp_grid = dome_mesh.to_bempp_grid()
        element_areas = dome_mesh.element_areas
    else:
        # Assume it's already a bempp grid
        bempp_grid = dome_mesh
        vertices = np.asarray(bempp_grid.vertices, dtype=float)
        elements = np.asarray(bempp_grid.elements, dtype=int)
        p0 = vertices[:, elements[0, :]]
        p1 = vertices[:, elements[1, :]]
        p2 = vertices[:, elements[2, :]]
        element_areas = 0.5 * np.linalg.norm(np.cross((p1 - p0).T, (p2 - p0).T), axis=1)

    n_elements = bempp_grid.number_of_elements
    total_area = element_areas.sum()

    # Compute element radii for mode shape approximation
    centroids = np.zeros((n_elements, 3))
    vertices = np.asarray(bempp_grid.vertices, dtype=float)
    elements = np.asarray(bempp_grid.elements, dtype=int)
    for i in range(n_elements):
        tri_verts = vertices[:, elements[:, i]]
        centroids[i] = tri_verts.mean(axis=1)
    center = centroids.mean(axis=0)
    element_radii = np.linalg.norm(centroids - center, axis=1)
    dome_radius = element_radii.max()

    # Create velocity distributions
    if fem_solver is not None:
        # Use provided FEM solver
        velocity_distributions = {}
        for freq in frequencies_hz:
            velocity_distributions[float(freq)] = fem_solver(freq)
    else:
        # Use approximate first-mode profile
        # Mode shape: J0(2.405 * r / a) for clamped circular plate
        from scipy.special import j0
        mode_shape = j0(2.405 * element_radii / dome_radius)

        # Estimate first resonance
        from bempp_audio.fea.materials import ShellMaterial, estimate_first_bending_mode
        # Default to titanium if no material info available
        f1_estimate = 8000.0  # Rough estimate for typical CD dome

        velocity_distributions = {}
        for freq in frequencies_hz:
            # Apply modal transfer function
            zeta = 0.01  # Damping ratio
            omega = 2 * np.pi * freq
            omega_n = 2 * np.pi * f1_estimate
            h_modal = 1.0 / (omega_n**2 - omega**2 + 2j * zeta * omega_n * omega)

            # Low frequency: piston-like; near resonance: modal response
            piston_weight = 1.0 / (1 + (freq / f1_estimate)**4)
            modal_weight = 1.0 - piston_weight

            v_piston = np.ones(n_elements, dtype=complex)
            v_modal = mode_shape.astype(complex) * np.abs(h_modal)

            v_total = piston_weight * v_piston + modal_weight * v_modal
            # Normalize to unit RMS
            v_rms = np.sqrt(np.sum(np.abs(v_total)**2 * element_areas) / total_area)
            velocity_distributions[float(freq)] = v_total / (v_rms + 1e-30)

    # Create elastic dome profile
    # For voice coil velocity, assume unit velocity (profile captures relative effect)
    voice_coil_velocity = np.ones(len(frequencies_hz), dtype=complex)

    profile = ElasticDomeProfile.from_fem_results(
        frequencies_hz=frequencies_hz,
        velocity_distributions=velocity_distributions,
        element_areas=element_areas,
        voice_coil_velocity=voice_coil_velocity,
        diaphragm_area_m2=total_area,
    )

    # Create adapted network
    adapter = ElasticDomeNetworkAdapter(network, profile)

    # Run sweeps
    rigid_sweep = {
        'frequencies': frequencies_hz,
        'volume_velocity': np.array([
            network.solve_volume_velocity(f) for f in frequencies_hz
        ]),
    }

    elastic_sweep = adapter.frequency_sweep(frequencies_hz)

    # Compare results
    comparison = RigidVsElasticComparison.from_sweep_results(
        rigid_sweep, elastic_sweep, threshold_db=3.0
    )

    return {
        'rigid_sweep': rigid_sweep,
        'elastic_sweep': elastic_sweep,
        'comparison': comparison,
        'profile': profile,
        'adapter': adapter,
        'breakup_frequency_hz': comparison.breakup_frequency_hz,
    }
