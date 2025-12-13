"""
Elastic dome BEM solver for compression driver radiation analysis.

This module provides the `ElasticDomeBEMSolver` class for computing:
- Far-field directivity from FEM velocity distributions
- Radiation impedance of vibrating shell surfaces
- Modal radiation efficiency
- SPL predictions with modal contributions

The solver bridges shell FEM (DOLFINx) with acoustic BEM (bempp-cl) for
vibroacoustic analysis of compression driver domes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import numpy as np

from bempp_audio._optional import optional_import

bempp, BEMPP_AVAILABLE = optional_import("bempp_cl.api")


@dataclass
class ModalVelocityProfile:
    """
    Velocity profile from shell FEM modal analysis.

    Attributes
    ----------
    frequencies_hz : np.ndarray
        Frequencies at which velocity was computed.
    velocity_distributions : dict[float, np.ndarray]
        Normal velocity at each element for each frequency.
        Keys are frequencies in Hz.
    mode_shapes : np.ndarray, optional
        Mode shape matrix (n_dofs x n_modes).
    mode_frequencies : np.ndarray, optional
        Natural frequencies of each mode.
    modal_participation : np.ndarray, optional
        Modal participation factors at each frequency.
    """
    frequencies_hz: np.ndarray
    velocity_distributions: dict[float, np.ndarray]
    mode_shapes: Optional[np.ndarray] = None
    mode_frequencies: Optional[np.ndarray] = None
    modal_participation: Optional[dict[float, np.ndarray]] = None

    @classmethod
    def from_piston(
        cls,
        frequencies_hz: np.ndarray,
        n_elements: int,
        amplitude: complex = 1.0,
    ) -> "ModalVelocityProfile":
        """Create a rigid piston velocity profile for comparison."""
        distributions = {
            f: np.full(n_elements, amplitude, dtype=complex)
            for f in frequencies_hz
        }
        return cls(
            frequencies_hz=np.asarray(frequencies_hz),
            velocity_distributions=distributions,
        )

    @classmethod
    def from_first_mode(
        cls,
        frequencies_hz: np.ndarray,
        element_radii: np.ndarray,
        dome_radius: float,
        amplitude: complex = 1.0,
    ) -> "ModalVelocityProfile":
        """Create a first bending mode velocity profile (J0 Bessel approximation)."""
        from scipy.special import j0

        # First bending mode shape: J0(2.405 * r / a) for clamped circular plate
        mode_shape = j0(2.405 * element_radii / dome_radius)

        distributions = {
            f: amplitude * mode_shape.astype(complex)
            for f in frequencies_hz
        }
        return cls(
            frequencies_hz=np.asarray(frequencies_hz),
            velocity_distributions=distributions,
        )


@dataclass
class ElasticDomeResult:
    """
    Result container for elastic dome radiation analysis.

    Attributes
    ----------
    frequency_hz : float
        Analysis frequency.
    wavenumber : float
        Acoustic wavenumber k = ω/c.
    surface_pressure : np.ndarray
        Complex pressure at each surface element.
    radiation_impedance : complex
        Acoustic radiation impedance [Pa·s/m³].
    radiated_power : float
        Time-averaged radiated power [W].
    velocity_distribution : np.ndarray
        Input velocity distribution.
    rigid_piston_comparison : Optional[dict]
        Comparison metrics against rigid piston model.
    modal_efficiencies : Optional[dict]
        Per-mode radiation efficiencies if modal data available.
    """
    frequency_hz: float
    wavenumber: float
    surface_pressure: np.ndarray
    radiation_impedance: complex
    radiated_power: float
    velocity_distribution: np.ndarray
    rigid_piston_comparison: Optional[dict] = None
    modal_efficiencies: Optional[dict] = None

    def radiation_resistance(self) -> float:
        """Real part of radiation impedance."""
        return float(np.real(self.radiation_impedance))

    def radiation_reactance(self) -> float:
        """Imaginary part of radiation impedance."""
        return float(np.imag(self.radiation_impedance))


@dataclass
class ElasticDomeSolverOptions:
    """Configuration options for the elastic dome solver."""
    tol: float = 1e-5  # GMRES tolerance
    maxiter: int = 1000  # Maximum GMRES iterations
    space_type: str = "DP"  # Function space type
    space_order: int = 0  # Function space order
    coupling_parameter: Optional[float] = None  # Burton-Miller η
    reference_area: Optional[float] = None  # Reference area for normalization
    include_rigid_comparison: bool = True  # Include rigid piston comparison


class ElasticDomeBEMSolver:
    """
    BEM solver for elastic dome radiation.

    This solver computes acoustic radiation from a vibrating dome surface
    with arbitrary (FEM-computed) velocity distributions.

    Parameters
    ----------
    bempp_grid : bempp_cl.api.Grid
        Surface mesh of the dome.
    c : float
        Speed of sound [m/s].
    rho : float
        Air density [kg/m³].
    options : ElasticDomeSolverOptions, optional
        Solver configuration options.

    Examples
    --------
    >>> from bempp_audio.fea import DomeMesher, DomeGeometry
    >>> geometry = DomeGeometry.spherical(base_diameter_m=0.035, dome_height_m=0.008)
    >>> mesh = DomeMesher(geometry).generate(element_size_m=0.001)
    >>> grid = mesh.to_bempp_grid()
    >>> solver = ElasticDomeBEMSolver(grid)
    >>> velocity_profile = ModalVelocityProfile.from_piston(
    ...     frequencies_hz=[1000, 5000, 10000],
    ...     n_elements=grid.number_of_elements,
    ... )
    >>> response = solver.solve_frequencies(velocity_profile)
    """

    def __init__(
        self,
        bempp_grid,
        c: float = 343.0,
        rho: float = 1.225,
        options: Optional[ElasticDomeSolverOptions] = None,
    ):
        if not BEMPP_AVAILABLE:
            raise ImportError("bempp-cl is required for ElasticDomeBEMSolver")

        self.grid = bempp_grid
        self.c = c
        self.rho = rho
        self.options = options or ElasticDomeSolverOptions()

        # Precompute mesh geometry
        self._element_areas = self._compute_element_areas()
        self._element_centroids = self._compute_element_centroids()
        self._total_area = self._element_areas.sum()

        # Reference area for normalization (defaults to total area)
        self._reference_area = (
            self.options.reference_area
            if self.options.reference_area is not None
            else self._total_area
        )

        # Operator cache
        self._operator_cache: dict = {}

    def _compute_element_areas(self) -> np.ndarray:
        """Compute area of each surface element."""
        vertices = np.asarray(self.grid.vertices, dtype=float)
        elements = np.asarray(self.grid.elements, dtype=int)
        p0 = vertices[:, elements[0, :]]
        p1 = vertices[:, elements[1, :]]
        p2 = vertices[:, elements[2, :]]
        return 0.5 * np.linalg.norm(np.cross((p1 - p0).T, (p2 - p0).T), axis=1)

    def _compute_element_centroids(self) -> np.ndarray:
        """Compute centroid of each surface element."""
        vertices = np.asarray(self.grid.vertices, dtype=float)
        elements = np.asarray(self.grid.elements, dtype=int)
        p0 = vertices[:, elements[0, :]]
        p1 = vertices[:, elements[1, :]]
        p2 = vertices[:, elements[2, :]]
        return ((p0 + p1 + p2) / 3).T

    @property
    def element_areas(self) -> np.ndarray:
        """Element areas [m²]."""
        return self._element_areas

    @property
    def element_centroids(self) -> np.ndarray:
        """Element centroids [m]."""
        return self._element_centroids

    @property
    def total_area(self) -> float:
        """Total surface area [m²]."""
        return self._total_area

    @property
    def characteristic_radius(self) -> float:
        """Effective radius based on total area."""
        return np.sqrt(self._total_area / np.pi)

    def _get_operators(self, k: complex):
        """Get or create BEM operators for given wavenumber."""
        cache_key = complex(k)
        if cache_key in self._operator_cache:
            return self._operator_cache[cache_key]

        # Create function space
        space = bempp.function_space(
            self.grid,
            self.options.space_type,
            self.options.space_order,
        )

        # Create operators
        identity = bempp.operators.boundary.sparse.identity(space, space, space)
        slp = bempp.operators.boundary.helmholtz.single_layer(space, space, space, k)
        adlp = bempp.operators.boundary.helmholtz.adjoint_double_layer(space, space, space, k)

        # Burton-Miller coupling parameter
        eta = self.options.coupling_parameter
        if eta is None:
            eta = k

        # Combined operator for exterior Neumann problem
        lhs = 0.5 * identity + adlp - 1j * eta * slp

        self._operator_cache[cache_key] = {
            "lhs": lhs,
            "space": space,
            "eta": eta,
            "identity": identity,
            "slp": slp,
            "adlp": adlp,
        }

        return self._operator_cache[cache_key]

    def solve(
        self,
        frequency_hz: float,
        velocity_distribution: np.ndarray,
    ) -> ElasticDomeResult:
        """
        Solve for surface pressure and radiation metrics at a single frequency.

        Parameters
        ----------
        frequency_hz : float
            Analysis frequency [Hz].
        velocity_distribution : np.ndarray
            Complex normal velocity at each element.

        Returns
        -------
        ElasticDomeResult
            Solution container with pressure, impedance, and power.
        """
        from scipy.sparse.linalg import gmres as scipy_gmres

        omega = 2 * np.pi * frequency_hz
        k = omega / self.c

        # Get operators
        bundle = self._get_operators(k)
        lhs = bundle["lhs"]
        space = bundle["space"]

        # Create RHS from Neumann data: g = ∂p/∂n = -iωρ v_n
        velocity_distribution = np.asarray(velocity_distribution, dtype=np.complex128)
        rhs_coeffs = -1j * omega * self.rho * velocity_distribution

        # Solve using scipy GMRES with strong_form
        lhs_discrete = lhs.strong_form()
        solution, info = scipy_gmres(
            lhs_discrete,
            rhs_coeffs,
            tol=self.options.tol,
            maxiter=self.options.maxiter,
        )

        if info != 0:
            import warnings
            warnings.warn(f"GMRES did not converge at {frequency_hz} Hz (info={info})")

        # Compute radiation metrics
        p_coeffs = solution
        v_coeffs = velocity_distribution
        areas = self._element_areas

        # Radiation impedance: Z = ∫ p v_n^* dS / ∫ |v_n|² dS
        numerator = np.sum(p_coeffs * np.conj(v_coeffs) * areas)
        denominator = np.sum(np.abs(v_coeffs)**2 * areas)
        z_rad = numerator / (denominator + 1e-30)

        # Radiated power: P = (1/2) Re(∫ p v_n^* dS) (factor of 1/2 for time-average)
        power = 0.5 * np.real(np.sum(p_coeffs * np.conj(v_coeffs) * areas))

        # Compare with rigid piston if requested
        rigid_comparison = None
        if self.options.include_rigid_comparison:
            rigid_comparison = self._rigid_piston_comparison(
                frequency_hz, velocity_distribution
            )

        return ElasticDomeResult(
            frequency_hz=frequency_hz,
            wavenumber=k,
            surface_pressure=p_coeffs,
            radiation_impedance=z_rad,
            radiated_power=power,
            velocity_distribution=velocity_distribution,
            rigid_piston_comparison=rigid_comparison,
        )

    def solve_frequencies(
        self,
        velocity_profile: ModalVelocityProfile,
        show_progress: bool = False,
    ) -> "ElasticDomeFrequencyResponse":
        """
        Solve for multiple frequencies.

        Parameters
        ----------
        velocity_profile : ModalVelocityProfile
            Velocity distributions at each frequency.
        show_progress : bool
            Display progress bar if True.

        Returns
        -------
        ElasticDomeFrequencyResponse
            Container with results at all frequencies.
        """
        results = []
        frequencies = velocity_profile.frequencies_hz

        iterator = frequencies
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(frequencies, desc="Frequency sweep", unit="freq")
            except ImportError:
                pass

        for freq in iterator:
            velocity = velocity_profile.velocity_distributions[float(freq)]
            result = self.solve(freq, velocity)
            results.append(result)

        return ElasticDomeFrequencyResponse(
            results=results,
            velocity_profile=velocity_profile,
        )

    def _rigid_piston_comparison(
        self,
        frequency_hz: float,
        velocity_distribution: np.ndarray,
    ) -> dict:
        """
        Compare elastic dome radiation with equivalent rigid piston.

        The rigid piston has the same total volume velocity.
        """
        # Compute effective (average) velocity weighted by area
        areas = self._element_areas
        total_area = self._total_area

        # Volume velocity from elastic dome
        v_elastic = velocity_distribution
        u_elastic = np.sum(v_elastic * areas)  # m³/s

        # Equivalent piston velocity (same volume velocity)
        v_piston = u_elastic / total_area

        # Solve for piston radiation
        v_piston_dist = np.full_like(v_elastic, v_piston)
        piston_result = self._solve_no_comparison(frequency_hz, v_piston_dist)

        # Compute comparison metrics (use _solve_no_comparison to avoid recursion)
        elastic_result = self._solve_no_comparison(frequency_hz, velocity_distribution)
        z_elastic = elastic_result.radiation_impedance
        z_piston = piston_result.radiation_impedance

        # Radiation efficiency ratio
        # σ = P_rad / (ρc * <|v|²> * S) normalized
        sigma_elastic = self._compute_radiation_efficiency(
            frequency_hz, velocity_distribution
        )
        sigma_piston = self._compute_radiation_efficiency(
            frequency_hz, v_piston_dist
        )

        return {
            "z_elastic": z_elastic,
            "z_piston": z_piston,
            "z_ratio": z_elastic / (z_piston + 1e-30),
            "efficiency_elastic": sigma_elastic,
            "efficiency_piston": sigma_piston,
            "efficiency_ratio": sigma_elastic / (sigma_piston + 1e-30),
            "volume_velocity": u_elastic,
        }

    def _solve_no_comparison(
        self,
        frequency_hz: float,
        velocity_distribution: np.ndarray,
    ) -> ElasticDomeResult:
        """Solve without triggering recursive comparison."""
        from scipy.sparse.linalg import gmres as scipy_gmres

        omega = 2 * np.pi * frequency_hz
        k = omega / self.c

        bundle = self._get_operators(k)
        lhs = bundle["lhs"]

        velocity_distribution = np.asarray(velocity_distribution, dtype=np.complex128)
        rhs_coeffs = -1j * omega * self.rho * velocity_distribution

        lhs_discrete = lhs.strong_form()
        solution, info = scipy_gmres(
            lhs_discrete, rhs_coeffs,
            tol=self.options.tol,
            maxiter=self.options.maxiter,
        )

        p_coeffs = solution
        v_coeffs = velocity_distribution
        areas = self._element_areas

        numerator = np.sum(p_coeffs * np.conj(v_coeffs) * areas)
        denominator = np.sum(np.abs(v_coeffs)**2 * areas)
        z_rad = numerator / (denominator + 1e-30)

        power = 0.5 * np.real(np.sum(p_coeffs * np.conj(v_coeffs) * areas))

        return ElasticDomeResult(
            frequency_hz=frequency_hz,
            wavenumber=k,
            surface_pressure=p_coeffs,
            radiation_impedance=z_rad,
            radiated_power=power,
            velocity_distribution=velocity_distribution,
        )

    def _compute_radiation_efficiency(
        self,
        frequency_hz: float,
        velocity_distribution: np.ndarray,
    ) -> float:
        """
        Compute radiation efficiency.

        σ = P_rad / (ρc * S * <v²>)

        where <v²> is the mean-square velocity.
        """
        result = self._solve_no_comparison(frequency_hz, velocity_distribution)

        v_coeffs = velocity_distribution
        areas = self._element_areas

        # Mean square velocity
        mean_sq_v = np.sum(np.abs(v_coeffs)**2 * areas) / self._total_area

        # Reference power for piston with same RMS velocity
        p_ref = self.rho * self.c * self._total_area * mean_sq_v

        # Radiation efficiency
        return result.radiated_power / (p_ref + 1e-30)

    def compute_directivity(
        self,
        frequency_hz: float,
        velocity_distribution: np.ndarray,
        theta_deg: np.ndarray = None,
        phi_deg: float = 0.0,
        r_field: float = 1.0,
    ) -> dict:
        """
        Compute far-field directivity pattern.

        Parameters
        ----------
        frequency_hz : float
            Analysis frequency [Hz].
        velocity_distribution : np.ndarray
            Normal velocity at each element.
        theta_deg : np.ndarray, optional
            Polar angles [degrees]. Default: 0 to 180 in 5° steps.
        phi_deg : float
            Azimuthal angle [degrees]. Default: 0.
        r_field : float
            Field point radius [m]. Default: 1.0.

        Returns
        -------
        dict
            Directivity data with keys:
            - 'theta_deg': polar angles
            - 'phi_deg': azimuthal angle
            - 'pressure': complex pressure at each angle
            - 'spl_db': SPL relative to on-axis
            - 'di_db': directivity index
        """
        if theta_deg is None:
            theta_deg = np.arange(0, 181, 5.0)

        # Solve for surface quantities
        result = self.solve(frequency_hz, velocity_distribution)
        p_surface = result.surface_pressure

        omega = 2 * np.pi * frequency_hz
        k = omega / self.c

        # Create field points in spherical coordinates
        theta_rad = np.deg2rad(theta_deg)
        phi_rad = np.deg2rad(phi_deg)

        n_points = len(theta_deg)
        field_points = np.zeros((3, n_points))
        field_points[0, :] = r_field * np.sin(theta_rad) * np.cos(phi_rad)
        field_points[1, :] = r_field * np.sin(theta_rad) * np.sin(phi_rad)
        field_points[2, :] = r_field * np.cos(theta_rad)

        # Use potential operators for field evaluation
        bundle = self._get_operators(k)
        space = bundle["space"]

        slp_pot = bempp.operators.potential.helmholtz.single_layer(
            space, field_points, k
        )
        dlp_pot = bempp.operators.potential.helmholtz.double_layer(
            space, field_points, k
        )

        # Neumann data for potential evaluation
        v_coeffs = velocity_distribution
        neumann_coeffs = -1j * omega * self.rho * v_coeffs
        neumann_gf = bempp.GridFunction(space, coefficients=neumann_coeffs)
        pressure_gf = bempp.GridFunction(space, coefficients=p_surface)

        # Evaluate field: p = DLP(p_surf) - SLP(∂p/∂n)
        # For exterior problem with Neumann BC
        field_pressure = (dlp_pot @ pressure_gf - slp_pot @ neumann_gf).ravel()

        # Compute SPL relative to on-axis (theta=0)
        p_on_axis = field_pressure[0] if len(field_pressure) > 0 else 1.0
        spl_db = 20 * np.log10(np.abs(field_pressure) / (np.abs(p_on_axis) + 1e-30) + 1e-30)

        # Directivity index
        # DI = 10 log10(4π * I_axis / P_total)
        p_abs_sq = np.abs(field_pressure)**2
        # Approximate integral over sphere using available angles
        if len(theta_rad) > 1:
            # Weight by solid angle (sin θ dθ)
            sin_theta = np.sin(theta_rad)
            sin_theta[sin_theta < 1e-10] = 1e-10
            p_avg = np.trapz(p_abs_sq * sin_theta, theta_rad) / 2  # hemisphere
        else:
            p_avg = p_abs_sq[0]

        di_db = 10 * np.log10(p_abs_sq[0] / (p_avg + 1e-30) + 1e-30)

        return {
            "theta_deg": theta_deg,
            "phi_deg": phi_deg,
            "pressure": field_pressure,
            "spl_db": spl_db,
            "di_db": float(di_db),
            "r_field": r_field,
        }

    def clear_cache(self):
        """Clear the operator cache."""
        self._operator_cache.clear()


@dataclass
class ElasticDomeFrequencyResponse:
    """
    Container for elastic dome frequency response data.

    Attributes
    ----------
    results : list[ElasticDomeResult]
        Results at each frequency.
    velocity_profile : ModalVelocityProfile
        Input velocity profiles.
    """
    results: list[ElasticDomeResult]
    velocity_profile: ModalVelocityProfile

    @property
    def frequencies(self) -> np.ndarray:
        """Analysis frequencies [Hz]."""
        return np.array([r.frequency_hz for r in self.results])

    @property
    def radiation_impedance(self) -> np.ndarray:
        """Radiation impedance at each frequency."""
        return np.array([r.radiation_impedance for r in self.results])

    @property
    def radiated_power(self) -> np.ndarray:
        """Radiated power at each frequency [W]."""
        return np.array([r.radiated_power for r in self.results])

    def radiation_resistance(self) -> np.ndarray:
        """Real part of radiation impedance."""
        return np.real(self.radiation_impedance)

    def radiation_reactance(self) -> np.ndarray:
        """Imaginary part of radiation impedance."""
        return np.imag(self.radiation_impedance)

    def spl_db(self, reference_power: float = 1e-12) -> np.ndarray:
        """
        Compute SPL from radiated power.

        Parameters
        ----------
        reference_power : float
            Reference power [W]. Default: 1e-12 W.

        Returns
        -------
        np.ndarray
            SPL in dB.
        """
        return 10 * np.log10(self.radiated_power / reference_power + 1e-30)

    def rigid_piston_deviation(self) -> Optional[np.ndarray]:
        """
        Compute deviation from rigid piston model in dB.

        Returns
        -------
        np.ndarray or None
            SPL deviation (elastic - piston) in dB.
        """
        if not all(r.rigid_piston_comparison for r in self.results):
            return None

        deviations = []
        for r in self.results:
            comp = r.rigid_piston_comparison
            # Compare power or impedance magnitude
            z_ratio = np.abs(comp["z_elastic"]) / (np.abs(comp["z_piston"]) + 1e-30)
            deviation_db = 20 * np.log10(z_ratio + 1e-30)
            deviations.append(deviation_db)

        return np.array(deviations)


def compute_modal_radiation_efficiency(
    bempp_grid,
    mode_shapes: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    c: float = 343.0,
    rho: float = 1.225,
) -> dict:
    """
    Compute radiation efficiency for each mode at each frequency.

    The radiation efficiency σ_n(f) indicates how effectively mode n
    couples to the acoustic field at frequency f.

    Parameters
    ----------
    bempp_grid : bempp_cl.api.Grid
        Surface mesh.
    mode_shapes : np.ndarray
        Mode shape matrix (n_elements x n_modes).
    frequencies_hz : np.ndarray
        Analysis frequencies [Hz].
    c : float
        Speed of sound [m/s].
    rho : float
        Air density [kg/m³].

    Returns
    -------
    dict
        Dictionary with:
        - 'frequencies': analysis frequencies
        - 'efficiencies': (n_freqs x n_modes) array of radiation efficiencies
        - 'effective_modes': indices of modes with σ > 0.1 at any frequency
    """
    if not BEMPP_AVAILABLE:
        raise ImportError("bempp-cl is required")

    solver = ElasticDomeBEMSolver(
        bempp_grid, c=c, rho=rho,
        options=ElasticDomeSolverOptions(include_rigid_comparison=False),
    )

    n_modes = mode_shapes.shape[1]
    n_freqs = len(frequencies_hz)

    efficiencies = np.zeros((n_freqs, n_modes))

    for i, freq in enumerate(frequencies_hz):
        for j in range(n_modes):
            mode_velocity = mode_shapes[:, j].astype(complex)
            sigma = solver._compute_radiation_efficiency(freq, mode_velocity)
            efficiencies[i, j] = sigma

    # Find modes that couple effectively (σ > 0.1) at any frequency
    effective_modes = np.where(np.max(efficiencies, axis=0) > 0.1)[0]

    return {
        "frequencies": frequencies_hz,
        "efficiencies": efficiencies,
        "effective_modes": effective_modes,
    }


def compute_modal_spl(
    bempp_grid,
    mode_shapes: np.ndarray,
    mode_frequencies_hz: np.ndarray,
    excitation_frequencies_hz: np.ndarray,
    modal_forces: np.ndarray,
    modal_damping: float = 0.01,
    *,
    c: float = 343.0,
    rho: float = 1.225,
) -> dict:
    """
    Compute SPL prediction with modal contributions.

    This sums contributions from each mode, weighted by:
    - Modal force (excitation amplitude)
    - Modal transfer function (resonance behavior)
    - Radiation efficiency

    Parameters
    ----------
    bempp_grid : bempp_cl.api.Grid
        Surface mesh.
    mode_shapes : np.ndarray
        Mode shape matrix (n_elements x n_modes).
    mode_frequencies_hz : np.ndarray
        Natural frequency of each mode [Hz].
    excitation_frequencies_hz : np.ndarray
        Frequencies at which to compute SPL [Hz].
    modal_forces : np.ndarray
        Modal force amplitude for each mode [N].
    modal_damping : float
        Modal damping ratio ζ (same for all modes).
    c : float
        Speed of sound [m/s].
    rho : float
        Air density [kg/m³].

    Returns
    -------
    dict
        Dictionary with:
        - 'frequencies': excitation frequencies
        - 'spl_total': total SPL [dB re 1e-12 W]
        - 'spl_per_mode': (n_freqs x n_modes) contribution from each mode
        - 'dominant_mode': index of dominant mode at each frequency
    """
    if not BEMPP_AVAILABLE:
        raise ImportError("bempp-cl is required")

    solver = ElasticDomeBEMSolver(
        bempp_grid, c=c, rho=rho,
        options=ElasticDomeSolverOptions(include_rigid_comparison=False),
    )

    n_modes = mode_shapes.shape[1]
    n_freqs = len(excitation_frequencies_hz)

    power_per_mode = np.zeros((n_freqs, n_modes), dtype=complex)

    for i, f_exc in enumerate(excitation_frequencies_hz):
        omega = 2 * np.pi * f_exc

        for j in range(n_modes):
            omega_n = 2 * np.pi * mode_frequencies_hz[j]
            zeta = modal_damping

            # Modal transfer function H(ω) = 1 / (ω_n² - ω² + 2jζω_n*ω)
            h_modal = 1.0 / (omega_n**2 - omega**2 + 2j * zeta * omega_n * omega)

            # Modal velocity = H(ω) * F_n * jω (displacement to velocity)
            q_modal = h_modal * modal_forces[j] * 1j * omega

            # Total velocity from this mode
            mode_velocity = q_modal * mode_shapes[:, j].astype(complex)

            # Radiated power from this mode
            result = solver._solve_no_comparison(f_exc, mode_velocity)
            power_per_mode[i, j] = result.radiated_power

    # Total power (sum of modal contributions - may need phase consideration)
    # For uncorrelated modes, powers add
    power_total = np.sum(np.abs(power_per_mode), axis=1)

    # Convert to SPL
    p_ref = 1e-12
    spl_total = 10 * np.log10(power_total / p_ref + 1e-30)
    spl_per_mode = 10 * np.log10(np.abs(power_per_mode) / p_ref + 1e-30)

    # Find dominant mode at each frequency
    dominant_mode = np.argmax(np.abs(power_per_mode), axis=1)

    return {
        "frequencies": excitation_frequencies_hz,
        "spl_total": spl_total,
        "spl_per_mode": spl_per_mode,
        "power_total": power_total,
        "power_per_mode": np.abs(power_per_mode),
        "dominant_mode": dominant_mode,
    }
