"""
Phase plug FEM to exterior BEM coupling and acoustic metrics.

This module provides:
1. Throat exit coupling to exterior BEM
   - Extract throat pressure/velocity from interior FEM solution
   - Create equivalent BEM source for exterior radiation
   - Simpler option: radiation impedance BC at throat

2. Acoustic metrics extraction
   - Acoustic power flow through phase plug
   - Impedance transformation ratio
   - Pressure distribution visualization helpers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Tuple, List, Union, TYPE_CHECKING
import numpy as np

try:
    import bempp_cl.api
    BEMPP_AVAILABLE = True
except ImportError:
    BEMPP_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover
    from bempp_audio.fea.helmholtz_fem import HelmholtzResult, HelmholtzFEMSolver


# Default air properties
AIR_DENSITY = 1.225  # kg/m³
SPEED_OF_SOUND = 343.0  # m/s


@dataclass
class ThroatExitData:
    """
    Acoustic data extracted from the throat exit surface.

    Contains pressure and velocity distributions for coupling to
    exterior BEM or for computing acoustic metrics.

    Attributes
    ----------
    pressure : np.ndarray
        Complex pressure at throat boundary DOFs [Pa]
    velocity : np.ndarray
        Complex normal velocity at throat boundary DOFs [m/s]
    coordinates : np.ndarray
        (N, 3) coordinates of throat boundary points [m]
    areas : np.ndarray
        Area associated with each point (for integration) [m²]
    frequency_hz : float
        Frequency at which data was extracted
    throat_radius_m : float
        Throat radius [m]
    """
    pressure: np.ndarray
    velocity: np.ndarray
    coordinates: np.ndarray
    areas: np.ndarray
    frequency_hz: float
    throat_radius_m: float

    @property
    def total_area_m2(self) -> float:
        """Total throat exit area [m²]."""
        return float(np.sum(self.areas))

    @property
    def mean_pressure(self) -> complex:
        """Area-weighted mean pressure [Pa]."""
        return np.sum(self.pressure * self.areas) / self.total_area_m2

    @property
    def mean_velocity(self) -> complex:
        """Area-weighted mean normal velocity [m/s]."""
        return np.sum(self.velocity * self.areas) / self.total_area_m2

    @property
    def volume_velocity(self) -> complex:
        """Volume velocity (flux) through throat [m³/s]."""
        return np.sum(self.velocity * self.areas)

    @property
    def acoustic_power_w(self) -> float:
        """Time-averaged acoustic power through throat [W]."""
        # P = (1/2) Re{ ∫ p v_n^* dS }
        power_density = self.pressure * np.conj(self.velocity)
        return 0.5 * np.real(np.sum(power_density * self.areas))

    @property
    def acoustic_impedance(self) -> complex:
        """Acoustic impedance at throat: Z = <p> / U [Pa·s/m³]."""
        U = self.volume_velocity
        if np.abs(U) < 1e-16:
            return complex('inf')
        return self.mean_pressure / U

    @property
    def specific_impedance(self) -> complex:
        """Specific acoustic impedance: z = <p> / <v> [Pa·s/m]."""
        v_mean = self.mean_velocity
        if np.abs(v_mean) < 1e-16:
            return complex('inf')
        return self.mean_pressure / v_mean


@dataclass
class PhasePlugMetrics:
    """
    Acoustic metrics for a phase plug at a single frequency.

    Attributes
    ----------
    frequency_hz : float
        Analysis frequency
    dome_power_w : float
        Acoustic power entering at dome interface
    throat_power_w : float
        Acoustic power exiting at throat
    transmission_efficiency : float
        Power transmission efficiency (throat_power / dome_power)
    dome_impedance : complex
        Acoustic impedance at dome interface
    throat_impedance : complex
        Acoustic impedance at throat exit
    impedance_transformation : complex
        Impedance transformation ratio (dome / throat)
    pressure_uniformity : float
        Throat pressure uniformity (1 = perfectly uniform, 0 = highly non-uniform)
    phase_spread_deg : float
        Maximum phase variation across throat [degrees]
    """
    frequency_hz: float
    dome_power_w: float
    throat_power_w: float
    transmission_efficiency: float
    dome_impedance: complex
    throat_impedance: complex
    impedance_transformation: complex
    pressure_uniformity: float
    phase_spread_deg: float


def extract_throat_data(
    result: "HelmholtzResult",
    facet_tags,
    throat_marker: int,
    *,
    medium_rho: float = AIR_DENSITY,
    throat_radius_m: Optional[float] = None,
) -> ThroatExitData:
    """
    Extract pressure and velocity data from the throat exit surface.

    Parameters
    ----------
    result : HelmholtzResult
        Solution from HelmholtzFEMSolver
    facet_tags : dolfinx.mesh.MeshTags
        Boundary markers
    throat_marker : int
        Marker ID for the throat exit surface
    medium_rho : float
        Air density [kg/m³]
    throat_radius_m : float, optional
        Throat radius. If None, estimated from mesh.

    Returns
    -------
    ThroatExitData
        Extracted acoustic data at the throat
    """
    # Import DOLFINx components
    try:
        from dolfinx import fem
        import ufl
    except ImportError:
        raise ImportError("DOLFINx is required for throat data extraction")

    mesh = result.mesh
    omega = 2 * np.pi * result.frequency_hz

    # Get throat facets
    throat_facets = facet_tags.find(throat_marker)

    if len(throat_facets) == 0:
        raise ValueError(f"No facets found with marker {throat_marker}")

    # Create connectivity
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # Get coordinates and areas of throat facets
    from dolfinx.mesh import locate_entities_boundary

    # Extract facet vertices and compute areas
    facet_to_vertex = mesh.topology.connectivity(mesh.topology.dim - 1, 0)
    coords = mesh.geometry.x

    facet_areas = []
    facet_centroids = []

    for facet_idx in throat_facets:
        vertex_indices = facet_to_vertex.links(facet_idx)
        vertices = coords[vertex_indices]

        # For triangular facets: area = 0.5 * ||(v1-v0) x (v2-v0)||
        if len(vertices) == 3:
            v0, v1, v2 = vertices
            cross = np.cross(v1 - v0, v2 - v0)
            area = 0.5 * np.linalg.norm(cross)
            centroid = (v0 + v1 + v2) / 3
        else:
            # For other shapes, approximate
            centroid = vertices.mean(axis=0)
            area = 0.0  # Need proper calculation for quads

        facet_areas.append(area)
        facet_centroids.append(centroid)

    facet_areas = np.array(facet_areas)
    facet_centroids = np.array(facet_centroids)

    # Evaluate pressure at facet centroids
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

    tree = bb_tree(mesh, mesh.topology.dim)
    cell_candidates = compute_collisions_points(tree, facet_centroids)
    cells = compute_colliding_cells(mesh, cell_candidates, facet_centroids)

    pressure_values = result.pressure.eval(facet_centroids, cells.array).flatten()

    # Compute normal velocity from pressure gradient
    # ∂p/∂n = -iωρ v_n  →  v_n = (i/ωρ) ∂p/∂n
    # For the throat exit (z = constant), normal is in +z direction
    # v_n ≈ p / (ρc) for radiation condition (plane wave approximation)
    c = SPEED_OF_SOUND
    velocity_values = pressure_values / (medium_rho * c)

    # Estimate throat radius if not provided
    if throat_radius_m is None:
        r_values = np.sqrt(facet_centroids[:, 0]**2 + facet_centroids[:, 1]**2)
        throat_radius_m = r_values.max()

    return ThroatExitData(
        pressure=pressure_values,
        velocity=velocity_values,
        coordinates=facet_centroids,
        areas=facet_areas,
        frequency_hz=result.frequency_hz,
        throat_radius_m=throat_radius_m,
    )


def extract_dome_data(
    result: "HelmholtzResult",
    facet_tags,
    dome_marker: int,
    dome_velocity: Union[complex, np.ndarray],
    *,
    medium_rho: float = AIR_DENSITY,
) -> ThroatExitData:
    """
    Extract pressure data from the dome interface surface.

    Similar to extract_throat_data but uses provided velocity BC.

    Parameters
    ----------
    result : HelmholtzResult
        Solution from HelmholtzFEMSolver
    facet_tags : dolfinx.mesh.MeshTags
        Boundary markers
    dome_marker : int
        Marker ID for the dome interface surface
    dome_velocity : complex or np.ndarray
        Prescribed dome normal velocity (from BC)
    medium_rho : float
        Air density [kg/m³]

    Returns
    -------
    ThroatExitData
        Extracted acoustic data at the dome interface
        (Uses ThroatExitData structure for convenience)
    """
    try:
        from dolfinx import fem
    except ImportError:
        raise ImportError("DOLFINx is required for dome data extraction")

    mesh = result.mesh

    # Get dome facets
    dome_facets = facet_tags.find(dome_marker)

    if len(dome_facets) == 0:
        raise ValueError(f"No facets found with marker {dome_marker}")

    # Get coordinates and areas of dome facets
    facet_to_vertex = mesh.topology.connectivity(mesh.topology.dim - 1, 0)
    coords = mesh.geometry.x

    facet_areas = []
    facet_centroids = []

    for facet_idx in dome_facets:
        vertex_indices = facet_to_vertex.links(facet_idx)
        vertices = coords[vertex_indices]

        if len(vertices) == 3:
            v0, v1, v2 = vertices
            cross = np.cross(v1 - v0, v2 - v0)
            area = 0.5 * np.linalg.norm(cross)
            centroid = (v0 + v1 + v2) / 3
        else:
            centroid = vertices.mean(axis=0)
            area = 0.0

        facet_areas.append(area)
        facet_centroids.append(centroid)

    facet_areas = np.array(facet_areas)
    facet_centroids = np.array(facet_centroids)

    # Evaluate pressure at facet centroids
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

    tree = bb_tree(mesh, mesh.topology.dim)
    cell_candidates = compute_collisions_points(tree, facet_centroids)
    cells = compute_colliding_cells(mesh, cell_candidates, facet_centroids)

    pressure_values = result.pressure.eval(facet_centroids, cells.array).flatten()

    # Use provided velocity
    if isinstance(dome_velocity, (int, float, complex)):
        velocity_values = np.full(len(facet_centroids), complex(dome_velocity))
    else:
        velocity_values = np.asarray(dome_velocity)

    # Estimate dome radius
    r_values = np.sqrt(facet_centroids[:, 0]**2 + facet_centroids[:, 1]**2)
    dome_radius_m = r_values.max()

    return ThroatExitData(
        pressure=pressure_values,
        velocity=velocity_values,
        coordinates=facet_centroids,
        areas=facet_areas,
        frequency_hz=result.frequency_hz,
        throat_radius_m=dome_radius_m,  # Actually dome radius
    )


def compute_phase_plug_metrics(
    result: "HelmholtzResult",
    facet_tags,
    dome_marker: int,
    throat_marker: int,
    dome_velocity: Union[complex, np.ndarray],
    *,
    medium_rho: float = AIR_DENSITY,
) -> PhasePlugMetrics:
    """
    Compute comprehensive acoustic metrics for the phase plug.

    Parameters
    ----------
    result : HelmholtzResult
        Solution from HelmholtzFEMSolver
    facet_tags : dolfinx.mesh.MeshTags
        Boundary markers
    dome_marker : int
        Marker ID for dome interface
    throat_marker : int
        Marker ID for throat exit
    dome_velocity : complex or np.ndarray
        Prescribed dome normal velocity
    medium_rho : float
        Air density [kg/m³]

    Returns
    -------
    PhasePlugMetrics
        Computed acoustic metrics
    """
    # Extract data from both surfaces
    dome_data = extract_dome_data(
        result, facet_tags, dome_marker, dome_velocity,
        medium_rho=medium_rho,
    )
    throat_data = extract_throat_data(
        result, facet_tags, throat_marker,
        medium_rho=medium_rho,
    )

    # Power flow
    dome_power = dome_data.acoustic_power_w
    throat_power = throat_data.acoustic_power_w

    # Transmission efficiency
    if abs(dome_power) > 1e-16:
        transmission_eff = throat_power / dome_power
    else:
        transmission_eff = 0.0

    # Impedance transformation
    dome_z = dome_data.acoustic_impedance
    throat_z = throat_data.acoustic_impedance

    if np.abs(throat_z) > 1e-16 and not np.isinf(throat_z):
        z_transform = dome_z / throat_z
    else:
        z_transform = complex('inf')

    # Pressure uniformity at throat
    # Use coefficient of variation: 1 - std/mean
    p_mag = np.abs(throat_data.pressure)
    if np.mean(p_mag) > 1e-16:
        uniformity = 1.0 - np.std(p_mag) / np.mean(p_mag)
        uniformity = max(0.0, uniformity)
    else:
        uniformity = 0.0

    # Phase spread at throat
    phases = np.angle(throat_data.pressure)
    # Unwrap phases relative to mean
    mean_phase = np.angle(throat_data.mean_pressure)
    phase_diffs = phases - mean_phase
    phase_diffs = np.arctan2(np.sin(phase_diffs), np.cos(phase_diffs))  # Wrap to [-π, π]
    phase_spread = np.max(phase_diffs) - np.min(phase_diffs)
    phase_spread_deg = np.degrees(phase_spread)

    return PhasePlugMetrics(
        frequency_hz=result.frequency_hz,
        dome_power_w=dome_power,
        throat_power_w=throat_power,
        transmission_efficiency=transmission_eff,
        dome_impedance=dome_z,
        throat_impedance=throat_z,
        impedance_transformation=z_transform,
        pressure_uniformity=uniformity,
        phase_spread_deg=phase_spread_deg,
    )


def compute_metrics_sweep(
    solver: "HelmholtzFEMSolver",
    facet_tags,
    dome_marker: int,
    throat_marker: int,
    dome_velocity: Union[complex, Callable[[float], complex]],
    frequencies_hz: np.ndarray,
    *,
    medium_rho: float = AIR_DENSITY,
) -> List[PhasePlugMetrics]:
    """
    Compute phase plug metrics over a frequency sweep.

    Parameters
    ----------
    solver : HelmholtzFEMSolver
        Configured FEM solver
    facet_tags : dolfinx.mesh.MeshTags
        Boundary markers
    dome_marker : int
        Marker ID for dome interface
    throat_marker : int
        Marker ID for throat exit
    dome_velocity : complex or Callable
        Dome velocity. If callable, f(frequency_hz) -> velocity
    frequencies_hz : np.ndarray
        Frequencies to analyze
    medium_rho : float
        Air density

    Returns
    -------
    List[PhasePlugMetrics]
        Metrics at each frequency
    """
    metrics_list = []

    for freq in frequencies_hz:
        # Get dome velocity for this frequency
        if callable(dome_velocity):
            v_dome = dome_velocity(freq)
        else:
            v_dome = dome_velocity

        # Solve at this frequency
        result = solver.solve(freq)

        # Compute metrics
        metrics = compute_phase_plug_metrics(
            result, facet_tags, dome_marker, throat_marker, v_dome,
            medium_rho=medium_rho,
        )
        metrics_list.append(metrics)

    return metrics_list


def throat_to_bem_monopole(
    throat_data: ThroatExitData,
    *,
    rho: float = AIR_DENSITY,
    c: float = SPEED_OF_SOUND,
) -> Tuple[np.ndarray, complex]:
    """
    Create an equivalent monopole source from throat exit data.

    For coupling to exterior BEM, the throat can be represented as
    a compact monopole source when ka_throat << 1.

    Parameters
    ----------
    throat_data : ThroatExitData
        Extracted throat exit data
    rho : float
        Air density
    c : float
        Speed of sound

    Returns
    -------
    position : np.ndarray
        Monopole position (throat center) [m]
    strength : complex
        Monopole strength Q = ∫ v_n dS [m³/s]

    Notes
    -----
    The monopole pressure field is:
        p(r) = (iωρ Q / 4π) * exp(-ikr) / r

    where Q is the volume velocity (strength).
    """
    # Position at throat center
    position = throat_data.coordinates.mean(axis=0)

    # Monopole strength = volume velocity
    strength = throat_data.volume_velocity

    return position, strength


def throat_to_bem_piston(
    throat_data: ThroatExitData,
    bempp_grid=None,
    *,
    rho: float = AIR_DENSITY,
) -> dict:
    """
    Create an equivalent BEM piston source from throat exit data.

    For more accurate coupling, represent the throat as a vibrating
    piston with the velocity distribution from FEM.

    Parameters
    ----------
    throat_data : ThroatExitData
        Extracted throat exit data
    bempp_grid : bempp_cl.api.Grid, optional
        If provided, creates velocity BC for this grid
    rho : float
        Air density

    Returns
    -------
    dict
        Coupling data:
        - 'mean_velocity': Mean normal velocity for simple piston BC
        - 'velocity_distribution': Full velocity distribution
        - 'throat_radius_m': Throat radius
        - 'throat_area_m2': Total throat area
        - 'neumann_data': Neumann BC coefficients (if bempp_grid provided)
    """
    result = {
        'mean_velocity': throat_data.mean_velocity,
        'velocity_distribution': throat_data.velocity,
        'throat_radius_m': throat_data.throat_radius_m,
        'throat_area_m2': throat_data.total_area_m2,
    }

    if bempp_grid is not None and BEMPP_AVAILABLE:
        omega = 2 * np.pi * throat_data.frequency_hz
        # Neumann BC: g = -iωρ v_n
        neumann_coeffs = -1j * omega * rho * throat_data.velocity
        result['neumann_data'] = neumann_coeffs

    return result


def apply_radiation_bc_at_throat(
    solver: "HelmholtzFEMSolver",
    throat_marker: int,
    *,
    impedance: Optional[complex] = None,
    rho: float = AIR_DENSITY,
    c: float = SPEED_OF_SOUND,
) -> None:
    """
    Apply radiation impedance BC at the throat exit.

    This is a simpler alternative to full BEM coupling. The throat
    is terminated with a radiation impedance that models wave
    propagation into a semi-infinite waveguide or free space.

    Parameters
    ----------
    solver : HelmholtzFEMSolver
        The Helmholtz solver
    throat_marker : int
        Boundary marker for throat exit
    impedance : complex, optional
        Specific acoustic impedance Z [Pa·s/m].
        If None, uses Z₀ = ρc (matched radiation condition).
    rho : float
        Air density
    c : float
        Speed of sound

    Notes
    -----
    The radiation BC is:
        ∂p/∂n = -ik(Z₀/Z)p

    For Z = Z₀ (matched), this becomes the Sommerfeld radiation condition:
        ∂p/∂n = -ikp

    For a waveguide exit, you might use:
        Z = Z₀ * (1 + R)/(1 - R)
    where R is the reflection coefficient.
    """
    if impedance is None:
        impedance = rho * c  # Matched radiation condition

    solver.add_impedance_bc(impedance, throat_marker)


@dataclass
class PressureFieldVisualization:
    """
    Helper for pressure field visualization and export.

    Attributes
    ----------
    coordinates : np.ndarray
        (N, 3) sample point coordinates
    pressure : np.ndarray
        Complex pressure at sample points
    frequency_hz : float
        Analysis frequency
    """
    coordinates: np.ndarray
    pressure: np.ndarray
    frequency_hz: float

    @property
    def pressure_magnitude(self) -> np.ndarray:
        """Pressure magnitude [Pa]."""
        return np.abs(self.pressure)

    @property
    def pressure_phase_deg(self) -> np.ndarray:
        """Pressure phase [degrees]."""
        return np.degrees(np.angle(self.pressure))

    @property
    def spl_db(self) -> np.ndarray:
        """Sound pressure level [dB re 20 µPa]."""
        p_ref = 20e-6
        mag = self.pressure_magnitude
        # Avoid log(0)
        mag = np.maximum(mag, p_ref * 1e-10)
        return 20 * np.log10(mag / p_ref)

    def to_vtk(self, filename: str) -> None:
        """
        Export to VTK point cloud for visualization.

        Parameters
        ----------
        filename : str
            Output filename (.vtp or .vtk)
        """
        try:
            import meshio
        except ImportError:
            raise ImportError("meshio is required for VTK export")

        # Create point cloud with pressure data
        points = self.coordinates

        # Point data
        point_data = {
            'pressure_real': np.real(self.pressure),
            'pressure_imag': np.imag(self.pressure),
            'pressure_magnitude': self.pressure_magnitude,
            'pressure_phase_deg': self.pressure_phase_deg,
            'spl_db': self.spl_db,
        }

        # Create mesh with just vertices (no cells)
        mesh = meshio.Mesh(
            points=points,
            cells=[],
            point_data=point_data,
        )
        mesh.write(filename)

    def to_dict(self) -> dict:
        """Export to dictionary for JSON serialization."""
        return {
            'frequency_hz': self.frequency_hz,
            'coordinates': self.coordinates.tolist(),
            'pressure_real': np.real(self.pressure).tolist(),
            'pressure_imag': np.imag(self.pressure).tolist(),
        }


def sample_pressure_field(
    result: "HelmholtzResult",
    sample_points: np.ndarray,
) -> PressureFieldVisualization:
    """
    Sample the pressure field at arbitrary points.

    Parameters
    ----------
    result : HelmholtzResult
        Solution from HelmholtzFEMSolver
    sample_points : np.ndarray
        (N, 3) points at which to evaluate pressure

    Returns
    -------
    PressureFieldVisualization
        Sampled pressure field data
    """
    pressure = result.pressure_at_points(sample_points)

    return PressureFieldVisualization(
        coordinates=sample_points,
        pressure=pressure.flatten(),
        frequency_hz=result.frequency_hz,
    )


def create_axial_sample_points(
    r_max: float,
    z_min: float,
    z_max: float,
    n_r: int = 20,
    n_z: int = 50,
) -> np.ndarray:
    """
    Create sample points on an axisymmetric cross-section (y=0 plane).

    Parameters
    ----------
    r_max : float
        Maximum radial extent [m]
    z_min : float
        Minimum axial position [m]
    z_max : float
        Maximum axial position [m]
    n_r : int
        Number of radial samples
    n_z : int
        Number of axial samples

    Returns
    -------
    np.ndarray
        (N, 3) sample point coordinates
    """
    r = np.linspace(0, r_max, n_r)
    z = np.linspace(z_min, z_max, n_z)

    R, Z = np.meshgrid(r, z)

    # Sample on y=0 plane (x = r, y = 0, z = z)
    points = np.column_stack([
        R.ravel(),
        np.zeros(R.size),
        Z.ravel(),
    ])

    return points


def create_azimuthal_sample_points(
    r: float,
    z_min: float,
    z_max: float,
    n_theta: int = 36,
    n_z: int = 50,
) -> np.ndarray:
    """
    Create sample points on a cylindrical surface at radius r.

    Parameters
    ----------
    r : float
        Cylinder radius [m]
    z_min : float
        Minimum axial position [m]
    z_max : float
        Maximum axial position [m]
    n_theta : int
        Number of azimuthal samples
    n_z : int
        Number of axial samples

    Returns
    -------
    np.ndarray
        (N, 3) sample point coordinates
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(z_min, z_max, n_z)

    THETA, Z = np.meshgrid(theta, z)

    points = np.column_stack([
        r * np.cos(THETA.ravel()),
        r * np.sin(THETA.ravel()),
        Z.ravel(),
    ])

    return points


def summarize_metrics_sweep(
    metrics_list: List[PhasePlugMetrics],
) -> dict:
    """
    Summarize metrics from a frequency sweep.

    Parameters
    ----------
    metrics_list : List[PhasePlugMetrics]
        Metrics at each frequency

    Returns
    -------
    dict
        Summary containing:
        - 'frequencies_hz': Frequency array
        - 'transmission_efficiency': Array of efficiencies
        - 'pressure_uniformity': Array of uniformities
        - 'phase_spread_deg': Array of phase spreads
        - 'impedance_magnitude': Array of |Z_dome|
        - 'mean_efficiency': Mean transmission efficiency
        - 'min_uniformity': Worst-case uniformity
        - 'max_phase_spread': Worst-case phase spread
    """
    freqs = np.array([m.frequency_hz for m in metrics_list])
    efficiencies = np.array([m.transmission_efficiency for m in metrics_list])
    uniformities = np.array([m.pressure_uniformity for m in metrics_list])
    phase_spreads = np.array([m.phase_spread_deg for m in metrics_list])
    z_mags = np.array([np.abs(m.dome_impedance) for m in metrics_list])

    return {
        'frequencies_hz': freqs,
        'transmission_efficiency': efficiencies,
        'pressure_uniformity': uniformities,
        'phase_spread_deg': phase_spreads,
        'impedance_magnitude': z_mags,
        'mean_efficiency': float(np.mean(efficiencies)),
        'min_uniformity': float(np.min(uniformities)),
        'max_phase_spread': float(np.max(phase_spreads)),
    }
