"""
FEM-BEM coupling utilities for shell surfaces.

This module provides functions to transfer data between DOLFINx shell FEM
meshes and bempp-cl BEM grids. Unlike the standard bempp_cl.api.external.fenicsx
module which extracts boundary surfaces from 3D tetrahedral meshes, this module
works directly with 2D manifold shell meshes.

Key functions:
- shell_surface_to_bempp_grid: Convert a 2D DOLFINx mesh to bempp Grid
- create_neumann_grid_function: Build Neumann BC GridFunction from velocity
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np
import importlib.util

try:
    import bempp_cl.api
    BEMPP_AVAILABLE = True
except ImportError:
    BEMPP_AVAILABLE = False
    bempp_cl = None

def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


DOLFINX_AVAILABLE = _has_module("dolfinx")


def shell_surface_to_bempp_grid(
    fenics_mesh,
    *,
    ensure_outward_normals: bool = True,
    reference_point: Optional[np.ndarray] = None,
) -> tuple["bempp_cl.api.Grid", np.ndarray]:
    """
    Create a bempp-cl Grid from a 2D DOLFINx surface mesh.

    Unlike `bempp_cl.api.external.fenicsx.boundary_grid_from_fenics_mesh()`,
    this function works directly with 2D manifold meshes (e.g., shell surfaces)
    rather than extracting boundaries from 3D tetrahedral meshes.

    Parameters
    ----------
    fenics_mesh : dolfinx.mesh.Mesh
        A DOLFINx mesh of topological dimension 2 (triangular surface).
    ensure_outward_normals : bool
        If True, orient triangle normals consistently (away from reference_point
        or in +z direction if no reference is given).
    reference_point : np.ndarray, optional
        Point considered "inside" the structure. Normals will point away from
        this point. If None, uses centroid of mesh.

    Returns
    -------
    bempp_grid : bempp_cl.api.Grid
        The bempp surface mesh.
    vertex_map : np.ndarray
        Mapping from DOLFINx vertex indices to bempp vertex indices.

    Raises
    ------
    ImportError
        If bempp-cl or DOLFINx is not available.
    ValueError
        If the mesh is not a 2D surface mesh.

    Example
    -------
    >>> from dolfinx.mesh import create_unit_square
    >>> from mpi4py import MPI
    >>> fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    >>> bempp_grid, vertex_map = shell_surface_to_bempp_grid(fenics_mesh)
    """
    if not BEMPP_AVAILABLE:
        raise ImportError("bempp-cl is required for FEM-BEM coupling")
    if not DOLFINX_AVAILABLE:
        raise ImportError("DOLFINx is required for FEM-BEM coupling")

    # Verify mesh dimension
    tdim = fenics_mesh.topology.dim
    gdim = fenics_mesh.geometry.dim

    if tdim != 2:
        raise ValueError(
            f"Expected a 2D surface mesh (tdim=2), got tdim={tdim}. "
            "For 3D meshes, use bempp_cl.api.external.fenicsx.boundary_grid_from_fenics_mesh()"
        )

    # Create necessary connectivity
    fenics_mesh.topology.create_entities(0)  # vertices
    fenics_mesh.topology.create_entities(2)  # cells (triangles)
    fenics_mesh.topology.create_connectivity(2, 0)

    # Get geometry coordinates
    coords = fenics_mesh.geometry.x  # (N_nodes, gdim)

    # Ensure 3D coordinates for bempp
    if gdim == 2:
        coords_3d = np.zeros((coords.shape[0], 3), dtype=np.float64)
        coords_3d[:, :2] = coords
    else:
        coords_3d = coords.astype(np.float64)

    # Get cell connectivity
    # For linear triangles, geometry.dofmap gives the vertex indices per cell
    dofmap = fenics_mesh.geometry.dofmap
    num_cells = fenics_mesh.topology.index_map(2).size_local

    # Handle different DOLFINx API versions
    if hasattr(dofmap, 'links'):
        cells = np.array([dofmap.links(i) for i in range(num_cells)])
    elif hasattr(dofmap, 'array'):
        # Newer API: dofmap is an adjacency list
        cells = dofmap.array.reshape(-1, 3)[:num_cells]
    else:
        # Try direct array access
        cells = np.array(dofmap).reshape(-1, 3)[:num_cells]

    # Verify triangular elements
    if cells.shape[1] != 3:
        raise ValueError(
            f"Expected triangular elements (3 vertices), got {cells.shape[1]} vertices per cell"
        )

    # Orient normals consistently if requested
    if ensure_outward_normals:
        if reference_point is None:
            # Use mesh centroid as reference
            reference_point = coords_3d.mean(axis=0)

        cells = _orient_triangles_outward(coords_3d, cells, reference_point)

    # Create bempp grid
    # bempp expects (3, N_vertices) and (3, N_cells)
    bempp_coords = coords_3d.T.copy()
    bempp_cells = cells.T.astype(np.uint32).copy()

    bempp_grid = bempp_cl.api.Grid(bempp_coords, bempp_cells)

    # Create vertex map (identity in this case since we use all vertices)
    vertex_map = np.arange(coords_3d.shape[0])

    return bempp_grid, vertex_map


def _orient_triangles_outward(
    coords: np.ndarray,
    cells: np.ndarray,
    reference_point: np.ndarray,
) -> np.ndarray:
    """
    Orient triangle normals to point away from a reference point.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) vertex coordinates
    cells : np.ndarray
        (M, 3) triangle connectivity
    reference_point : np.ndarray
        Point considered "inside"

    Returns
    -------
    np.ndarray
        Reoriented cell connectivity
    """
    cells = cells.copy()

    for i, tri in enumerate(cells):
        v0, v1, v2 = coords[tri[0]], coords[tri[1]], coords[tri[2]]
        centroid = (v0 + v1 + v2) / 3
        normal = np.cross(v1 - v0, v2 - v0)

        # Vector from reference to centroid (outward direction)
        outward = centroid - reference_point

        # If normal points inward, flip the triangle
        if np.dot(normal, outward) < 0:
            cells[i, 1], cells[i, 2] = cells[i, 2], cells[i, 1]

    return cells


def create_p1_trace_space(
    bempp_grid,
    fenics_space=None,
) -> tuple["bempp_cl.api.Space", "object | None"]:
    """
    Create a P1 function space on the bempp grid and optionally the trace matrix.

    Parameters
    ----------
    bempp_grid : bempp_cl.api.Grid
        The bempp surface mesh.
    fenics_space : dolfinx.fem.FunctionSpace, optional
        If provided, creates a trace matrix for FEM→BEM coefficient transfer.

    Returns
    -------
    bempp_space : bempp_cl.api.function_space
        P1 continuous function space on the grid.
    trace_matrix : scipy.sparse.csr_matrix or None
        Sparse matrix mapping FEM coefficients to BEM coefficients.
        None if fenics_space is not provided.
    """
    if not BEMPP_AVAILABLE:
        raise ImportError("bempp-cl is required")

    bempp_space = bempp_cl.api.function_space(bempp_grid, "P", 1)

    if fenics_space is None:
        return bempp_space, None

    # For shell surfaces, the FEM space and BEM space have the same DOFs
    # if we're using P1 elements on the same mesh
    from scipy.sparse import eye

    fem_size = fenics_space.dofmap.index_map.size_global
    bem_size = bempp_space.global_dof_count

    if fem_size == bem_size:
        # Identity mapping (same mesh)
        trace_matrix = eye(fem_size, format="csr")
    else:
        raise NotImplementedError(
            f"FEM and BEM DOF counts differ ({fem_size} vs {bem_size}). "
            "Interpolation not yet implemented."
        )

    return bempp_space, trace_matrix


def shell_displacement_to_velocity(
    displacement: np.ndarray,
    frequency_hz: float,
) -> np.ndarray:
    """
    Convert shell displacement to velocity for harmonic excitation.

    For harmonic motion: v = jω * u, where u is displacement.

    Parameters
    ----------
    displacement : np.ndarray
        Complex displacement field (can be nodal values or coefficients).
    frequency_hz : float
        Excitation frequency [Hz].

    Returns
    -------
    np.ndarray
        Complex velocity field.
    """
    omega = 2 * np.pi * frequency_hz
    return 1j * omega * displacement


def create_neumann_grid_function(
    bempp_space,
    normal_velocity: np.ndarray | Callable,
) -> "bempp_cl.api.GridFunction":
    """
    Create a bempp GridFunction for Neumann BC from shell velocity data.

    Parameters
    ----------
    bempp_space : bempp_cl.api.function_space
        The BEM function space (typically DP0 for Neumann data).
    normal_velocity : np.ndarray or Callable
        Either:
        - An array of normal velocity values at DOFs (complex)
        - A callable f(x, n, domain_index, result) for evaluation
    Returns
    -------
    bempp_cl.api.GridFunction
        Grid function representing the Neumann boundary condition.
    """
    if not BEMPP_AVAILABLE:
        raise ImportError("bempp-cl is required")

    if callable(normal_velocity):
        # Use as a callable directly
        @bempp_cl.api.complex_callable
        def neumann_data(x, n, domain_index, result):
            normal_velocity(x, n, domain_index, result)

        return bempp_cl.api.GridFunction(bempp_space, fun=neumann_data)
    else:
        # Use as coefficient array
        coeffs = np.asarray(normal_velocity, dtype=np.complex128)
        return bempp_cl.api.GridFunction(bempp_space, coefficients=coeffs)


def compute_radiation_impedance(
    bempp_grid,
    frequency_hz: float,
    velocity_distribution: np.ndarray,
    *,
    wavenumber: float | None = None,
    rho: float = 1.225,
    c: float = 343.0,
    return_pressure: bool = False,
) -> complex | tuple[complex, np.ndarray]:
    """
    Compute radiation impedance from a vibrating shell surface.

    Z_rad = ∫∫ p * v_n^* dS / ∫∫ |v_n|² dS

    This uses BEM to compute the surface pressure from the velocity BC,
    then integrates to find the radiation impedance.

    Parameters
    ----------
    bempp_grid : bempp_cl.api.Grid
        Surface mesh of the radiator.
    frequency_hz : float
        Frequency [Hz].
    velocity_distribution : np.ndarray
        Complex normal velocity at each element (or node).
    wavenumber : float, optional
        Acoustic wavenumber. If None, computed from frequency.
    rho : float
        Air density [kg/m³].
    c : float
        Speed of sound [m/s].
    return_pressure : bool
        If True, also return the surface pressure distribution.

    Returns
    -------
    z_rad : complex
        Radiation impedance [Pa·s/m³] (acoustic ohms).
    pressure : np.ndarray (optional)
        Surface pressure distribution (if return_pressure=True).
    """
    if not BEMPP_AVAILABLE:
        raise ImportError("bempp-cl is required")

    from scipy.sparse.linalg import gmres as scipy_gmres

    if wavenumber is None:
        wavenumber = 2 * np.pi * frequency_hz / c

    # Create function spaces
    dp0_space = bempp_cl.api.function_space(bempp_grid, "DP", 0)

    # Create velocity grid function
    velocity_distribution = np.asarray(velocity_distribution, dtype=np.complex128)

    # Set up Helmholtz BEM operators
    # Using combined field formulation: (I/2 + K' - iηV) p = -iωρ v_n
    identity = bempp_cl.api.operators.boundary.sparse.identity(
        dp0_space, dp0_space, dp0_space
    )
    slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(
        dp0_space, dp0_space, dp0_space, wavenumber
    )
    adlp = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
        dp0_space, dp0_space, dp0_space, wavenumber
    )

    eta = wavenumber  # Burton-Miller coupling parameter

    # LHS operator
    lhs = 0.5 * identity + adlp - 1j * eta * slp

    # RHS: -iωρ * v_n
    omega = 2 * np.pi * frequency_hz
    rhs_coeffs = -1j * omega * rho * velocity_distribution

    # Solve using scipy GMRES with strong_form discretization
    lhs_discrete = lhs.strong_form()
    x, info = scipy_gmres(lhs_discrete, rhs_coeffs, rtol=1e-5)

    if info != 0:
        raise RuntimeError(f"GMRES failed to converge (info={info})")

    # Create grid function from solution
    p_coeffs = x

    # Compute radiation impedance
    # Z = ∫ p v_n^* dS / ∫ |v_n|² dS
    v_coeffs = velocity_distribution

    # Get element areas for integration
    # For DP0, each coefficient corresponds to one element
    vertices = np.asarray(bempp_grid.vertices, dtype=float)  # (3, n_vertices)
    elements = np.asarray(bempp_grid.elements, dtype=int)  # (3, n_elements)
    p0 = vertices[:, elements[0, :]]
    p1 = vertices[:, elements[1, :]]
    p2 = vertices[:, elements[2, :]]
    # Triangle area = 0.5 * ||(p1-p0) x (p2-p0)||
    areas = 0.5 * np.linalg.norm(np.cross((p1 - p0).T, (p2 - p0).T), axis=1)

    numerator = np.sum(p_coeffs * np.conj(v_coeffs) * areas)
    denominator = np.sum(np.abs(v_coeffs)**2 * areas)

    z_rad = numerator / denominator

    if return_pressure:
        return z_rad, p_coeffs
    return z_rad


def get_element_areas(bempp_grid) -> np.ndarray:
    """
    Compute element areas for a bempp grid.

    Parameters
    ----------
    bempp_grid : bempp_cl.api.Grid
        Surface mesh.

    Returns
    -------
    np.ndarray
        Area of each element [m²].
    """
    vertices = np.asarray(bempp_grid.vertices, dtype=float)  # (3, n_vertices)
    elements = np.asarray(bempp_grid.elements, dtype=int)  # (3, n_elements)
    p0 = vertices[:, elements[0, :]]
    p1 = vertices[:, elements[1, :]]
    p2 = vertices[:, elements[2, :]]
    return 0.5 * np.linalg.norm(np.cross((p1 - p0).T, (p2 - p0).T), axis=1)


def get_element_centroids(bempp_grid) -> np.ndarray:
    """
    Compute element centroids for a bempp grid.

    Parameters
    ----------
    bempp_grid : bempp_cl.api.Grid
        Surface mesh.

    Returns
    -------
    np.ndarray
        Centroid coordinates (n_elements, 3) [m].
    """
    vertices = np.asarray(bempp_grid.vertices, dtype=float)  # (3, n_vertices)
    elements = np.asarray(bempp_grid.elements, dtype=int)  # (3, n_elements)
    p0 = vertices[:, elements[0, :]]
    p1 = vertices[:, elements[1, :]]
    p2 = vertices[:, elements[2, :]]
    centroids = ((p0 + p1 + p2) / 3).T
    return centroids


def pressure_to_fem_surface_load(
    pressure_coeffs: np.ndarray,
    bempp_grid,
    *,
    normal_direction: str = "outward",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert BEM surface pressure to FEM surface load (force per area).

    The acoustic pressure acts normal to the surface. For shell FEM,
    this becomes a distributed load in the transverse direction.

    Parameters
    ----------
    pressure_coeffs : np.ndarray
        Complex pressure at each element (DP0 coefficients).
    bempp_grid : bempp_cl.api.Grid
        Surface mesh (same as used for BEM).
    normal_direction : str
        "outward" (pressure pushes outward) or "inward".

    Returns
    -------
    load_per_area : np.ndarray
        Pressure magnitude at each element [Pa].
    element_centroids : np.ndarray
        Centroid coordinates (n_elements, 3) for applying loads.

    Notes
    -----
    For FEM integration, multiply by element areas and shape functions
    to form the load vector. This function provides the pressure values
    at element centroids for use with DOLFINx's load application methods.
    """
    pressure_coeffs = np.asarray(pressure_coeffs, dtype=np.complex128)
    centroids = get_element_centroids(bempp_grid)

    # Pressure acts normal to surface; sign convention:
    # - Positive pressure = compression (acts inward)
    # - For shell transverse loads, we typically want pressure pointing
    #   in the direction of positive w (outward)
    if normal_direction == "inward":
        load_per_area = -pressure_coeffs
    else:
        load_per_area = pressure_coeffs

    return load_per_area, centroids


def iterative_fem_bem_coupling(
    fem_solve_func: Callable,
    bem_solve_func: Callable,
    initial_velocity: np.ndarray,
    *,
    max_iterations: int = 10,
    tolerance: float = 1e-4,
    relaxation: float = 0.5,
) -> dict:
    """
    Iterative two-way FEM-BEM coupling solver.

    This implements a fixed-point iteration between:
    1. FEM shell solver (velocity from applied load + acoustic pressure)
    2. BEM acoustic solver (pressure from velocity BC)

    Parameters
    ----------
    fem_solve_func : Callable
        Function: fem_solve_func(acoustic_pressure) -> velocity
        Takes acoustic pressure distribution, returns surface velocity.
    bem_solve_func : Callable
        Function: bem_solve_func(velocity) -> pressure
        Takes surface velocity, returns acoustic pressure.
    initial_velocity : np.ndarray
        Initial guess for velocity distribution.
    max_iterations : int
        Maximum iterations.
    tolerance : float
        Convergence tolerance (relative change in velocity).
    relaxation : float
        Under-relaxation factor (0 < α ≤ 1). Lower values improve stability
        but slow convergence.

    Returns
    -------
    dict
        Results containing:
        - 'velocity': Final converged velocity
        - 'pressure': Final converged pressure
        - 'iterations': Number of iterations
        - 'converged': Whether iteration converged
        - 'residual_history': List of residuals per iteration

    Notes
    -----
    The iteration scheme is:
        v^{n+1} = α * v_new + (1-α) * v^n

    where v_new = fem_solve(bem_solve(v^n))

    For strongly coupled problems (large acoustic loading relative to
    structural stiffness), relaxation < 0.5 may be needed.

    When to use two-way coupling:
    - Lightweight radiators (thin domes) at high frequencies
    - When radiation impedance significantly affects resonance frequencies
    - When predicted SPL differs significantly from one-way coupling

    One-way coupling is typically sufficient when:
    - Heavy/stiff radiators (radiation load << structural impedance)
    - Low frequencies (ka << 1 where k=wavenumber, a=radiator size)
    - Initial design exploration (faster computation)

    Example
    -------
    >>> def fem_solve(pressure):
    ...     # Solve shell FEM with pressure load
    ...     return velocity_from_fem
    >>> def bem_solve(velocity):
    ...     # Solve exterior Helmholtz with velocity BC
    ...     return pressure_from_bem
    >>> result = iterative_fem_bem_coupling(
    ...     fem_solve, bem_solve,
    ...     initial_velocity=v0,
    ...     relaxation=0.3,
    ... )
    >>> if result['converged']:
    ...     final_velocity = result['velocity']
    """
    velocity = np.asarray(initial_velocity, dtype=np.complex128).copy()
    residual_history = []

    for iteration in range(max_iterations):
        # BEM: velocity -> pressure
        pressure = bem_solve_func(velocity)

        # FEM: pressure -> new velocity
        velocity_new = fem_solve_func(pressure)

        # Compute residual
        diff = velocity_new - velocity
        residual = np.linalg.norm(diff) / (np.linalg.norm(velocity) + 1e-16)
        residual_history.append(residual)

        # Under-relaxation update
        velocity = relaxation * velocity_new + (1 - relaxation) * velocity

        if residual < tolerance:
            return {
                'velocity': velocity,
                'pressure': pressure,
                'iterations': iteration + 1,
                'converged': True,
                'residual_history': residual_history,
            }

    # Did not converge
    return {
        'velocity': velocity,
        'pressure': pressure,
        'iterations': max_iterations,
        'converged': False,
        'residual_history': residual_history,
    }


def assess_coupling_strength(
    bempp_grid,
    frequency_hz: float,
    shell_mass_per_area: float,
    shell_stiffness: float,
    *,
    rho: float = 1.225,
    c: float = 343.0,
) -> dict:
    """
    Assess whether two-way FEM-BEM coupling is needed.

    Compares the acoustic radiation impedance to the structural impedance
    to determine if acoustic loading significantly affects the response.

    Parameters
    ----------
    bempp_grid : bempp_cl.api.Grid
        Surface mesh of the radiator.
    frequency_hz : float
        Analysis frequency [Hz].
    shell_mass_per_area : float
        Shell mass per unit area ρ*h [kg/m²].
    shell_stiffness : float
        Approximate structural stiffness [N/m²] (can use flexural rigidity / a²
        where a is characteristic length).
    rho : float
        Air density [kg/m³].
    c : float
        Speed of sound [m/s].

    Returns
    -------
    dict
        Assessment results:
        - 'coupling_ratio': |Z_acoustic| / |Z_structural|
        - 'ka': Acoustic compactness parameter
        - 'recommendation': "one-way" or "two-way"
        - 'reason': Explanation of recommendation
    """
    omega = 2 * np.pi * frequency_hz
    k = omega / c

    # Estimate radiator size from grid
    vertices = np.asarray(bempp_grid.vertices, dtype=float)
    centroid = vertices.mean(axis=1)
    distances = np.linalg.norm(vertices.T - centroid, axis=1)
    a = distances.max()  # Characteristic radius

    ka = k * a

    # Structural impedance (simplified lumped model)
    # Z_struct ≈ j*ω*m + k/jω for mass-spring system
    z_mass = 1j * omega * shell_mass_per_area
    z_stiffness = shell_stiffness / (1j * omega)
    z_structural = abs(z_mass + z_stiffness)

    # Acoustic impedance estimate (piston in infinite baffle)
    # Real part (radiation resistance): R ≈ ρc (ka)² / 2 for ka << 1
    # Imaginary part (radiation reactance): X ≈ ρc * (8ka / 3π) for ka << 1
    if ka < 1:
        r_rad = rho * c * (ka**2) / 2
        x_rad = rho * c * 8 * ka / (3 * np.pi)
    else:
        # For ka > 1, radiation resistance approaches ρc
        r_rad = rho * c
        x_rad = rho * c / ka

    z_acoustic = np.sqrt(r_rad**2 + x_rad**2)

    # Coupling ratio
    coupling_ratio = z_acoustic / (z_structural + 1e-16)

    # Recommendation
    if coupling_ratio < 0.01:
        recommendation = "one-way"
        reason = f"Acoustic loading is negligible ({coupling_ratio:.1%} of structural impedance)"
    elif coupling_ratio < 0.1:
        recommendation = "one-way"
        reason = f"Acoustic loading is small ({coupling_ratio:.1%}); one-way coupling is adequate for most applications"
    elif coupling_ratio < 0.3:
        recommendation = "two-way"
        reason = f"Moderate acoustic loading ({coupling_ratio:.1%}); two-way coupling recommended for accuracy"
    else:
        recommendation = "two-way"
        reason = f"Strong acoustic loading ({coupling_ratio:.1%}); two-way coupling essential"

    return {
        'coupling_ratio': coupling_ratio,
        'ka': ka,
        'z_acoustic': z_acoustic,
        'z_structural': z_structural,
        'recommendation': recommendation,
        'reason': reason,
    }
