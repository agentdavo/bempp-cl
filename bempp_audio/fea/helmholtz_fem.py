"""
3D Helmholtz FEM solver for interior acoustic domains.

This module provides a DOLFINx-based finite element solver for the
Helmholtz equation in 3D domains, suitable for modeling:
- Phase plug channels in compression drivers
- Rear compression chambers
- Horn/waveguide throat sections

The weak form solved is:
    ∫ (∇p · ∇q - k²pq) dΩ = ∫ g q dΓ_N

where p is the acoustic pressure, k is the wavenumber, and g is the
Neumann boundary data (related to normal velocity via g = -iωρ v_n).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Tuple, Union, TYPE_CHECKING
import numpy as np
import importlib.util

if TYPE_CHECKING:  # pragma: no cover
    import dolfinx  # noqa: F401
    from dolfinx import fem, mesh as dfx_mesh, io  # noqa: F401
    from dolfinx.fem import petsc  # noqa: F401
    import ufl  # noqa: F401
    from mpi4py import MPI  # noqa: F401
    from petsc4py import PETSc  # noqa: F401


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


DOLFINX_AVAILABLE = _has_module("dolfinx")
_DOLFINX_CACHE = None

# Runtime-loaded modules (set by _require_dolfinx). These are module globals so
# the implementation below can use `fem/ufl/PETSc/...` without importing them at
# import-time.
dolfinx = None
fem = None
dfx_mesh = None
io = None
petsc = None
ufl = None
MPI = None
PETSc = None


def _require_dolfinx():
    """
    Import DOLFINx stack lazily.

    Important: importing dolfinx/petsc4py can trigger MPI init on some systems.
    Keeping these imports out of module import-time avoids hard failures when
    users only need geometry/meshing utilities.
    """
    global _DOLFINX_CACHE
    if _DOLFINX_CACHE is not None:
        return _DOLFINX_CACHE
    if not DOLFINX_AVAILABLE:
        raise ImportError("dolfinx is required for HelmholtzFEMSolver (install python3-dolfinx-complex).")

    import dolfinx
    from dolfinx import fem, mesh as dfx_mesh, io
    from dolfinx.fem import petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    # Populate globals for the rest of this module.
    globals().update(
        {
            "dolfinx": dolfinx,
            "fem": fem,
            "dfx_mesh": dfx_mesh,
            "io": io,
            "petsc": petsc,
            "ufl": ufl,
            "MPI": MPI,
            "PETSc": PETSc,
        }
    )

    _DOLFINX_CACHE = (dolfinx, fem, dfx_mesh, io, petsc, ufl, MPI, PETSc)
    return _DOLFINX_CACHE


# Default air properties at 20°C, 1 atm
AIR_DENSITY = 1.225  # kg/m³
SPEED_OF_SOUND = 343.0  # m/s


@dataclass
class AcousticMedium:
    """Properties of the acoustic medium."""

    density: float = AIR_DENSITY  # kg/m³
    speed_of_sound: float = SPEED_OF_SOUND  # m/s

    @property
    def characteristic_impedance(self) -> float:
        """Characteristic acoustic impedance Z₀ = ρc [Pa·s/m]."""
        return self.density * self.speed_of_sound

    def wavenumber(self, frequency_hz: float) -> float:
        """Acoustic wavenumber k = ω/c [rad/m]."""
        return 2 * np.pi * frequency_hz / self.speed_of_sound


@dataclass
class HelmholtzResult:
    """
    Result of a Helmholtz FEM solve.

    Attributes
    ----------
    pressure : dolfinx.fem.Function
        Complex pressure field solution
    frequency_hz : float
        Frequency at which the solution was computed
    wavenumber : float
        Acoustic wavenumber k = ω/c
    mesh : dolfinx.mesh.Mesh
        The computational mesh
    """

    pressure: "fem.Function"
    frequency_hz: float
    wavenumber: float
    mesh: "dfx_mesh.Mesh"

    def pressure_at_points(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate pressure at arbitrary points.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) array of evaluation points

        Returns
        -------
        np.ndarray
            Complex pressure values at each point
        """
        dolfinx, fem, dfx_mesh, io, petsc, ufl, MPI, PETSc = _require_dolfinx()
        from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

        # Build bounding box tree
        tree = bb_tree(self.mesh, self.mesh.topology.dim)

        # Find cells containing points
        cell_candidates = compute_collisions_points(tree, points)
        cells = compute_colliding_cells(self.mesh, cell_candidates, points)

        # Evaluate function
        return self.pressure.eval(points, cells.array)

    def surface_integral(
        self,
        boundary_marker: int,
        facet_tags: "dfx_mesh.MeshTags",
    ) -> complex:
        """
        Compute integral of pressure over a marked boundary.

        Useful for computing acoustic power flow through surfaces.
        """
        dolfinx, fem, dfx_mesh, io, petsc, ufl, MPI, PETSc = _require_dolfinx()
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tags)
        integral = fem.assemble_scalar(fem.form(self.pressure * ds(boundary_marker)))
        return complex(integral)


class HelmholtzFEMSolver:
    """
    3D Helmholtz equation solver using DOLFINx.

    Solves the interior Helmholtz problem with various boundary conditions:
    - Neumann (velocity): ∂p/∂n = -iωρ v_n
    - Robin (impedance): ∂p/∂n + ik(Z₀/Z)p = 0
    - Dirichlet (pressure): p = p₀

    Example
    -------
    >>> from dolfinx.mesh import create_unit_cube
    >>> mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    >>> solver = HelmholtzFEMSolver(mesh)
    >>> # Add velocity BC on left face (x=0)
    >>> solver.add_velocity_bc(lambda x: 1.0, marker=1, facet_tags=tags)
    >>> # Solve at 1 kHz
    >>> result = solver.solve(frequency_hz=1000.0)
    """

    def __init__(
        self,
        mesh: "dfx_mesh.Mesh",
        degree: int = 1,
        medium: Optional[AcousticMedium] = None,
    ):
        """
        Initialize the Helmholtz solver.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            3D tetrahedral mesh of the acoustic domain
        degree : int
            Polynomial degree of the finite element space
        medium : AcousticMedium, optional
            Acoustic medium properties (default: air at 20°C)
        """
        _require_dolfinx()

        self.mesh = mesh
        self.degree = degree
        self.medium = medium or AcousticMedium()

        # Create complex-valued function space
        self.V = fem.functionspace(mesh, ("Lagrange", degree))

        # Trial and test functions
        self.p = ufl.TrialFunction(self.V)
        self.q = ufl.TestFunction(self.V)

        # Boundary conditions storage
        self._velocity_bcs: list = []
        self._impedance_bcs: list = []
        self._dirichlet_bcs: list = []

        # Facet tags (set when BCs are added)
        self._facet_tags: Optional["dfx_mesh.MeshTags"] = None

    def set_facet_tags(self, facet_tags: "dfx_mesh.MeshTags") -> None:
        """
        Set the facet tags for boundary identification.

        Parameters
        ----------
        facet_tags : dolfinx.mesh.MeshTags
            Mesh tags marking different boundary regions
        """
        self._facet_tags = facet_tags

    def add_velocity_bc(
        self,
        velocity: Union[complex, Callable, np.ndarray],
        marker: int,
    ) -> None:
        """
        Add a velocity (Neumann) boundary condition.

        The normal velocity v_n is related to the Neumann data by:
            ∂p/∂n = -iωρ v_n

        Parameters
        ----------
        velocity : complex, Callable, or np.ndarray
            Normal velocity value(s). Can be:
            - A constant complex value
            - A callable f(x) -> complex
            - An array of values at boundary DOFs
        marker : int
            Boundary marker identifying the surface
        """
        self._velocity_bcs.append((velocity, marker))

    def add_impedance_bc(
        self,
        impedance: Union[complex, Callable],
        marker: int,
    ) -> None:
        """
        Add an impedance (Robin) boundary condition.

        Models a surface with specific acoustic impedance Z:
            ∂p/∂n = -ik(Z₀/Z)p

        For a radiation condition into a semi-infinite duct, use Z = Z₀.

        Parameters
        ----------
        impedance : complex or Callable
            Specific acoustic impedance Z [Pa·s/m]
        marker : int
            Boundary marker identifying the surface
        """
        self._impedance_bcs.append((impedance, marker))

    def add_pressure_bc(
        self,
        pressure: Union[complex, Callable],
        marker: int,
    ) -> None:
        """
        Add a pressure (Dirichlet) boundary condition.

        Parameters
        ----------
        pressure : complex or Callable
            Pressure value(s)
        marker : int
            Boundary marker identifying the surface
        """
        self._dirichlet_bcs.append((pressure, marker))

    def clear_bcs(self) -> None:
        """Clear all boundary conditions."""
        self._velocity_bcs.clear()
        self._impedance_bcs.clear()
        self._dirichlet_bcs.clear()

    def solve(
        self,
        frequency_hz: float,
        solver_options: Optional[Dict] = None,
    ) -> HelmholtzResult:
        """
        Solve the Helmholtz equation at a given frequency.

        Parameters
        ----------
        frequency_hz : float
            Frequency [Hz]
        solver_options : dict, optional
            PETSc solver options

        Returns
        -------
        HelmholtzResult
            Solution container with pressure field
        """
        k = self.medium.wavenumber(frequency_hz)
        omega = 2 * np.pi * frequency_hz
        rho = self.medium.density

        # Measures for integration
        dx = ufl.Measure("dx", domain=self.mesh)

        if self._facet_tags is not None:
            ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self._facet_tags)
        else:
            ds = ufl.Measure("ds", domain=self.mesh)

        # Bilinear form: a(p, q) = ∫ (∇p·∇q - k²pq) dΩ
        a = ufl.inner(ufl.grad(self.p), ufl.grad(self.q)) * dx
        a -= k**2 * self.p * self.q * dx

        # Add impedance BC contributions to bilinear form
        # Robin BC: ∂p/∂n + ik(Z₀/Z)p = 0 → adds -ik(Z₀/Z)pq to LHS
        for impedance, marker in self._impedance_bcs:
            Z0 = self.medium.characteristic_impedance
            if callable(impedance):
                # TODO: handle spatially varying impedance
                raise NotImplementedError("Spatially varying impedance not yet supported")
            else:
                Z = complex(impedance)
                beta = 1j * k * Z0 / Z
                a += beta * self.p * self.q * ds(marker)

        # Linear form: L(q) = ∫ g q dΓ_N
        # Start with zero RHS
        L = fem.Constant(self.mesh, PETSc.ScalarType(0.0)) * self.q * dx

        # Add velocity BC contributions
        # Neumann BC: ∂p/∂n = -iωρ v_n
        for velocity, marker in self._velocity_bcs:
            if callable(velocity):
                # Create expression from callable
                # Note: need to handle complex values properly
                raise NotImplementedError("Callable velocity BC not yet fully supported")
            else:
                v_n = complex(velocity)
                g = -1j * omega * rho * v_n
                g_const = fem.Constant(self.mesh, PETSc.ScalarType(g))
                L += g_const * self.q * ds(marker)

        # Compile forms
        a_form = fem.form(a)
        L_form = fem.form(L)

        # Handle Dirichlet BCs
        bcs = []
        for pressure, marker in self._dirichlet_bcs:
            # Find DOFs on marked boundary
            facets = self._facet_tags.find(marker)
            dofs = fem.locate_dofs_topological(
                self.V, self.mesh.topology.dim - 1, facets
            )
            if callable(pressure):
                raise NotImplementedError("Callable pressure BC not yet supported")
            else:
                p_val = complex(pressure)
                bc = fem.dirichletbc(PETSc.ScalarType(p_val), dofs, self.V)
                bcs.append(bc)

        # Assemble system
        A = petsc.assemble_matrix(a_form, bcs=bcs)
        A.assemble()

        b = petsc.assemble_vector(L_form)
        petsc.apply_lifting(b, [a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)

        # Create solution function
        p_h = fem.Function(self.V, dtype=np.complex128)

        # Solve linear system
        solver = PETSc.KSP().create(self.mesh.comm)
        solver.setOperators(A)

        # Configure solver
        opts = solver_options or {}
        solver.setType(opts.get("ksp_type", "preonly"))
        pc = solver.getPC()
        pc.setType(opts.get("pc_type", "lu"))

        solver.solve(b, p_h.x.petsc_vec)
        p_h.x.scatter_forward()

        return HelmholtzResult(
            pressure=p_h,
            frequency_hz=frequency_hz,
            wavenumber=k,
            mesh=self.mesh,
        )

    def solve_frequency_sweep(
        self,
        frequencies_hz: np.ndarray,
        **kwargs,
    ) -> list[HelmholtzResult]:
        """
        Solve at multiple frequencies.

        Parameters
        ----------
        frequencies_hz : np.ndarray
            Array of frequencies [Hz]
        **kwargs
            Additional arguments passed to solve()

        Returns
        -------
        list[HelmholtzResult]
            Results at each frequency
        """
        results = []
        for f in frequencies_hz:
            result = self.solve(f, **kwargs)
            results.append(result)
        return results


def compute_acoustic_power(
    result: HelmholtzResult,
    facet_tags: "dfx_mesh.MeshTags",
    marker: int,
    medium: Optional[AcousticMedium] = None,
) -> float:
    """
    Compute time-averaged acoustic power through a surface.

    P = (1/2) Re{ ∫ p v_n^* dS }

    For a surface with known normal velocity v_n.

    Parameters
    ----------
    result : HelmholtzResult
        Solution from HelmholtzFEMSolver
    facet_tags : dolfinx.mesh.MeshTags
        Boundary markers
    marker : int
        Surface marker
    medium : AcousticMedium, optional
        Acoustic medium (for impedance calculation)

    Returns
    -------
    float
        Time-averaged acoustic power [W]
    """
    _require_dolfinx()
    medium = medium or AcousticMedium()
    omega = 2 * np.pi * result.frequency_hz
    rho = medium.density

    # Get pressure gradient normal component (proportional to velocity)
    # ∂p/∂n = -iωρ v_n → v_n = (i/ωρ) ∂p/∂n

    # For simplicity, use impedance relation if available
    # This is a simplified computation; full implementation would
    # require evaluating the normal derivative

    ds = ufl.Measure("ds", domain=result.mesh, subdomain_data=facet_tags)
    p = result.pressure

    # Compute |p|² integral (proxy for power in some cases)
    p_squared = fem.assemble_scalar(fem.form(ufl.inner(p, p) * ds(marker)))

    # Actual power requires normal velocity; return |p|² / (2 ρc) as estimate
    Z0 = medium.characteristic_impedance
    return float(np.real(p_squared)) / (2 * Z0)
