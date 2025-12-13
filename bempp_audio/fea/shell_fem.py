"""
Mindlin–Reissner plate/shell FEM utilities (DOLFINx).

This implements a practical Mindlin–Reissner *plate* model (flat mid-surface)
with 5 primary DOFs per node: (u, v, w, θx, θy).

Notes
-----
* This module is intended as Phase 7.2 infrastructure (TASKS.md).
* It currently targets **flat plates** (gdim=2). A fully general curved-shell
  formulation on an arbitrary triangulated manifold requires additional
  surface differential-geometry/discrete-curvature machinery and is not
  implemented here yet.
* Imports of dolfinx/petsc4py/slepc4py are done lazily to avoid hard MPI init
  crashes on misconfigured systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING
import importlib.util
import numpy as np

from bempp_audio.fea.materials import ShellMaterial

if TYPE_CHECKING:  # pragma: no cover
    from dolfinx.mesh import Mesh, MeshTags  # noqa: F401


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


_DOLFINX_AVAILABLE = _has_module("dolfinx")
_SLEPC_AVAILABLE = _has_module("slepc4py")

dolfinx = None
fem = None
mesh = None
io = None
ufl = None
MPI = None
PETSc = None
SLEPc = None


def _require_dolfinx():
    global dolfinx, fem, mesh, io, ufl, MPI, PETSc
    if not _DOLFINX_AVAILABLE:
        raise ImportError("dolfinx is required (install python3-dolfinx-complex).")
    if dolfinx is not None:
        return
    import dolfinx as _dolfinx
    from dolfinx import fem as _fem, mesh as _mesh, io as _io
    import ufl as _ufl
    from mpi4py import MPI as _MPI
    from petsc4py import PETSc as _PETSc

    dolfinx, fem, mesh, io, ufl, MPI, PETSc = _dolfinx, _fem, _mesh, _io, _ufl, _MPI, _PETSc


def _require_slepc():
    global SLEPc
    _require_dolfinx()
    if not _SLEPC_AVAILABLE:
        raise ImportError("slepc4py is required for modal analysis (install python3-slepc4py-64-complex).")
    if SLEPc is not None:
        return
    from slepc4py import SLEPc as _SLEPc

    SLEPc = _SLEPc


@dataclass(frozen=True)
class RayleighDamping:
    """Rayleigh damping: C = alpha*M + beta*K."""

    alpha: float = 0.0  # mass-proportional
    beta: float = 0.0  # stiffness-proportional


@dataclass(frozen=True)
class ModalResult:
    """Modal analysis result."""

    eigenvalues: np.ndarray  # λ = ω^2
    frequencies_hz: np.ndarray  # f = ω / (2π)
    modes: list  # list[dolfinx.fem.Function] on the mixed space


@dataclass(frozen=True)
class HarmonicResult:
    """Frequency response result container."""

    frequencies_hz: np.ndarray
    displacement: list  # list[dolfinx.fem.Function] for each frequency


def mindlin_reissner_mixed_space(dfx_mesh: "Mesh", degree: int = 1):
    """
    Create a mixed function space for Mindlin–Reissner plate:
      u ∈ (CG)^2, w ∈ CG, θ ∈ (CG)^2.
    """
    _require_dolfinx()
    if int(dfx_mesh.geometry.dim) != 2:
        raise NotImplementedError("MindlinReissnerPlate currently expects a flat 2D mesh (gdim=2).")

    Ve_u = ufl.VectorElement("Lagrange", dfx_mesh.ufl_cell(), degree, dim=2)
    Ve_w = ufl.FiniteElement("Lagrange", dfx_mesh.ufl_cell(), degree)
    Ve_th = ufl.VectorElement("Lagrange", dfx_mesh.ufl_cell(), degree, dim=2)
    W = fem.functionspace(dfx_mesh, ufl.MixedElement([Ve_u, Ve_w, Ve_th]))
    return W


def _iso2d_energy_tensor(E: float, nu: float) -> tuple[float, float]:
    """
    2D isotropic plate constitutive factors for plane stress resultants.
    Returns (prefactor, nu).
    """
    pref = E / (1.0 - nu**2)
    return pref, nu


def _symgrad(v):
    return ufl.sym(ufl.grad(v))


def _inner_iso2d(A: float, nu: float, eps_a, eps_b):
    """
    Inner product for isotropic 2D strain tensors (plane stress form):
      A * [ (1-ν) <ε_a, ε_b> + ν tr(ε_a) tr(ε_b) ]
    """
    return A * ((1.0 - nu) * ufl.inner(eps_a, eps_b) + nu * ufl.tr(eps_a) * ufl.tr(eps_b))


class MindlinReissnerPlateSolver:
    """
    Flat Mindlin–Reissner plate solver (modal + harmonic response).

    Primary DOFs per node:
      u = (u, v)  in-plane displacements
      w           transverse displacement
      θ = (θx, θy) rotations

    Uses selective reduced integration for the transverse shear term to mitigate
    shear locking.
    """

    def __init__(
        self,
        dfx_mesh: "Mesh",
        *,
        material: ShellMaterial,
        degree: int = 1,
        shear_correction: float = 5.0 / 6.0,
        shear_quadrature_degree: int = 1,
    ) -> None:
        _require_dolfinx()
        if int(dfx_mesh.geometry.dim) != 2:
            raise NotImplementedError("MindlinReissnerPlateSolver currently expects a flat 2D mesh (gdim=2).")

        self.mesh = dfx_mesh
        self.material = material
        self.degree = int(degree)
        self.shear_correction = float(shear_correction)
        self.shear_quadrature_degree = int(shear_quadrature_degree)

        self.W = mindlin_reissner_mixed_space(dfx_mesh, degree=self.degree)
        self._facet_tags: Optional["MeshTags"] = None
        self._bcs = []

    def set_facet_tags(self, facet_tags: "MeshTags") -> None:
        self._facet_tags = facet_tags

    def add_clamped_bc(self, marker: int) -> None:
        """
        Add clamped BC (u=v=w=θx=θy=0) on facets with the given marker.
        Requires facet tags to be set via `set_facet_tags`.
        """
        if self._facet_tags is None:
            raise ValueError("Facet tags not set. Call set_facet_tags(facet_tags) first.")

        facets = self._facet_tags.find(int(marker))
        tdim = self.mesh.topology.dim

        # Mixed space subspaces: 0=u(2), 1=w(1), 2=theta(2)
        Vu = self.W.sub(0)
        Vw = self.W.sub(1)
        Vt = self.W.sub(2)

        dofs_u = fem.locate_dofs_topological(Vu, tdim - 1, facets)
        dofs_w = fem.locate_dofs_topological(Vw, tdim - 1, facets)
        dofs_t = fem.locate_dofs_topological(Vt, tdim - 1, facets)

        zero_u = np.array([0.0, 0.0], dtype=np.float64)
        zero_t = np.array([0.0, 0.0], dtype=np.float64)
        bc_u = fem.dirichletbc(zero_u, dofs_u, Vu)
        bc_w = fem.dirichletbc(np.array(0.0, dtype=np.float64), dofs_w, Vw)
        bc_t = fem.dirichletbc(zero_t, dofs_t, Vt)
        self._bcs.extend([bc_u, bc_w, bc_t])

    def forms(self):
        """
        Build bilinear stiffness and mass forms (K, M).
        """
        E = float(self.material.youngs_modulus_pa)
        nu = float(self.material.poissons_ratio)
        rho = float(self.material.density_kg_m3)
        h = float(self.material.thickness_m)
        G = E / (2.0 * (1.0 + nu))

        # Membrane/bending stiffness resultants (plane stress).
        A0, _ = _iso2d_energy_tensor(E, nu)
        A = A0 * h
        D = A0 * (h**3) / 12.0

        (u, w, th) = ufl.TrialFunctions(self.W)
        (v, q, psi) = ufl.TestFunctions(self.W)

        eps_u = _symgrad(u)
        eps_v = _symgrad(v)

        kappa_th = _symgrad(th)
        kappa_psi = _symgrad(psi)

        # Shear strain: gamma = grad(w) - theta
        gamma_u = ufl.grad(w) - th
        gamma_v = ufl.grad(q) - psi

        dx = ufl.Measure("dx", domain=self.mesh)
        dx_s = ufl.Measure("dx", domain=self.mesh, metadata={"quadrature_degree": self.shear_quadrature_degree})

        # Stiffness form
        a_mem = _inner_iso2d(A, nu, eps_u, eps_v) * dx
        a_ben = _inner_iso2d(D, nu, kappa_th, kappa_psi) * dx
        a_shr = (self.shear_correction * G * h) * ufl.inner(gamma_u, gamma_v) * dx_s
        a = a_mem + a_ben + a_shr

        # Mass form (translational + rotational inertia)
        m_trans = (rho * h) * (ufl.inner(u, v) + w * q) * dx
        m_rot = (rho * (h**3) / 12.0) * ufl.inner(th, psi) * dx
        m = m_trans + m_rot

        return fem.form(a), fem.form(m)

    def assemble_matrices(self):
        """Assemble PETSc stiffness and mass matrices."""
        a_form, m_form = self.forms()
        K = fem.petsc.assemble_matrix(a_form, bcs=self._bcs)
        K.assemble()
        M = fem.petsc.assemble_matrix(m_form, bcs=self._bcs)
        M.assemble()
        return K, M

    def solve_modes(
        self,
        *,
        num_modes: int = 10,
        target: str = "smallest",
        tol: float = 1e-9,
        max_it: int = 200,
    ) -> ModalResult:
        """
        Solve generalized eigenproblem: K x = λ M x with SLEPc.

        Parameters
        ----------
        num_modes : int
            Number of eigenmodes to compute.
        target : str
            "smallest" or "largest".
        """
        _require_slepc()
        K, M = self.assemble_matrices()

        eps = SLEPc.EPS().create(self.mesh.comm)
        eps.setOperators(K, M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setDimensions(int(num_modes), PETSc.DECIDE)
        eps.setTolerances(tol, max_it)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)

        if target == "largest":
            eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        else:
            eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)

        eps.solve()
        nconv = eps.getConverged()
        n = min(int(num_modes), int(nconv))

        eigenvalues = np.zeros(n, dtype=float)
        freqs = np.zeros(n, dtype=float)
        modes = []

        vr, _ = K.createVecs()
        for i in range(n):
            lam = eps.getEigenpair(i, vr)
            lam_r = float(np.real(lam))
            eigenvalues[i] = lam_r
            omega = np.sqrt(max(lam_r, 0.0))
            freqs[i] = omega / (2.0 * np.pi)

            fn = fem.Function(self.W, name=f"mode_{i}")
            fn.x.petsc_vec.array[:] = vr.array_r  # type: ignore[attr-defined]
            fn.x.scatter_forward()
            modes.append(fn)

        return ModalResult(eigenvalues=eigenvalues, frequencies_hz=freqs, modes=modes)

    def solve_harmonic(
        self,
        frequencies_hz: Sequence[float],
        *,
        load_marker: int,
        load_amplitude: float,
        damping: Optional[RayleighDamping] = None,
        ksp_type: str = "preonly",
        pc_type: str = "lu",
    ) -> HarmonicResult:
        """
        Harmonic frequency response:
          (K + i ω C - ω^2 M) x = F

        Load is applied as a transverse pressure-like force on the w test function:
          ∫ q_load * q_test ds(load_marker)
        """
        _require_dolfinx()
        if self._facet_tags is None:
            raise ValueError("Facet tags not set. Call set_facet_tags(facet_tags) first.")

        damping = damping or RayleighDamping()
        a_form, m_form = self.forms()
        K = fem.petsc.assemble_matrix(a_form, bcs=self._bcs)
        K.assemble()
        M = fem.petsc.assemble_matrix(m_form, bcs=self._bcs)
        M.assemble()

        # Assemble load vector template (depends only on marker and amplitude).
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self._facet_tags)
        (v, q, psi) = ufl.TestFunctions(self.W)
        L = fem.form((float(load_amplitude) * q) * ds(int(load_marker)))
        b0 = fem.petsc.assemble_vector(L)
        fem.petsc.apply_lifting(b0, [a_form], [self._bcs])
        b0.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b0, self._bcs)

        # Convert K, M to Mat; build solver once and reuse for each frequency by changing A entries.
        results = []
        freqs = np.asarray(list(frequencies_hz), dtype=float)

        C = None
        if damping.alpha != 0.0 or damping.beta != 0.0:
            C = PETSc.Mat().createAIJ(size=K.getSize(), comm=self.mesh.comm)
            C.setUp()
            C.axpy(damping.alpha, M, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
            C.axpy(damping.beta, K, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
            C.assemble()

        ksp = PETSc.KSP().create(self.mesh.comm)
        ksp.setType(ksp_type)
        ksp.getPC().setType(pc_type)

        x = K.createVecRight()

        for f_hz in freqs:
            omega = 2.0 * np.pi * float(f_hz)
            # A = K - ω² M + i ω C
            A = PETSc.Mat().createAIJ(size=K.getSize(), comm=self.mesh.comm)
            A.setUp()
            A.axpy(1.0, K, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
            A.axpy(-(omega**2), M, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
            if C is not None:
                # Complex scalar; ensure complex mat
                A = A.convert("aij")  # no-op if already
                A.axpy(1j * omega, C, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
            A.assemble()

            ksp.setOperators(A)
            b = b0.copy()
            ksp.solve(b, x)

            sol = fem.Function(self.W, dtype=np.complex128, name=f"u_h_{f_hz:.1f}Hz")
            sol.x.petsc_vec.array[:] = x.array  # type: ignore[attr-defined]
            sol.x.scatter_forward()
            results.append(sol)

        return HarmonicResult(frequencies_hz=freqs, displacement=results)

    def write_modes_xdmf(self, filename: str, modal: ModalResult) -> None:
        _require_dolfinx()
        with io.XDMFFile(self.mesh.comm, filename, "w") as f:
            f.write_mesh(self.mesh)
            for mode in modal.modes:
                f.write_function(mode)

