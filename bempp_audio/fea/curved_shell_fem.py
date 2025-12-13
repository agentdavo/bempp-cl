"""
Curved Mindlin–Reissner shell FEM utilities (DOLFINx) for 2D manifold surfaces.

This module provides a practical *spherical-cap* Mindlin–Reissner shell model
intended for Benchmark3TiConfig_v1 validation (Phase 7.2).

Scope / Caveats
---------------
* This is not a fully general arbitrary-curvature shell formulation.
* Curvature is modeled as constant (1/R) in the local tangent basis, which is
  appropriate for spherical caps and a good first step for deep-drawn domes.
* Imports of dolfinx/petsc4py/slepc4py are done lazily to avoid hard MPI init
  crashes on misconfigured systems (WSL2 quirks, etc).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import importlib.util
import numpy as np

from bempp_audio.fea.materials import ShellMaterial

if TYPE_CHECKING:  # pragma: no cover
    from dolfinx.mesh import Mesh  # noqa: F401


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


_DOLFINX_AVAILABLE = _has_module("dolfinx")
_SLEPC_AVAILABLE = _has_module("slepc4py")

dolfinx = None
fem = None
io = None
ufl = None
MPI = None
PETSc = None
SLEPc = None


def _require_dolfinx():
    global dolfinx, fem, io, ufl, MPI, PETSc
    if not _DOLFINX_AVAILABLE:
        raise ImportError("dolfinx is required (install python3-dolfinx-complex).")
    if dolfinx is not None:
        return
    import dolfinx as _dolfinx
    from dolfinx import fem as _fem, io as _io
    import ufl as _ufl
    from mpi4py import MPI as _MPI
    from petsc4py import PETSc as _PETSc

    dolfinx, fem, io, ufl, MPI, PETSc = _dolfinx, _fem, _io, _ufl, _MPI, _PETSc


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
class ModalResult:
    eigenvalues: np.ndarray  # λ = ω^2
    frequencies_hz: np.ndarray  # f = ω / (2π)
    modes: list  # list[dolfinx.fem.Function]


def _safe_normalize(v, eps: float = 1e-12):
    n2 = ufl.dot(v, v)
    return ufl.conditional(ufl.lt(n2, eps * eps), ufl.as_vector((0.0, 0.0, 0.0)), v / ufl.sqrt(n2))

def _frame_from_cell_normal(n):
    ez = ufl.as_vector((0.0, 0.0, 1.0))
    ey = ufl.as_vector((0.0, 1.0, 0.0))
    t1a = ufl.cross(ez, n)
    t1a_n2 = ufl.dot(t1a, t1a)
    t1b = ufl.cross(ey, n)
    t1_raw = ufl.conditional(ufl.lt(t1a_n2, 1e-12), t1b, t1a)
    t1 = _safe_normalize(t1_raw)
    t2 = _safe_normalize(ufl.cross(n, t1))
    return t1, t2


def _d_tangent(scalar_field, t1, t2):
    g = ufl.grad(scalar_field)  # tangential gradient on manifold meshes (gdim=3)
    return ufl.as_vector((ufl.dot(g, t1), ufl.dot(g, t2)))


def _sym2(a11, a12, a22):
    return ufl.as_matrix(((a11, a12), (a12, a22)))


class SphericalCapMindlinReissnerShellSolver:
    """
    Spherical-cap Mindlin–Reissner shell solver on a 2D manifold mesh (gdim=3).

    Unknowns are expressed in the local tangent basis (t1,t2,n):
      u = (u1,u2)  in-plane components along (t1,t2)
      w            normal displacement along n
      θ = (th1,th2) rotations about (t1,t2)
    Total = 5 DOF per node.
    """

    def __init__(
        self,
        dfx_mesh: "Mesh",
        *,
        sphere_radius_m: float,
        material: ShellMaterial,
        degree: int = 1,
        shear_correction: float = 5.0 / 6.0,
        shear_quadrature_degree: int = 1,
    ) -> None:
        _require_dolfinx()
        if int(dfx_mesh.topology.dim) != 2 or int(dfx_mesh.geometry.dim) != 3:
            raise NotImplementedError("This solver expects a 2D surface mesh embedded in 3D (tdim=2, gdim=3).")
        if float(sphere_radius_m) <= 0:
            raise ValueError("sphere_radius_m must be positive.")

        self.mesh = dfx_mesh
        self.sphere_radius_m = float(sphere_radius_m)
        self.material = material
        self.degree = int(degree)
        self.shear_correction = float(shear_correction)
        self.shear_quadrature_degree = int(shear_quadrature_degree)

        Ve_u = ufl.VectorElement("Lagrange", dfx_mesh.ufl_cell(), self.degree, dim=2)
        Ve_w = ufl.FiniteElement("Lagrange", dfx_mesh.ufl_cell(), self.degree)
        Ve_th = ufl.VectorElement("Lagrange", dfx_mesh.ufl_cell(), self.degree, dim=2)
        self.W = fem.functionspace(dfx_mesh, ufl.MixedElement([Ve_u, Ve_w, Ve_th]))
        self._bcs = []

    def add_clamped_annulus(self, *, r_inner_m: float, r_outer_m: float) -> None:
        """
        Clamp all DOFs for vertices within r ∈ [r_inner, r_outer] (in xy-plane).
        """
        r0 = float(r_inner_m)
        r1 = float(r_outer_m)
        if r0 < 0 or r1 <= 0 or r1 < r0:
            raise ValueError("Invalid annulus radii.")

        Vu = self.W.sub(0)
        Vw = self.W.sub(1)
        Vt = self.W.sub(2)

        def _in_band(x):
            r = np.sqrt(x[0] ** 2 + x[1] ** 2)
            return (r >= r0) & (r <= r1)

        dofs_u = fem.locate_dofs_geometrical(Vu, _in_band)
        dofs_w = fem.locate_dofs_geometrical(Vw, _in_band)
        dofs_t = fem.locate_dofs_geometrical(Vt, _in_band)

        zero_u = np.array([0.0, 0.0], dtype=np.float64)
        zero_t = np.array([0.0, 0.0], dtype=np.float64)
        self._bcs.extend(
            [
                fem.dirichletbc(zero_u, dofs_u, Vu),
                fem.dirichletbc(np.array(0.0, dtype=np.float64), dofs_w, Vw),
                fem.dirichletbc(zero_t, dofs_t, Vt),
            ]
        )

    def forms(self):
        """
        Build stiffness and mass forms (K, M) for spherical-cap MR shell.
        """
        E = float(self.material.youngs_modulus_pa)
        nu = float(self.material.poissons_ratio)
        rho = float(self.material.density_kg_m3)
        h = float(self.material.thickness_m)
        G = E / (2.0 * (1.0 + nu))

        # Plane-stress resultants
        A0 = E / (1.0 - nu**2)
        A = A0 * h
        D = A0 * (h**3) / 12.0
        kappa = float(self.shear_correction)
        k_s = kappa * G * h

        # Unknowns
        u_vec, w, th = ufl.TrialFunctions(self.W)
        v_vec, q, ps = ufl.TestFunctions(self.W)

        # Local frame
        n = ufl.CellNormal(self.mesh)
        t1, t2 = _frame_from_cell_normal(n)

        # Derivatives in tangent basis
        du1 = _d_tangent(u_vec[0], t1, t2)
        du2 = _d_tangent(u_vec[1], t1, t2)
        dv1 = _d_tangent(v_vec[0], t1, t2)
        dv2 = _d_tangent(v_vec[1], t1, t2)

        dth1 = _d_tangent(th[0], t1, t2)
        dth2 = _d_tangent(th[1], t1, t2)
        dps1 = _d_tangent(ps[0], t1, t2)
        dps2 = _d_tangent(ps[1], t1, t2)

        dw = _d_tangent(w, t1, t2)
        dq = _d_tangent(q, t1, t2)

        # Membrane strains (2x2) in tangent basis, including spherical curvature coupling:
        # eps = sym(grad_s u) + (w/R) I
        eps_u = _sym2(du1[0], 0.5 * (du1[1] + du2[0]), du2[1])
        eps_v = _sym2(dv1[0], 0.5 * (dv1[1] + dv2[0]), dv2[1])

        curv = (1.0 / float(self.sphere_radius_m))
        eps_u = eps_u + (w * curv) * ufl.Identity(2)
        eps_v = eps_v + (q * curv) * ufl.Identity(2)

        # Bending curvatures (2x2) from rotations: kappa_b = sym(grad_s theta)
        kap_u = _sym2(dth1[0], 0.5 * (dth1[1] + dth2[0]), dth2[1])
        kap_v = _sym2(dps1[0], 0.5 * (dps1[1] + dps2[0]), dps2[1])

        # Transverse shear strains in tangent basis: gamma = theta - grad_s(w)
        gam_u = ufl.as_vector((th[0] - dw[0], th[1] - dw[1]))
        gam_v = ufl.as_vector((ps[0] - dq[0], ps[1] - dq[1]))

        def _inner_iso2d(pref: float, nu_: float, a2, b2):
            return pref * ((1.0 - nu_) * ufl.inner(a2, b2) + nu_ * ufl.tr(a2) * ufl.tr(b2))

        # Stiffness
        dx_shear = ufl.Measure("dx", domain=self.mesh, metadata={"quadrature_degree": int(self.shear_quadrature_degree)})
        K = (
            _inner_iso2d(A, nu, eps_u, eps_v) * ufl.dx
            + _inner_iso2d(D, nu, kap_u, kap_v) * ufl.dx
            + (k_s * ufl.inner(gam_u, gam_v)) * dx_shear
        )

        # Mass (translational + rotary)
        m0 = rho * h
        j0 = rho * (h**3) / 12.0
        M = (
            m0 * (ufl.inner(u_vec, v_vec) + w * q) * ufl.dx
            + j0 * ufl.inner(th, ps) * ufl.dx
        )
        return K, M

    def solve_modes(self, *, n_modes: int = 8, sigma: Optional[float] = None) -> ModalResult:
        """
        Solve generalized eigenproblem K x = λ M x, λ=ω^2.
        """
        _require_slepc()
        K_form, M_form = self.forms()
        K = fem.petsc.assemble_matrix(fem.form(K_form), bcs=self._bcs)
        K.assemble()
        M = fem.petsc.assemble_matrix(fem.form(M_form), bcs=self._bcs)
        M.assemble()

        eps = SLEPc.EPS().create(self.mesh.comm)
        eps.setOperators(K, M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        eps.setDimensions(int(n_modes), PETSc.DECIDE)
        if sigma is not None:
            eps.setTarget(float(sigma))
        eps.setFromOptions()
        eps.solve()

        nconv = int(eps.getConverged())
        n_take = min(int(n_modes), nconv) if nconv > 0 else 0

        eigenvalues = np.zeros(n_take, dtype=float)
        modes = []

        vr, _ = K.createVecs()
        for i in range(n_take):
            lam = eps.getEigenpair(i, vr)
            lam_r = float(np.real(lam))
            eigenvalues[i] = lam_r
            fn = fem.Function(self.W, name=f"mode_{i}")
            fn.x.petsc_vec.array[:] = vr.array_r  # type: ignore[attr-defined]
            fn.x.scatter_forward()
            modes.append(fn)

        omega = np.sqrt(np.maximum(eigenvalues, 0.0))
        freqs = omega / (2.0 * np.pi)
        return ModalResult(eigenvalues=eigenvalues, frequencies_hz=freqs, modes=modes)

    def write_modes_xdmf(self, filename: str, modal: ModalResult) -> None:
        _require_dolfinx()
        with io.XDMFFile(self.mesh.comm, str(filename), "w") as f:
            f.write_mesh(self.mesh)
            for i, mode in enumerate(modal.modes):
                mode.name = f"mode_{i}"
                f.write_function(mode, float(modal.frequencies_hz[i]))
