"""
OSRC-preconditioned radiation solver for high-frequency problems.

The On-Surface Radiation Condition (OSRC) provides excellent preconditioning
for high-frequency acoustic problems, significantly reducing iteration counts
in the GMRES solver.
"""

from __future__ import annotations

from typing import Optional, Tuple
from dataclasses import replace
import numpy as np

from scipy.sparse.linalg import gmres as scipy_gmres

from bempp_audio._optional import optional_import

bempp, BEMPP_AVAILABLE = optional_import("bempp_cl.api")

from bempp_audio.solver.base import RadiationSolver, SolverOptions
from bempp_audio.mesh import LoudspeakerMesh
from bempp_audio.velocity import VelocityProfile


class OSRCRadiationSolver(RadiationSolver):
    """
    High-frequency optimized solver using OSRC preconditioning.

    The OSRC formulation uses a Padé approximation of the Dirichlet-to-Neumann
    map, providing better conditioning at high frequencies where standard
    Burton-Miller can require many iterations.

    The preconditioned equation is:
        (½I - K - NtD·W) · φ = f

    where:
        - K is the double layer operator
        - W is the hypersingular operator
        - NtD is the OSRC Neumann-to-Dirichlet operator

    Parameters
    ----------
    mesh : LoudspeakerMesh
        The mesh representing the radiating surface.
    c : float
        Speed of sound in m/s.
    rho : float
        Air density in kg/m³.
    npade : int
        Order of the Padé approximation (default 2).
    osrc_damping : complex, optional
        Damped wavenumber for OSRC. If None, uses real wavenumber.
    options : SolverOptions, optional
        Additional solver options.

    Notes
    -----
    OSRC preconditioning requires P1 (continuous linear) function spaces.
    This solver will override the space_type to "P" and space_order to 1.

    Best suited for:
    - Frequencies above ~1 kHz for typical loudspeaker sizes
    - Meshes with > 1000 elements
    - Problems where standard solver converges slowly

    References
    ----------
    Antoine & Darbas, "Generalized combined field integral equations for
    the iterative solution of the three-dimensional Helmholtz equation",
    M2AN (2007).
    """

    def __init__(
        self,
        mesh: LoudspeakerMesh,
        c: float = 343.0,
        rho: float = 1.225,
        npade: int = 2,
        osrc_damping: Optional[complex] = None,
        options: Optional[SolverOptions] = None,
    ):
        # Force P1 space for OSRC without mutating caller-provided options.
        if options is None:
            options = SolverOptions(space_type="P", space_order=1)
        else:
            options = replace(options, space_type="P", space_order=1)

        super().__init__(mesh, c, rho, options)
        self.npade = npade
        self.osrc_damping = osrc_damping

    def _get_operators(self, k: complex) -> Tuple:
        """
        Get or create OSRC-preconditioned operators.

        The OSRC formulation: (½I - K - NtD·W)
        """
        cache_key = complex(k)
        if cache_key in self._operator_cache:
            bundle = self._operator_cache[cache_key]
            if isinstance(bundle, dict):
                return bundle["lhs"], bundle["space"]
            return bundle

        # P1 space required for OSRC
        space = bempp.function_space(self.mesh.grid, "P", 1)

        # Assembler
        assembler = "fmm" if self.options.use_fmm else "default_nonlocal"

        # Standard operators
        identity = bempp.operators.boundary.sparse.identity(space, space, space)
        dlp = bempp.operators.boundary.helmholtz.double_layer(
            space, space, space, k, assembler=assembler
        )
        adlp = bempp.operators.boundary.helmholtz.adjoint_double_layer(
            space, space, space, k, assembler=assembler
        )
        slp = bempp.operators.boundary.helmholtz.single_layer(
            space, space, space, k, assembler=assembler
        )
        hyp = bempp.operators.boundary.helmholtz.hypersingular(
            space, space, space, k, assembler=assembler
        )

        # OSRC Neumann-to-Dirichlet operator
        damped_k = self.osrc_damping if self.osrc_damping is not None else k
        ntd = bempp.operators.boundary.helmholtz.osrc_ntd(
            space, k, npade=self.npade, damped_wavenumber=damped_k
        )

        # OSRC-preconditioned operator: ½I - K - NtD·W
        lhs = 0.5 * identity - dlp - ntd * hyp

        self._operator_cache[cache_key] = {
            "lhs": lhs,
            "space": space,
            "assembler": assembler,
            "identity": identity,
            "dlp": dlp,
            "adlp": adlp,
            "slp": slp,
            "hyp": hyp,
            "ntd": ntd,
        }
        return lhs, space

    def _create_rhs(
        self,
        velocity: VelocityProfile,
        space,
        k: complex,
        omega: float,
    ):
        """
        Create RHS grid function for the OSRC-preconditioned formulation.

        With Neumann data g = ∂p/∂n = -iωρ v_n, subtracting a Neumann-to-Dirichlet
        preconditioned hypersingular equation from the direct equation yields:

            (½I - K - NtD·W) p = -(V + NtD(½I + K')) g
        """
        cache_key = complex(k)
        bundle = self._operator_cache.get(cache_key)
        if not isinstance(bundle, dict) or bundle.get("space") is not space:
            self._get_operators(k)
            bundle = self._operator_cache.get(cache_key)

        velocity_gf = velocity.to_grid_function(space)
        g_coeffs = (-1j * omega * float(self.rho)) * velocity_gf.coefficients
        neumann_gf = bempp.GridFunction(space, coefficients=g_coeffs)

        rhs_op = -(
            bundle["slp"] + bundle["ntd"] * (0.5 * bundle["identity"] + bundle["adlp"])
        )
        return rhs_op * neumann_gf

    def __repr__(self) -> str:
        return (
            f"OSRCRadiationSolver(mesh={self.mesh}, c={self.c}, "
            f"npade={self.npade})"
        )


class AdaptiveSolver:
    """
    Automatically selects the best solver based on frequency and mesh.

    Uses standard Burton-Miller for low frequencies and OSRC for high
    frequencies, with automatic crossover determination.

    Parameters
    ----------
    mesh : LoudspeakerMesh
        The radiating surface mesh.
    c : float
        Speed of sound.
    rho : float
        Air density.
    crossover_ka : float
        Crossover ka value (k * characteristic_length). Default 5.0.

    Examples
    --------
    >>> mesh = LoudspeakerMesh.circular_piston(radius=0.05)
    >>> solver = AdaptiveSolver(mesh)
    >>> # Automatically uses Burton-Miller at low freq, OSRC at high freq
    >>> result = solver.solve(frequency=10000, velocity=piston_vel)
    """

    def __init__(
        self,
        mesh: LoudspeakerMesh,
        c: float = 343.0,
        rho: float = 1.225,
        crossover_ka: float = 5.0,
    ):
        self.mesh = mesh
        self.c = c
        self.rho = rho
        self.crossover_ka = crossover_ka

        # Estimate characteristic length from mesh
        info = mesh.info()
        self.char_length = (
            info.bounding_box[1] - info.bounding_box[0]
        ).max() / 2

        # Create both solvers
        self._burton_miller = RadiationSolver(mesh, c, rho)
        self._osrc = OSRCRadiationSolver(mesh, c, rho)

    def solve(
        self,
        frequency: float,
        velocity: VelocityProfile,
    ) -> "RadiationResult":
        """
        Solve using the appropriate solver for the frequency.
        """
        k = 2 * np.pi * frequency / self.c
        ka = k * self.char_length

        if ka < self.crossover_ka:
            return self._burton_miller.solve(frequency, velocity)
        else:
            return self._osrc.solve(frequency, velocity)

    def solve_frequencies(
        self,
        frequencies: np.ndarray,
        velocity: VelocityProfile,
        progress_callback=None,
    ) -> "FrequencyResponse":
        """
        Solve for multiple frequencies using adaptive solver selection.
        """
        from bempp_audio.results import FrequencyResponse

        response = FrequencyResponse()

        for i, freq in enumerate(frequencies):
            result = self.solve(freq, velocity)
            response.add(result)

            if progress_callback:
                progress_callback(freq, i + 1, len(frequencies))

        return response

    def __repr__(self) -> str:
        return (
            f"AdaptiveSolver(mesh={self.mesh}, crossover_ka={self.crossover_ka})"
        )
