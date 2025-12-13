"""
Base radiation solver using Burton-Miller formulation.

The Burton-Miller combined field integral equation avoids the spurious
interior resonances that plague the standard direct BEM formulation.
This is essential for frequency sweeps across the audio range.
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass
import numpy as np

from bempp_audio._optional import optional_import

bempp, BEMPP_AVAILABLE = optional_import("bempp_cl.api")

from bempp_audio.mesh import LoudspeakerMesh
from bempp_audio.velocity import VelocityProfile
from bempp_audio.progress import progress, get_logger, ProgressTracker


@dataclass
class SolverOptions:
    """Configuration options for the radiation solver."""
    tol: float = 1e-5  # GMRES tolerance
    maxiter: int = 1000  # Maximum GMRES iterations
    space_type: str = "DP"  # Function space type: "DP" or "P"
    # Burton–Miller uses the hypersingular operator, which in bempp-cl requires
    # discontinuous P1 ("p1_discontinuous") rather than piecewise constants.
    space_order: int = 1  # Function space order
    use_fmm: bool = False  # Use Fast Multipole Method
    fmm_expansion_order: int = 5  # FMM expansion order
    coupling_parameter: Optional[float] = None  # Burton-Miller η (default: k)

    # Optional progress feedback from the iterative solver
    gmres_log_every: int = 0  # 0=off, otherwise log every N iterations
    gmres_log_residuals: bool = True  # include residual norm in logs when enabled


class RadiationSolver:
    """
    Solve Helmholtz radiation problem using Burton-Miller formulation.

    This solver computes the acoustic radiation from a vibrating surface
    using the Boundary Element Method with the Burton-Miller combined
    field integral equation, which avoids interior resonance issues.

    This solver assumes a prescribed normal velocity on the surface, i.e.
    a Neumann boundary condition for the acoustic pressure:

        g := ∂p/∂n = -iωρ v_n

    and solves for the surface pressure (Dirichlet trace) `p` via the
    Burton–Miller combined formulation for the exterior Neumann problem:

        (W + iη(½I - K)) · p = (½I + K' - iηV) · g

    where:
        - V is the single layer operator
        - K is the double layer operator
        - K' is the adjoint double layer operator
        - W is the hypersingular operator
        - η is the coupling parameter (typically = k)
        - p is the surface pressure
        - v_n is the prescribed normal velocity

    Parameters
    ----------
    mesh : LoudspeakerMesh
        The mesh representing the radiating surface.
    c : float
        Speed of sound in m/s. Default 343.0.
    rho : float
        Air density in kg/m³. Default 1.225.
    options : SolverOptions, optional
        Solver configuration options.

    Attributes
    ----------
    mesh : LoudspeakerMesh
        The radiator mesh.
    c : float
        Speed of sound.
    rho : float
        Air density.

    Examples
    --------
    >>> mesh = LoudspeakerMesh.circular_piston(radius=0.05)
    >>> solver = RadiationSolver(mesh)
    >>> velocity = VelocityProfile.piston(amplitude=0.01)
    >>> result = solver.solve(frequency=1000, velocity=velocity)
    >>> print(result.radiated_power())
    """

    def __init__(
        self,
        mesh: LoudspeakerMesh,
        c: float = 343.0,
        rho: float = 1.225,
        options: Optional[SolverOptions] = None,
    ):
        if not BEMPP_AVAILABLE:
            raise ImportError("bempp_cl is required for RadiationSolver")

        self.mesh = mesh
        self.c = c
        self.rho = rho
        self.options = options or SolverOptions()
        self._operator_cache: Dict[complex, Tuple] = {}

    def solve(
        self,
        frequency: float,
        velocity: VelocityProfile,
    ) -> "RadiationResult":
        """
        Solve for surface pressure at a single frequency.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.
        velocity : VelocityProfile
            Normal velocity distribution on the surface.

        Returns
        -------
        RadiationResult
            Solution container with surface pressure and metadata.
        """
        from bempp_audio.results import RadiationResult
        from bempp_audio.acoustic_reference import AcousticReference

        logger = get_logger()
        k = 2 * np.pi * frequency / self.c
        omega = 2 * np.pi * frequency

        # Get or create operators
        logger.debug(f"[{frequency:.0f} Hz] Assembling BEM operators (k={k:.4f})")
        lhs, space = self._get_operators(k)

        # Create RHS from velocity
        logger.debug(f"[{frequency:.0f} Hz] Creating RHS from velocity BC")
        rhs_gf = self._create_rhs(velocity, space, k, omega)

        # Solve the system
        logger.debug(f"[{frequency:.0f} Hz] Solving linear system (GMRES)")
        surface_pressure, iterations, info = self._solve_system(lhs, rhs_gf, frequency)

        logger.debug(f"[{frequency:.0f} Hz] Solution complete")

        return RadiationResult(
            frequency=frequency,
            wavenumber=k,
            surface_pressure=surface_pressure,
            solver_info=info,
            iterations=iterations,
            mesh=self.mesh,
            velocity=velocity,
            c=self.c,
            rho=self.rho,
            reference=AcousticReference.from_mesh(self.mesh),
        )

    def solve_frequencies(
        self,
        frequencies: np.ndarray,
        velocity: VelocityProfile,
        progress_callback=None,
        show_progress: bool = True,
    ) -> "FrequencyResponse":
        """
        Solve for multiple frequencies sequentially.

        Parameters
        ----------
        frequencies : np.ndarray
            Array of frequencies in Hz.
        velocity : VelocityProfile
            Normal velocity distribution.
        progress_callback : callable, optional
            Function called with (freq, index, total) for progress updates.
        show_progress : bool
            If True, display progress bar. Default True.

        Returns
        -------
        FrequencyResponse
            Container with results at all frequencies.
        """
        from bempp_audio.results import FrequencyResponse
        from bempp_audio.progress import log_device_info

        logger = get_logger()
        
        # Log device info on first solve
        if not hasattr(self, '_device_info_logged'):
            logger.blank()
            log_device_info()
            logger.blank()
            self._device_info_logged = True
        
        logger.info(f"Starting frequency sweep: {len(frequencies)} frequencies")

        response = FrequencyResponse()

        with ProgressTracker(
            total=len(frequencies),
            desc="Frequency sweep",
            unit="freq",
            disable=not show_progress
        ) as pbar:
            for i, freq in enumerate(frequencies):
                result = self.solve(freq, velocity)
                response.add(result)
                pbar.update(item=f"{freq:.0f} Hz")

                if progress_callback:
                    progress_callback(freq, i + 1, len(frequencies))

        logger.info("Frequency sweep complete")
        return response

    def _get_operators(self, k: complex) -> Tuple:
        """
        Get or create Burton-Miller operators for given wavenumber.

        Uses caching to avoid re-assembling operators at the same frequency.
        """
        logger = get_logger()

        # Check cache
        cache_key = complex(k)
        if cache_key in self._operator_cache:
            logger.debug(f"Using cached operators for k={k:.4f}")
            bundle = self._operator_cache[cache_key]
            if isinstance(bundle, dict):
                return bundle["lhs"], bundle["space"]
            return bundle

        logger.debug(f"Assembling new operators for k={k:.4f}")

        # Create function space
        space = bempp.function_space(
            self.mesh.grid,
            self.options.space_type,
            self.options.space_order,
        )

        # Assembler type
        assembler = "fmm" if self.options.use_fmm else "default_nonlocal"
        if self.options.use_fmm:
            bempp.GLOBAL_PARAMETERS.fmm.expansion_order = (
                self.options.fmm_expansion_order
            )

        # Create operators
        identity = bempp.operators.boundary.sparse.identity(space, space, space)
        slp = bempp.operators.boundary.helmholtz.single_layer(
            space, space, space, k, assembler=assembler
        )  # V
        dlp = bempp.operators.boundary.helmholtz.double_layer(
            space, space, space, k, assembler=assembler
        )  # K
        adlp = bempp.operators.boundary.helmholtz.adjoint_double_layer(
            space, space, space, k, assembler=assembler
        )  # K'
        hyp = bempp.operators.boundary.helmholtz.hypersingular(
            space, space, space, k, assembler=assembler
        )  # W

        # Burton-Miller coupling parameter
        eta = self.options.coupling_parameter
        if eta is None:
            eta = k  # Default: η = k

        # Burton–Miller operator for the exterior Neumann problem:
        # (W + iη(½I - K)) p = (½I + K' - iηV) g
        lhs = hyp + 1j * eta * (0.5 * identity - dlp)

        # Cache the result
        self._operator_cache[cache_key] = {
            "lhs": lhs,
            "space": space,
            "assembler": assembler,
            "eta": eta,
            "identity": identity,
            "slp": slp,
            "dlp": dlp,
            "adlp": adlp,
            "hyp": hyp,
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
        Create the RHS grid function for the Burton–Miller formulation.

        For prescribed normal velocity `v_n`, the Neumann data is:
            g = ∂p/∂n = -iωρ v_n

        The RHS is the operator application:
            (½I + K' - iηV) g
        """
        logger = get_logger()

        cache_key = complex(k)
        bundle = self._operator_cache.get(cache_key)
        if not isinstance(bundle, dict) or bundle.get("space") is not space:
            logger.debug("Re-assembling RHS operators (cache miss or space mismatch)")
            self._get_operators(k)
            bundle = self._operator_cache.get(cache_key)

        velocity_gf = velocity.to_grid_function(space)
        g_coeffs = (-1j * omega * float(self.rho)) * velocity_gf.coefficients
        neumann_gf = bempp.GridFunction(space, coefficients=g_coeffs)

        eta = bundle["eta"]
        rhs_op = 0.5 * bundle["identity"] + bundle["adlp"] - 1j * eta * bundle["slp"]
        return rhs_op * neumann_gf

    def _solve_system(
        self,
        lhs,
        rhs_gf,
        frequency: float = 0,
    ) -> Tuple:
        """
        Solve the linear system using GMRES.

        Returns
        -------
        tuple
            (solution GridFunction, iteration count, convergence info)
        """
        logger = get_logger()

        # Use bempp-cl's gmres wrapper which handles scipy API compatibility
        from bempp_cl.api.linalg import gmres as bempp_gmres

        log_every = int(getattr(self.options, "gmres_log_every", 0) or 0)
        log_residuals = bool(getattr(self.options, "gmres_log_residuals", True))

        iteration_callback = None
        if log_every > 0:
            def _iter_cb(it: int, res_norm: float) -> None:
                # `bempp_cl` calls this for each iteration; keep our own throttling.
                if it == 1 or (it % log_every) == 0:
                    logger.info(f"[{frequency:.0f} Hz] GMRES iter {it} residual {res_norm:.3e}")

            iteration_callback = _iter_cb

        # Use bempp-cl's gmres wrapper (scipy-compatible) with optional iteration callback.
        if log_every > 0 and log_residuals:
            solution, info, _residuals, iterations = bempp_gmres(
                lhs,
                rhs_gf,
                tol=self.options.tol,
                maxiter=self.options.maxiter,
                use_strong_form=True,
                return_residuals=True,
                return_iteration_count=True,
                log_every=0,  # avoid double-logging from bempp_cl.api.log
                iteration_callback=iteration_callback,
            )
        else:
            solution, info, iterations = bempp_gmres(
                lhs,
                rhs_gf,
                tol=self.options.tol,
                maxiter=self.options.maxiter,
                use_strong_form=True,
                return_iteration_count=True,
                log_every=0 if log_every > 0 else 1,
                iteration_callback=iteration_callback,
            )

        # Log convergence status
        if info == 0:
            logger.debug(f"[{frequency:.0f} Hz] GMRES converged in {iterations} iterations")
        else:
            logger.warning(f"[{frequency:.0f} Hz] GMRES did not converge (info={info})")

        return solution, iterations, info

    def clear_cache(self):
        """Clear the operator cache."""
        self._operator_cache.clear()

    def __repr__(self) -> str:
        return (
            f"RadiationSolver(mesh={self.mesh}, c={self.c}, rho={self.rho}, "
            f"options={self.options})"
        )


# Module-level worker function for parallel execution
# (Must be at module level for pickling)
_parallel_worker_data = {}


def _solve_single_frequency(freq):
    """Worker function to solve at single frequency."""
    data = _parallel_worker_data
    solver = data["solver"]
    vel = data["velocity"]
    result = solver.solve(freq, vel)

    # Return serializable data
    return {
        "frequency": result.frequency,
        "wavenumber": complex(result.wavenumber),
        "pressure_coeffs": result.surface_pressure.coefficients.copy(),
        "solver_info": result.solver_info,
        "iterations": result.iterations,
    }


def _init_worker(mesh_data, velocity_data, c, rho, options_dict, solver_kind: str, solver_params: dict):
    """Initialize worker process with shared data and a reusable solver instance."""
    global _parallel_worker_data
    mesh = LoudspeakerMesh.from_dict(mesh_data)
    vel = VelocityProfile.from_dict(velocity_data)
    opts = SolverOptions(**options_dict)

    if solver_kind == "osrc":
        from bempp_audio.solver.osrc_solver import OSRCRadiationSolver

        local_solver = OSRCRadiationSolver(
            mesh,
            c=c,
            rho=rho,
            npade=int(solver_params.get("npade", 2)),
            osrc_damping=solver_params.get("osrc_damping", None),
            options=opts,
        )
    else:
        local_solver = RadiationSolver(mesh, c, rho, opts)

    _parallel_worker_data = {"solver": local_solver, "velocity": vel}


def solve_frequencies_parallel(
    solver: RadiationSolver,
    frequencies: np.ndarray,
    velocity: VelocityProfile,
    n_workers: Optional[int] = None,
    progress_callback=None,
    show_progress: bool = True,
) -> "FrequencyResponse":
    """
    Parallel frequency sweep using multiprocessing.

    Parameters
    ----------
    solver : RadiationSolver
        Configured solver instance.
    frequencies : np.ndarray
        Array of frequencies to solve.
    velocity : VelocityProfile
        Velocity profile (must be serializable).
    n_workers : int, optional
        Number of parallel workers. Default: min(8, CPUs).
    progress_callback : callable, optional
        Progress callback function.
    show_progress : bool
        If True, display progress bar. Default True.

    Returns
    -------
    FrequencyResponse
        Combined results from all frequencies.
    """
    import os
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from bempp_audio.results import FrequencyResponse
    from bempp_audio.progress import log_device_info
    from bempp_audio.runtime import multiprocessing_start_method, is_wsl

    logger = get_logger()
    
    # Log device info on first parallel solve
    if not hasattr(solver, '_device_info_logged_parallel'):
        logger.blank()
        log_device_info()
        logger.blank()
        solver._device_info_logged_parallel = True

    if n_workers is None:
        n_workers = min(8, os.cpu_count() or 4)
    if n_workers == 1:
        logger.info("Parallel sweep disabled (n_workers=1): running sequential solve")
        return solver.solve_frequencies(
            frequencies,
            velocity,
            progress_callback=progress_callback,
            show_progress=show_progress,
        )

    mp_start = multiprocessing_start_method()
    mp_context = None
    if mp_start:
        try:
            mp_context = mp.get_context(mp_start)
        except Exception:
            mp_context = None
    if mp_context is not None and is_wsl():
        logger.warning(f"WSL detected: using multiprocessing start method '{mp_start}' for stability")

    logger.info(f"Starting parallel frequency sweep: {len(frequencies)} frequencies with {n_workers} workers")

    # Serialize data for workers
    mesh_data = solver.mesh.to_dict()
    velocity_data = velocity.to_dict()
    c = solver.c
    rho = solver.rho
    options_dict = {
        "tol": solver.options.tol,
        "maxiter": solver.options.maxiter,
        "space_type": solver.options.space_type,
        "space_order": solver.options.space_order,
        "use_fmm": solver.options.use_fmm,
        "fmm_expansion_order": solver.options.fmm_expansion_order,
        "coupling_parameter": solver.options.coupling_parameter,
    }

    solver_kind = "standard"
    solver_params: dict = {}
    try:
        from bempp_audio.solver.osrc_solver import OSRCRadiationSolver

        if isinstance(solver, OSRCRadiationSolver):
            solver_kind = "osrc"
            solver_params = {
                "npade": int(getattr(solver, "npade", 2)),
                "osrc_damping": getattr(solver, "osrc_damping", None),
            }
    except Exception:
        solver_kind = "standard"
        solver_params = {}

    results_data = []
    completed = 0

    with ProgressTracker(
        total=len(frequencies),
        desc=f"Parallel sweep ({n_workers} workers)",
        unit="freq",
        disable=not show_progress
    ) as pbar:
        try:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(mesh_data, velocity_data, c, rho, options_dict, solver_kind, solver_params),
                mp_context=mp_context,
            ) as executor:
                futures = {executor.submit(_solve_single_frequency, f): f for f in frequencies}

                for future in as_completed(futures):
                    freq = futures[future]
                    try:
                        result_data = future.result()
                        results_data.append(result_data)
                        completed += 1
                        pbar.update(item=f"{freq:.0f} Hz")
                        if progress_callback:
                            progress_callback(freq, completed, len(frequencies))
                    except Exception as e:
                        logger.error(f"Error at {freq} Hz: {e}")
                        completed += 1
                        pbar.update(item=f"{freq:.0f} Hz (error)")
        except (PermissionError, OSError) as e:
            logger.warning(f"Parallel execution unavailable ({type(e).__name__}: {e}); falling back to sequential solve")
            return solver.solve_frequencies(
                frequencies,
                velocity,
                progress_callback=progress_callback,
                show_progress=show_progress,
            )

    # Sort by frequency
    results_data.sort(key=lambda x: x["frequency"])

    logger.info("Parallel frequency sweep complete")
    logger.info(f"Post-processing: reconstructing {len(results_data)} results...")

    # Reconstruct FrequencyResponse - create mesh and space ONCE (not per frequency)
    response = FrequencyResponse()
    from bempp_audio.results import RadiationResult
    from bempp_audio.acoustic_reference import AcousticReference

    # Create shared mesh and space once
    shared_mesh = LoudspeakerMesh.from_dict(mesh_data)
    shared_space = bempp.function_space(
        shared_mesh.grid,
        options_dict["space_type"],
        options_dict["space_order"],
    )
    shared_velocity = VelocityProfile.from_dict(velocity_data)

    with ProgressTracker(
        total=len(results_data),
        desc="Reconstructing results",
        unit="result",
        disable=not show_progress
    ) as pbar:
        for rd in results_data:
            # Create GridFunction with shared space
            pressure_gf = bempp.GridFunction(shared_space, coefficients=rd["pressure_coeffs"])

            result = RadiationResult(
                frequency=rd["frequency"],
                wavenumber=rd["wavenumber"],
                surface_pressure=pressure_gf,
                solver_info=rd["solver_info"],
                iterations=rd["iterations"],
                mesh=shared_mesh,
                velocity=shared_velocity,
                c=c,
                rho=rho,
                reference=AcousticReference.from_mesh(shared_mesh),
            )
            response.add(result)
            pbar.update(item=f"{rd['frequency']:.0f} Hz")

    logger.info("Post-processing complete")
    n_failed = sum(1 for r in response.results if getattr(r, "solver_info", 1) != 0)
    if n_failed == 0:
        logger.info(f"Frequency sweep finished: {len(response.results)} results, all converged")
    else:
        logger.warning(f"Frequency sweep finished: {len(response.results)} results, {n_failed} non-converged")
    return response
