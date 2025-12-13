"""
Solve orchestration for the `Loudspeaker` fluent API.

This module centralizes the heavy orchestration logic (validation, solver
selection, parallel execution, post-processing), keeping `api/loudspeaker.py`
smaller and focused on the public façade.
"""

from __future__ import annotations

from typing import Optional, Callable
import numpy as np

from bempp_audio._optional import require_bempp
from bempp_audio.progress import get_logger
from bempp_audio.velocity import VelocityProfile
from bempp_audio.results import FrequencyResponse, RadiationResult
from bempp_audio.solver import RadiationSolver, OSRCRadiationSolver
from bempp_audio.solver.base import solve_frequencies_parallel
from bempp_audio.api.driver_coupling import solve_with_compression_driver_network
from bempp_audio.baffles import InfiniteBaffle, CircularBaffle


def solve_request(
    request: "SimulationRequest",
    *,
    n_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
    show_progress: Optional[bool] = None,
) -> FrequencyResponse:
    """
    Solve an immutable `SimulationRequest`.

    This is the preferred execution entrypoint for code that wants to separate
    builder configuration from execution.
    """
    logger = get_logger()
    logger.section("BEM Solve")

    mesh = request.mesh
    velocity = request.velocity
    frequencies = np.asarray(request.frequencies, dtype=float)
    reference = request.reference

    exec_cfg = request.execution_config
    if n_workers is None and exec_cfg is not None and getattr(exec_cfg, "n_workers", None) is not None:
        n_workers = int(exec_cfg.n_workers)
    if show_progress is None and exec_cfg is not None and getattr(exec_cfg, "show_progress", None) is not None:
        show_progress = bool(exec_cfg.show_progress)
    if show_progress is None:
        show_progress = True

    with logger.step("Summarize configuration"):
        logger.info(f"Mesh elements: {mesh.grid.number_of_elements}")
        logger.info(f"Frequencies: {len(frequencies)} ({frequencies[0]:.0f}–{frequencies[-1]:.0f} Hz)")
        logger.info(f"Medium: c={request.c:.1f} m/s, rho={request.rho:.3f} kg/m³")
        logger.info(
            f"Solver: tol={request.solver_options.tol:g}, maxiter={request.solver_options.maxiter}, "
            f"space={request.solver_options.space_type}{request.solver_options.space_order}, "
            f"fmm={'on' if request.solver_options.use_fmm else 'off'}"
        )

    with logger.step("Prepare boundary conditions"):
        if isinstance(request.baffle, CircularBaffle):
            unique_domains = np.unique(mesh.grid.domain_indices)
            if len(unique_domains) != 1:
                raise ValueError(
                    "Circular baffle cannot be combined with a multi-domain mesh. "
                    "Use `waveguide_on_box(...)` for baffled waveguide meshes, "
                    "or solve without a finite baffle."
                )
            mesh = mesh.with_baffle(request.baffle.radius, element_size=request.baffle.element_size)
            domains = np.unique(mesh.grid.domain_indices)
            radiator_domain = int(unique_domains[0])
            baffle_domain = int(domains[0] if int(domains[1]) == radiator_domain else domains[1])
            velocity = VelocityProfile.by_domain({radiator_domain: velocity, baffle_domain: VelocityProfile.zero()})

    with logger.step("Construct solver"):
        if request.use_osrc:
            logger.info(f"Preconditioner: OSRC (npade={request.osrc_npade})")
            solver = OSRCRadiationSolver(
                mesh,
                request.c,
                request.rho,
                npade=request.osrc_npade,
                options=request.solver_options,
            )
        else:
            logger.info("Preconditioner: none")
            solver = RadiationSolver(mesh, request.c, request.rho, options=request.solver_options)

    if request.driver_network is not None and request.driver_excitation is not None:
        throat_domain = int(getattr(request.waveguide, "throat_domain", 1)) if request.waveguide is not None else 1

        return solve_with_compression_driver_network(
            solver=solver,
            frequencies=frequencies,
            driver_network=request.driver_network,
            excitation=request.driver_excitation,
            throat_domain=throat_domain,
            baffle_mode=request.baffle,
            progress_callback=progress_callback,
        )

    with logger.step("Solve frequency sweep"):
        if n_workers == 1:
            logger.info("Execution: sequential")
            response = solver.solve_frequencies(
                frequencies,
                velocity,
                progress_callback=progress_callback,
                show_progress=show_progress,
            )
        else:
            logger.info(f"Execution: parallel (n_workers={'auto' if n_workers is None else n_workers})")
            response = solve_frequencies_parallel(
                solver,
                frequencies,
                velocity,
                n_workers=n_workers,
                progress_callback=progress_callback,
                show_progress=show_progress,
            )

    for result in response.results:
        result.reference = reference

    if request.waveguide is not None:
        response.set_waveguide_metadata(request.waveguide)

    if isinstance(request.baffle, InfiniteBaffle):
        with logger.step("Post-process"):
            logger.info(f"Infinite baffle scaling: ×{request.baffle.pressure_scale:g}")
            bempp = require_bempp()

            for result in response.results:
                result.baffle = request.baffle
                result.baffle_plane_z = float(request.baffle.plane_z)
                scaled_coeffs = result.surface_pressure.coefficients * float(request.baffle.pressure_scale)
                result.surface_pressure = bempp.GridFunction(
                    result.surface_pressure.space,
                    coefficients=scaled_coeffs,
                )

    logger.success("Solve complete")
    return response


def solve(
    speaker: "Loudspeaker",
    n_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
    show_progress: Optional[bool] = None,
    strict: bool = False,
) -> FrequencyResponse:
    request = speaker.build(strict=strict)
    return solve_request(
        request,
        n_workers=n_workers,
        progress_callback=progress_callback,
        show_progress=show_progress,
    )


def solve_single(speaker: "Loudspeaker", frequency: float) -> RadiationResult:
    # Avoid mutating the original fluent builder.
    tmp = speaker.clone().single_frequency(frequency)
    response = solve(tmp, strict=False)
    return response[0]
