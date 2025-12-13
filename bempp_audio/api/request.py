"""
Immutable "built request" object for the bempp_audio fluent API.

This provides a separation between:
- configuration/builder (`Loudspeaker`)
- execution (`SimulationRequest.solve()`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

from bempp_audio.acoustic_reference import AcousticReference
from bempp_audio.baffles import Baffle
from bempp_audio.mesh import LoudspeakerMesh
from bempp_audio.solver.base import SolverOptions
from bempp_audio.velocity import VelocityProfile


@dataclass(frozen=True)
class SimulationRequest:
    """An immutable, ready-to-solve configuration snapshot."""

    mesh: LoudspeakerMesh
    velocity: VelocityProfile
    frequencies: np.ndarray
    c: float
    rho: float
    baffle: Baffle
    solver_options: SolverOptions
    use_osrc: bool
    osrc_npade: int
    measurement_distance: float
    reference: AcousticReference

    # Optional waveguide/driver context for downstream plotting and coupling
    waveguide: Optional[object] = None  # WaveguideMetadata
    driver_network: Optional[object] = None  # CompressionDriverNetwork
    driver_excitation: Optional[object] = None  # CompressionDriverExcitation

    # Execution defaults (optional)
    execution_config: Optional[object] = None  # ExecutionConfig

    def solve(
        self,
        *,
        n_workers: Optional[int] = None,
        show_progress: Optional[bool] = None,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
    ):
        """Solve this request and return a `FrequencyResponse`."""
        from bempp_audio.api.solve import solve_request

        return solve_request(
            self,
            n_workers=n_workers,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )

