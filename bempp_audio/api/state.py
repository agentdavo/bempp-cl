"""
Internal state containers for the `Loudspeaker` fluent API.

The public API is intentionally fluent/mutable, but internally we keep a
single state object to reduce the number of loosely-related private fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from bempp_audio.solver.base import SolverOptions
from bempp_audio.baffles import Baffle, FreeSpace
from bempp_audio.acoustic_reference import AcousticReference


@dataclass(frozen=True)
class WaveguideMetadata:
    throat_diameter: float
    mouth_diameter: float
    length: float
    profile: str
    throat_domain: int
    wall_domain: int
    mouth_center: Tuple[float, float, float]
    # CTS profile parameters (optional, only for profile='cts')
    cts_throat_blend: float = 0.0
    cts_transition: float = 0.75
    cts_tangency: float = 1.0
    cts_driver_exit_angle_deg: Optional[float] = None
    cts_throat_angle_deg: Optional[float] = None
    cts_mouth_roll: float = 0.0
    cts_curvature_regularizer: float = 1.0
    cts_mid_curvature: float = 0.0
    # OS profile parameters (optional, only for profile='os'/'oblate_spheroidal')
    os_opening_angle_deg: Optional[float] = None


@dataclass(frozen=True)
class LoudspeakerState:
    mesh: Optional["LoudspeakerMesh"] = None
    baffle: Baffle = field(default_factory=FreeSpace)
    velocity: Optional["VelocityProfile"] = None
    frequencies: Optional[np.ndarray] = None
    reference: Optional[AcousticReference] = None
    default_bc_policy: str = "rigid"  # 'rigid' (fill missing with zero) or 'error' (require explicit domains)

    c: float = 343.0
    rho: float = 1.225

    # Optional performance preset hint. Geometry builders can use this as a default
    # when explicit mesh sizing is not provided.
    mesh_preset: Optional[str] = None

    solver_options: SolverOptions = field(default_factory=SolverOptions)
    use_osrc: bool = False
    osrc_npade: int = 2
    driver_network: Optional["CompressionDriverNetwork"] = None
    driver_excitation: Optional["CompressionDriverExcitation"] = None

    polar_start: float = 0.0
    polar_end: float = 180.0
    polar_num: int = 37
    norm_angle: float = 0.0
    measurement_distance: float = 1.0
    spl_angles: Optional[List[float]] = None

    waveguide: Optional[WaveguideMetadata] = None
    execution_config: Optional[object] = None
    domain_names: Optional[Dict[int, str]] = None  # Map domain ID -> human-readable name
