"""
bempp-audio: High-level acoustic simulation API built on bempp-cl.

A professional Python API for loudspeaker acoustic simulation using the
Boundary Element Method (BEM).

Example usage:
    from bempp_audio import Loudspeaker

    result = (Loudspeaker()
        .circular_piston(radius=0.05)
        .infinite_baffle()
        .frequency_range(20, 20000)
        .solve())

    result.plot_spl()
"""

__version__ = "0.1.0"

from bempp_audio._optional import UnavailableDependency

# Core classes
from bempp_audio.mesh import LoudspeakerMesh
from bempp_audio.velocity import VelocityProfile
from bempp_audio.solver import RadiationSolver, OSRCRadiationSolver
from bempp_audio.acoustic_reference import AcousticReference
from bempp_audio.results import (
    RadiationResult,
    PressureField,
    DirectivityPattern,
    FrequencyResponse,
    RadiationImpedance,
)

# High-level fluent API
from bempp_audio.api import (
    Loudspeaker,
    LoudspeakerBuilder,
    LoudspeakerSimulation,
    SimulationRequest,
    FrequencySpacing,
    BoundaryConditionPolicy,
    Plane2D,
    VelocityMode,
    PolarPlane,
)

# I/O utilities
from bempp_audio.io import AcousticIO

try:
    from bempp_audio.viz import ReportBuilder
except ImportError as e:  # pragma: no cover
    ReportBuilder = UnavailableDependency(
        name="bempp_audio.viz (matplotlib)",
        install_hint="Install with `pip install bempp-cl[audio]`.",
        original_error=e,
    )

# Configuration
from bempp_audio.config import (
    SimulationConfig,
    FrequencyConfig,
    DirectivityConfig,
    MediumConfig,
    SolverConfig,
    ExecutionConfig,
    ConfigPresets,
)

# Logging and progress
from bempp_audio.progress import (
    progress,
    configure,
    get_logger,
    set_log_level,
    ProgressTracker,
    get_device_info,
    log_device_info,
)

__all__ = [
    # Core
    "LoudspeakerMesh",
    "VelocityProfile",
    "RadiationSolver",
    "OSRCRadiationSolver",
    "AcousticReference",
    # Results
    "RadiationResult",
    "PressureField",
    "DirectivityPattern",
    "FrequencyResponse",
    "RadiationImpedance",
    # API
    "Loudspeaker",
    "LoudspeakerBuilder",
    "LoudspeakerSimulation",
    "SimulationRequest",
    "FrequencySpacing",
    "BoundaryConditionPolicy",
    "Plane2D",
    "VelocityMode",
    "PolarPlane",
    # Configuration
    "SimulationConfig",
    "FrequencyConfig",
    "DirectivityConfig",
    "MediumConfig",
    "SolverConfig",
    "ExecutionConfig",
    "ConfigPresets",
    # I/O
    "AcousticIO",
    # Viz
    "ReportBuilder",
    # Logging
    "progress",
    "configure",
    "get_logger",
    "set_log_level",
    "ProgressTracker",
    "get_device_info",
    "log_device_info",
]
