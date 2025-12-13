"""High-level fluent API for acoustic simulation.

This package intentionally avoids importing heavy submodules at import time so
that `bempp_audio.results` can safely depend on lightweight API types without
creating circular imports.
"""

from __future__ import annotations

from typing import Any

from bempp_audio.api.request import SimulationRequest
from bempp_audio.api.types import FrequencySpacing, BoundaryConditionPolicy, Plane2D, VelocityMode, PolarPlane

__all__ = [
    "Loudspeaker",
    "LoudspeakerBuilder",
    "LoudspeakerSimulation",
    "SimulationRequest",
    "FrequencySpacing",
    "BoundaryConditionPolicy",
    "Plane2D",
    "VelocityMode",
    "PolarPlane",
]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in ("Loudspeaker", "LoudspeakerBuilder", "LoudspeakerSimulation"):
        from bempp_audio.api.loudspeaker import Loudspeaker

        if name == "Loudspeaker":
            return Loudspeaker
        return Loudspeaker
    raise AttributeError(name)
