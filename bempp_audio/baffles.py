"""Typed baffle models used across bempp_audio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FreeSpace:
    """Unbaffled (full-space) radiation."""


@dataclass(frozen=True)
class InfiniteBaffle:
    """
    Infinite rigid baffle approximation.

    Notes
    -----
    Current implementation uses post-processing rather than a true half-space kernel.
    """

    plane_z: float = 0.0
    pressure_scale: float = 2.0


@dataclass(frozen=True)
class CircularBaffle:
    """Finite circular baffle represented as an additional rigid domain."""

    radius: float
    element_size: Optional[float] = None


Baffle = FreeSpace | InfiniteBaffle | CircularBaffle

