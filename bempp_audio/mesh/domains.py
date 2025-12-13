"""
Domain (physical group) identifiers used across bempp_audio meshes.

These constants standardize how multi-domain meshes label surfaces for
boundary conditions (e.g., vibrating throat vs rigid walls).
"""

from __future__ import annotations

from enum import IntEnum


class Domain(IntEnum):
    THROAT = 1
    WALLS = 2
    ENCLOSURE = 3

