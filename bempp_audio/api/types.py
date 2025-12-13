"""
Typed enums for the bempp_audio fluent API.

These are intentionally small and string-compatible (`str, Enum`) so that:
- existing string-based code keeps working, and
- users get better editor/typing support.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Union


class FrequencySpacing(str, Enum):
    LOG = "log"
    LINEAR = "linear"
    OCTAVE = "octave"


class BoundaryConditionPolicy(str, Enum):
    RIGID = "rigid"
    ERROR = "error"


class Plane2D(str, Enum):
    XZ = "xz"
    XY = "xy"
    YZ = "yz"

class VelocityMode(str, Enum):
    PISTON = "piston"
    GAUSSIAN = "gaussian"
    RADIAL_TAPER = "radial_taper"
    ZERO = "zero"


class PolarPlane(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


FrequencySpacingLike = Union[FrequencySpacing, Literal["log", "logarithmic", "linear", "lin", "octave"]]
BCPolicyLike = Union[BoundaryConditionPolicy, Literal["rigid", "error"]]
Plane2DLike = Union[Plane2D, Literal["xz", "xy", "yz"]]
VelocityModeLike = Union[VelocityMode, Literal["piston", "gaussian", "radial_taper", "zero"]]
PolarPlaneLike = Union[PolarPlane, Literal["horizontal", "vertical"]]


def normalize_frequency_spacing(spacing: FrequencySpacingLike) -> str:
    if isinstance(spacing, FrequencySpacing):
        return spacing.value
    return str(spacing).strip().lower()


def normalize_bc_policy(policy: BCPolicyLike) -> str:
    if isinstance(policy, BoundaryConditionPolicy):
        return policy.value
    return str(policy).strip().lower()


def normalize_plane_2d(plane: Plane2DLike) -> str:
    if isinstance(plane, Plane2D):
        return plane.value
    return str(plane).strip().lower()


def normalize_velocity_mode(mode: VelocityModeLike) -> str:
    if isinstance(mode, VelocityMode):
        return mode.value
    return str(mode).strip().lower()


def normalize_polar_plane(plane: PolarPlaneLike) -> str:
    if isinstance(plane, PolarPlane):
        return plane.value
    return str(plane).strip().lower()
