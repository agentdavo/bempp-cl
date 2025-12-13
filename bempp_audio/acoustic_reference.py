"""
Acoustic reference geometry for consistent evaluation defaults.

`bempp_audio` meshes can be positioned/offset (e.g. waveguide-on-box). To keep
"on-axis", polar sweeps, and measurement distances consistent across the API,
we model an explicit acoustic reference: an origin + forward axis + default
reference distance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np


def _as_vec3(x: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    v = np.asarray(x, dtype=float).reshape(3)
    return v


def _unit(x: np.ndarray) -> np.ndarray:
    v = _as_vec3(x)
    n = float(np.linalg.norm(v))
    if n <= 0:
        raise ValueError("axis must be non-zero")
    return v / n


@dataclass(frozen=True)
class AcousticReference:
    """
    Reference for "on-axis" and measurement evaluation.

    Parameters
    ----------
    origin:
        A point in space defining the acoustic origin (typically mouth center).
    axis:
        Forward axis direction (will be normalized).
    default_distance_m:
        Default measurement distance used by helpers when a distance is not
        provided explicitly.
    """

    origin: np.ndarray
    axis: np.ndarray
    default_distance_m: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "origin", _as_vec3(self.origin))
        object.__setattr__(self, "axis", _unit(self.axis))
        if float(self.default_distance_m) <= 0:
            raise ValueError("default_distance_m must be positive")

    @classmethod
    def from_origin_axis(
        cls,
        origin: np.ndarray | list[float] | tuple[float, float, float],
        axis: np.ndarray | list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0),
        *,
        default_distance_m: float = 1.0,
    ) -> "AcousticReference":
        return cls(origin=_as_vec3(origin), axis=_unit(axis), default_distance_m=float(default_distance_m))

    @classmethod
    def from_mesh(cls, mesh: object, *, default_distance_m: float = 1.0) -> "AcousticReference":
        origin = getattr(mesh, "center")
        axis = getattr(mesh, "axis")
        return cls.from_origin_axis(origin, axis, default_distance_m=default_distance_m)

    def with_default_distance(self, distance_m: float) -> "AcousticReference":
        return replace(self, default_distance_m=float(distance_m))

    def point_on_axis(self, distance_m: float | None = None) -> np.ndarray:
        d = self.default_distance_m if distance_m is None else float(distance_m)
        return self.origin + d * self.axis

