"""
Polar/measurement utilities for loudspeaker results.

This module defines a small, first-class representation of a polar sweep
computed from a `FrequencyResponse`, with explicit conventions.

Conventions
-----------
- Angles are degrees from the forward axis (0° = on-axis, +z by convention).
- Polar sweeps are taken in a plane defined relative to the mesh axis:
  - `plane="horizontal"` prefers the global +x direction as the lateral axis.
  - `plane="vertical"` prefers the global +y direction as the lateral axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional
import numpy as np


PolarPlane = Literal["horizontal", "vertical"]


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 0:
        raise ValueError("zero-length vector")
    return v / n


def _lateral_unit(axis: np.ndarray, prefer: np.ndarray) -> np.ndarray:
    axis_u = _unit(axis)
    prefer_u = _unit(prefer)
    if abs(float(np.dot(axis_u, prefer_u))) > 0.9:
        prefer_u = np.array([0.0, 1.0, 0.0]) if abs(float(axis_u[1])) < 0.9 else np.array([1.0, 0.0, 0.0])
    # Project the preferred direction onto the plane orthogonal to the axis.
    # This ensures that `plane="horizontal"` with axis≈+z actually uses +x as
    # the lateral direction (rather than a 90°-rotated cross-product axis).
    u = prefer_u - float(np.dot(prefer_u, axis_u)) * axis_u
    n = float(np.linalg.norm(u))
    if n <= 1e-14:
        # Fallback if projection is nearly zero (should be rare due to the
        # prefer_u re-selection above).
        u = np.array([1.0, 0.0, 0.0]) if abs(float(axis_u[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _unit(u)


def polar_directions(axis: np.ndarray, angles_deg: np.ndarray, plane: PolarPlane = "horizontal") -> np.ndarray:
    """
    Directions for a 2D polar sweep around the mesh axis.

    Parameters
    ----------
    axis:
        Forward axis (3,), will be normalized.
    angles_deg:
        Angles in degrees from the axis (0° = axis direction).
    plane:
        `horizontal` prefers global +x as the lateral direction, `vertical`
        prefers global +y.
    """
    axis_u = _unit(axis)
    prefer = np.array([1.0, 0.0, 0.0]) if plane == "horizontal" else np.array([0.0, 1.0, 0.0])
    u = _lateral_unit(axis_u, prefer)

    thetas = np.deg2rad(np.asarray(angles_deg, dtype=float))
    dirs = (np.cos(thetas)[None, :] * axis_u[:, None]) + (np.sin(thetas)[None, :] * u[:, None])
    return dirs


def polar_points(center: np.ndarray, directions: np.ndarray, distance: float) -> np.ndarray:
    center = np.asarray(center, dtype=float).reshape(3)
    return center[:, None] + float(distance) * np.asarray(directions, dtype=float)


@dataclass(frozen=True)
class PolarSweep:
    frequencies_hz: np.ndarray  # (n_f,)
    angles_deg: np.ndarray  # (n_a,)
    pressure: np.ndarray  # (n_f, n_a) complex
    distance_m: float
    plane: PolarPlane

    def spl_db(self, ref_pa: float = 20e-6) -> np.ndarray:
        """SPL in dB re `ref_pa` (uses RMS from complex peak by /sqrt(2))."""
        p_rms = np.abs(self.pressure) / np.sqrt(2.0)
        p_rms = np.maximum(p_rms, 1e-20)
        return 20.0 * np.log10(p_rms / float(ref_pa))

    def by_angle(self) -> Dict[float, np.ndarray]:
        """Mapping: angle(deg) -> SPL(dB) over frequency."""
        spl = self.spl_db()
        return {float(a): spl[:, i] for i, a in enumerate(self.angles_deg)}


def compute_polar_sweep(
    response: "FrequencyResponse",
    *,
    angles_deg: Iterable[float],
    distance_m: float = 1.0,
    plane: PolarPlane = "horizontal",
    show_progress: bool = False,
) -> PolarSweep:
    from bempp_audio.results.pressure_field import PressureField
    from bempp_audio.progress import ProgressTracker

    results = response.results
    if not results:
        return PolarSweep(
            frequencies_hz=np.array([], dtype=float),
            angles_deg=np.asarray(list(angles_deg), dtype=float),
            pressure=np.zeros((0, len(list(angles_deg))), dtype=complex),
            distance_m=float(distance_m),
            plane=plane,
        )

    angles = np.asarray(list(angles_deg), dtype=float)
    freqs = np.asarray([r.frequency for r in results], dtype=float)

    # Use per-result mesh axis/center to avoid relying on global conventions.
    pressures = np.zeros((len(results), len(angles)), dtype=complex)
    with ProgressTracker(
        total=len(results),
        desc="Polar sweep",
        unit="freq",
        disable=not bool(show_progress),
    ) as pbar:
        for i, result in enumerate(results):
            axis = getattr(getattr(result, "reference", None), "axis", result.mesh.axis)
            origin = getattr(getattr(result, "reference", None), "origin", result.mesh.center)
            dirs = polar_directions(axis, angles, plane=plane)
            pts = polar_points(origin, dirs, distance=float(distance_m))
            pressures[i, :] = PressureField(result).at_points(pts)
            pbar.update(item=f"{result.frequency:.0f} Hz")

    return PolarSweep(
        frequencies_hz=freqs,
        angles_deg=angles,
        pressure=pressures,
        distance_m=float(distance_m),
        plane=plane,
    )
