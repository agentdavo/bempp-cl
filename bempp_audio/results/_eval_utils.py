"""
Shared evaluation helpers for results.

This module centralizes small utilities used by multiple evaluators
(pressure field, far-field directivity, etc.) to reduce duplication and keep
conventions consistent.
"""

from __future__ import annotations

import numpy as np


def cache_key_for_array(a: np.ndarray) -> tuple:
    """Stable cache key for a numeric array (shape/dtype/content)."""
    a = np.asarray(a)
    return (a.shape, str(a.dtype), a.tobytes())


def mask_points_in_front_of_plane(points: np.ndarray, plane_z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask, points_eval) for points with z >= plane_z."""
    pts = np.asarray(points)
    if pts.ndim == 1:
        pts = pts.reshape(3, 1)
    mask = pts[2, :] >= float(plane_z)
    return mask, pts[:, mask]


def mask_directions_front_hemisphere(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask, directions_eval) for directions with z >= 0 (far-field hemisphere)."""
    dirs = np.asarray(directions)
    if dirs.ndim == 1:
        dirs = dirs.reshape(3, 1)
    mask = dirs[2, :] >= 0.0
    return mask, dirs[:, mask]

