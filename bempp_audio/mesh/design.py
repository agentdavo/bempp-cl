"""Waveguide sizing helpers (pure geometry).

These helpers provide simple first-order mappings between geometry and
"coverage-like" angles. They are intended as design utilities and should be
treated as approximations (true directivity depends on frequency and details
of the contour, mouth termination, and baffle).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def conical_half_angle_rad(
    *,
    throat_radius: float,
    mouth_radius: float,
    length: float,
) -> float:
    """Return the conical half-angle (from z-axis) implied by endpoints."""
    if length <= 0:
        raise ValueError("length must be positive")
    if throat_radius <= 0 or mouth_radius <= 0:
        raise ValueError("throat_radius and mouth_radius must be > 0")
    if mouth_radius <= throat_radius:
        raise ValueError("mouth_radius must be > throat_radius")
    return float(np.arctan((float(mouth_radius) - float(throat_radius)) / float(length)))


def conical_half_angle_deg(
    *,
    throat_radius: float,
    mouth_radius: float,
    length: float,
) -> float:
    """Return the conical half-angle (degrees from z-axis) implied by endpoints."""
    return float(np.rad2deg(conical_half_angle_rad(throat_radius=throat_radius, mouth_radius=mouth_radius, length=length)))


def mouth_radius_from_half_angle(
    *,
    throat_radius: float,
    length: float,
    half_angle_deg: Optional[float] = None,
    half_angle_rad: Optional[float] = None,
) -> float:
    """Return mouth radius for a conical expansion at the given half-angle."""
    if half_angle_deg is not None and half_angle_rad is not None:
        raise TypeError("Pass only one of `half_angle_deg` or `half_angle_rad`.")
    if length <= 0:
        raise ValueError("length must be positive")
    if throat_radius <= 0:
        raise ValueError("throat_radius must be > 0")

    if half_angle_deg is not None:
        alpha = float(np.deg2rad(float(half_angle_deg)))
    elif half_angle_rad is not None:
        alpha = float(half_angle_rad)
    else:
        raise TypeError("Provide `half_angle_deg` or `half_angle_rad`.")

    if not (0.0 < alpha < np.deg2rad(89.9)):
        raise ValueError("half_angle must be in (0, 89.9°)")

    return float(throat_radius) + float(length) * float(np.tan(alpha))


def mouth_diameter_from_half_angle(
    *,
    throat_diameter: float,
    length: float,
    half_angle_deg: Optional[float] = None,
    half_angle_rad: Optional[float] = None,
) -> float:
    """Return mouth diameter for a conical expansion at the given half-angle."""
    throat_r = float(throat_diameter) / 2.0
    mouth_r = mouth_radius_from_half_angle(
        throat_radius=throat_r,
        length=length,
        half_angle_deg=half_angle_deg,
        half_angle_rad=half_angle_rad,
    )
    return 2.0 * float(mouth_r)


def rectangular_mouth_from_half_angles(
    *,
    throat_diameter: float,
    length: float,
    half_angle_h_deg: float,
    half_angle_v_deg: float,
) -> Tuple[float, float]:
    """Return (width, height) from horizontal/vertical conical half-angles.

    This is a simple design heuristic for rectangular mouths:
        width  ≈ 2 * (throat_r + L * tan(alpha_h))
        height ≈ 2 * (throat_r + L * tan(alpha_v))
    """
    throat_r = float(throat_diameter) / 2.0
    w = 2.0 * mouth_radius_from_half_angle(throat_radius=throat_r, length=length, half_angle_deg=float(half_angle_h_deg))
    h = 2.0 * mouth_radius_from_half_angle(throat_radius=throat_r, length=length, half_angle_deg=float(half_angle_v_deg))
    return float(w), float(h)


def os_opening_angle_bounds_deg(
    *,
    throat_radius: float,
    mouth_radius: float,
    length: float,
    alpha_max_limit_deg: float = 89.9,
    n_scan: int = 2048,
) -> Tuple[float, float]:
    """Return (min_deg, max_deg) feasible for the OS hyperboloid parameterization.

    The implementation in `bempp_audio.mesh.profiles.oblate_spheroidal_profile` uses:
        r(x)^2 = w^2 + ((x + z0) tan α)^2
    with `α` the opening half-angle (from z-axis). For fixed (throat_r, mouth_r, L),
    not all α are feasible because `w^2` must be non-negative.

    Returns
    -------
    (min_deg, max_deg)
        - min_deg is the conical half-angle implied by endpoints.
        - max_deg is the largest α (<= alpha_max_limit_deg) such that w^2 >= 0.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if throat_radius <= 0 or mouth_radius <= 0:
        raise ValueError("throat_radius and mouth_radius must be > 0")
    if mouth_radius <= throat_radius:
        raise ValueError("mouth_radius must be > throat_radius")

    r0 = float(throat_radius)
    r1 = float(mouth_radius)
    L = float(length)
    alpha_min = float(np.arctan((r1 - r0) / L))
    alpha_hi = float(np.deg2rad(float(alpha_max_limit_deg)))
    if alpha_hi <= alpha_min:
        return float(np.rad2deg(alpha_min)), float(np.rad2deg(alpha_min))

    def w2(alpha: float) -> float:
        alpha = float(alpha)
        t = float(np.tan(alpha))
        if not np.isfinite(t) or t <= 0:
            return float("nan")
        z0 = (r1 * r1 - r0 * r0) / (2.0 * L * (t * t)) - 0.5 * L
        return float(r0 * r0 - (z0 * t) ** 2)

    eps = max(1e-24, 1e-12 * r0 * r0)

    # Coarse scan to find the last angle with w^2 >= 0.
    n = int(max(64, n_scan))
    angles = np.linspace(alpha_min, alpha_hi, n, endpoint=True)
    vals = np.array([w2(a) for a in angles], dtype=float)
    ok = np.isfinite(vals) & (vals >= -eps)
    if not ok.any():
        # Degenerate: treat as conical only.
        a = float(np.rad2deg(alpha_min))
        return a, a

    last = int(np.where(ok)[0][-1])
    if last == n - 1:
        return float(np.rad2deg(alpha_min)), float(np.rad2deg(alpha_hi))

    # Refine the boundary between `last` (ok) and `last+1` (not ok) by bisection.
    lo = float(angles[last])
    hi = float(angles[last + 1])
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if w2(mid) >= -eps:
            lo = mid
        else:
            hi = mid

    return float(np.rad2deg(alpha_min)), float(np.rad2deg(lo))
