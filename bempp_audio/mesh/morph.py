"""Cross-section morphing utilities (pure geometry).

This module provides the math needed to morph a circular cross-section into
other shapes (rectangle/ellipse/superellipse) along the waveguide axis.

The mesh generators can consume these helpers to create non-axisymmetric
waveguides via cross-section lofting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Sequence

import numpy as np


class MorphTargetShape(IntEnum):
    KEEP = 0
    RECTANGLE = 1
    ELLIPSE = 2
    SUPERELLIPSE = 3
    SUPERFORMULA = 4


@dataclass(frozen=True)
class SuperFormulaConfig:
    """Superformula parameters and sizing.

    Two equivalent definitions are supported:
    - `f`: (a, b, m, n1, n2, n3) where m1=m2=m
    - explicit named parameters: a, b, m1, m2, n1, n2, n3

    Absolute size is set with `width` and `aspect_ratio` (width / height).
    """

    # Compact definition: (a, b, m, n1, n2, n3), with m1=m2=m.
    f: Optional[Tuple[float, float, float, float, float, float]] = None

    # Alternate definition (named parameters).
    a: float = 1.0
    b: float = 1.0
    m1: float = 0.0
    m2: float = 0.0
    n1: float = 1.0
    n2: float = 1.0
    n3: float = 1.0

    # Size controls (at the mouth).
    width: float = 0.0
    aspect_ratio: float = 1.0

    def resolved(self) -> Tuple[float, float, float, float, float, float, float]:
        """Return resolved (a, b, m1, m2, n1, n2, n3), applying `f` if present."""
        if self.f is None:
            return (
                float(self.a),
                float(self.b),
                float(self.m1),
                float(self.m2),
                float(self.n1),
                float(self.n2),
                float(self.n3),
            )
        if len(self.f) != 6:
            raise ValueError("SuperFormulaConfig.f must have 6 values: (a, b, m, n1, n2, n3)")
        a, b, m, n1, n2, n3 = (float(v) for v in self.f)
        return float(a), float(b), float(m), float(m), float(n1), float(n2), float(n3)

    def validate(self) -> None:
        a, b, _m1, _m2, n1, n2, n3 = self.resolved()
        if a <= 0 or b <= 0:
            raise ValueError("SuperFormulaConfig a/b must be > 0")
        if n1 <= 0 or n2 <= 0 or n3 <= 0:
            raise ValueError("SuperFormulaConfig n1/n2/n3 must be > 0")
        if self.width < 0:
            raise ValueError("SuperFormulaConfig.width must be >= 0")
        if self.aspect_ratio <= 0:
            raise ValueError("SuperFormulaConfig.aspect_ratio must be > 0")

    @classmethod
    def from_f(
        cls,
        f: Sequence[float],
        *,
        width: float,
        aspect_ratio: float = 1.0,
    ) -> "SuperFormulaConfig":
        if len(f) != 6:
            raise ValueError("f must have 6 values: (a, b, m, n1, n2, n3)")
        return cls(
            f=(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5])),
            width=float(width),
            aspect_ratio=float(aspect_ratio),
        )


@dataclass(frozen=True)
class MorphConfig:
    """Parameters controlling cross-section morphing from circle to a target shape."""

    target_shape: MorphTargetShape = MorphTargetShape.KEEP
    target_width: float = 0.0   # [m] full width at mouth; 0 => use mouth diameter
    target_height: float = 0.0  # [m] full height at mouth; 0 => use mouth diameter
    corner_radius: float = 0.0  # [m] fillet radius for rectangles (implemented geometrically)

    fixed_part: float = 0.0     # [0,1) fraction of length kept circular
    end_part: float = 1.0       # (0,1] normalized position where morph completes
    rate: float = 3.0           # >0, lower => faster transition
    allow_shrinkage: bool = False
    profile_mode: str = "blend"  # "blend" (legacy), "radial" (support-blend), or "axes" (monotone dims)
    enforce_noncontracting: bool = False  # If True, allow per-slice scaling to prevent contraction.
    enforce_mode: str = "axes"  # "axes" or "directions"
    enforce_axes: Tuple[str, ...] = ("y",)  # When mode="axes": ("x",), ("y",), or ("x","y").
    enforce_n_directions: int = 16  # When mode="directions": number of directions around the circle.
    enforce_tol: float = 0.0  # Allowed contraction tolerance (meters) for enforced axes.

    # Optional: explicit superellipse exponent for SUPERELLIPSE mode
    superellipse_n: Optional[float] = None

    # Optional: parameters for SUPERFORMULA mode
    superformula: Optional[SuperFormulaConfig] = None

    def validate(self) -> None:
        if self.target_width < 0 or self.target_height < 0:
            raise ValueError("target_width/target_height must be >= 0")
        if self.corner_radius < 0:
            raise ValueError("corner_radius must be >= 0")
        if not (0.0 <= float(self.fixed_part) < 1.0):
            raise ValueError("fixed_part must be in [0, 1)")
        if not (0.0 < float(self.end_part) <= 1.0):
            raise ValueError("end_part must be in (0, 1]")
        if float(self.end_part) <= float(self.fixed_part):
            raise ValueError("end_part must be > fixed_part")
        if self.rate <= 0:
            raise ValueError("rate must be > 0")
        if self.profile_mode not in ("blend", "radial", "axes"):
            raise ValueError("profile_mode must be 'blend', 'radial', or 'axes'")
        if self.enforce_mode not in ("axes", "directions"):
            raise ValueError("enforce_mode must be 'axes' or 'directions'")
        if any(a not in ("x", "y") for a in self.enforce_axes):
            raise ValueError("enforce_axes entries must be 'x' and/or 'y'")
        if int(self.enforce_n_directions) < 4:
            raise ValueError("enforce_n_directions must be >= 4")
        if float(self.enforce_tol) < 0:
            raise ValueError("enforce_tol must be >= 0")
        if self.target_shape == MorphTargetShape.SUPERELLIPSE:
            if self.superellipse_n is None:
                raise ValueError("superellipse_n must be set for SUPERELLIPSE target")
            if self.superellipse_n < 2:
                raise ValueError("superellipse_n must be >= 2")
        if self.target_shape == MorphTargetShape.SUPERFORMULA:
            if self.superformula is None:
                raise ValueError("superformula must be set for SUPERFORMULA target")
            self.superformula.validate()

    @classmethod
    def rectangle(
        cls,
        *,
        width: float,
        height: float,
        corner_radius: float = 0.0,
        fixed_part: float = 0.0,
        end_part: float = 1.0,
        rate: float = 3.0,
        allow_shrinkage: bool = False,
        profile_mode: str = "blend",
        enforce_noncontracting: bool = False,
        enforce_mode: str = "axes",
        enforce_axes: Tuple[str, ...] = ("y",),
        enforce_n_directions: int = 16,
        enforce_tol: float = 0.0,
    ) -> "MorphConfig":
        return cls(
            target_shape=MorphTargetShape.RECTANGLE,
            target_width=float(width),
            target_height=float(height),
            corner_radius=float(corner_radius),
            fixed_part=float(fixed_part),
            end_part=float(end_part),
            rate=float(rate),
            allow_shrinkage=bool(allow_shrinkage),
            profile_mode=str(profile_mode),
            enforce_noncontracting=bool(enforce_noncontracting),
            enforce_mode=str(enforce_mode),
            enforce_axes=tuple(enforce_axes),
            enforce_n_directions=int(enforce_n_directions),
            enforce_tol=float(enforce_tol),
        )

    @classmethod
    def superformula_f(
        cls,
        f: Sequence[float],
        *,
        width: float,
        aspect_ratio: float = 1.0,
        fixed_part: float = 0.0,
        end_part: float = 1.0,
        rate: float = 3.0,
        allow_shrinkage: bool = False,
        profile_mode: str = "blend",
        enforce_noncontracting: bool = False,
        enforce_mode: str = "axes",
        enforce_axes: Tuple[str, ...] = ("y",),
        enforce_n_directions: int = 16,
        enforce_tol: float = 0.0,
    ) -> "MorphConfig":
        return cls(
            target_shape=MorphTargetShape.SUPERFORMULA,
            fixed_part=float(fixed_part),
            end_part=float(end_part),
            rate=float(rate),
            allow_shrinkage=bool(allow_shrinkage),
            profile_mode=str(profile_mode),
            enforce_noncontracting=bool(enforce_noncontracting),
            enforce_mode=str(enforce_mode),
            enforce_axes=tuple(enforce_axes),
            enforce_n_directions=int(enforce_n_directions),
            enforce_tol=float(enforce_tol),
            superformula=SuperFormulaConfig.from_f(f, width=float(width), aspect_ratio=float(aspect_ratio)),
        )

    @classmethod
    def superformula_params(
        cls,
        *,
        a: float,
        b: float,
        m1: float,
        m2: float,
        n1: float,
        n2: float,
        n3: float,
        width: float,
        aspect_ratio: float = 1.0,
        fixed_part: float = 0.0,
        end_part: float = 1.0,
        rate: float = 3.0,
        allow_shrinkage: bool = False,
        profile_mode: str = "blend",
        enforce_noncontracting: bool = False,
        enforce_mode: str = "axes",
        enforce_axes: Tuple[str, ...] = ("y",),
        enforce_n_directions: int = 16,
        enforce_tol: float = 0.0,
    ) -> "MorphConfig":
        return cls(
            target_shape=MorphTargetShape.SUPERFORMULA,
            fixed_part=float(fixed_part),
            end_part=float(end_part),
            rate=float(rate),
            allow_shrinkage=bool(allow_shrinkage),
            profile_mode=str(profile_mode),
            enforce_noncontracting=bool(enforce_noncontracting),
            enforce_mode=str(enforce_mode),
            enforce_axes=tuple(enforce_axes),
            enforce_n_directions=int(enforce_n_directions),
            enforce_tol=float(enforce_tol),
            superformula=SuperFormulaConfig(
                a=float(a),
                b=float(b),
                m1=float(m1),
                m2=float(m2),
                n1=float(n1),
                n2=float(n2),
                n3=float(n3),
                width=float(width),
                aspect_ratio=float(aspect_ratio),
            ),
        )


def apply_noncontracting_axis_scaling(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    prev_x: Optional[float],
    prev_y: Optional[float],
    axes: Tuple[str, ...] = ("y",),
    tol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Apply per-slice axis scaling to prevent contraction along selected axes.

    Returns:
        (xs_scaled, ys_scaled, new_prev_x, new_prev_y, scale_x, scale_y)
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have the same shape")

    sx = 1.0
    sy = 1.0
    support_x = float(np.max(np.abs(xs))) if xs.size else 0.0
    support_y = float(np.max(np.abs(ys))) if ys.size else 0.0

    tol = float(tol)

    if "x" in axes and prev_x is not None and support_x < float(prev_x) - tol and support_x > 0:
        sx = float(prev_x) / support_x
    if "y" in axes and prev_y is not None and support_y < float(prev_y) - tol and support_y > 0:
        sy = float(prev_y) / support_y

    xs2 = xs * sx
    ys2 = ys * sy

    new_support_x = float(np.max(np.abs(xs2))) if xs2.size else 0.0
    new_support_y = float(np.max(np.abs(ys2))) if ys2.size else 0.0
    new_prev_x = new_support_x if prev_x is None else max(float(prev_x), new_support_x)
    new_prev_y = new_support_y if prev_y is None else max(float(prev_y), new_support_y)
    return xs2, ys2, float(new_prev_x), float(new_prev_y), float(sx), float(sy)


def apply_noncontracting_direction_scaling(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    prev_support: Optional[np.ndarray],
    n_directions: int,
    tol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Apply isotropic per-slice scaling to prevent contraction in multiple directions.

    Directions are evenly spaced angles in [0, 2π). Support in each direction is:
        s_j = max_k (x_k cosθ_j + y_k sinθ_j).

    Returns:
        (xs_scaled, ys_scaled, new_prev_support, scale)
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have the same shape")

    n = int(n_directions)
    if n < 4:
        raise ValueError("n_directions must be >= 4")

    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    c = np.cos(angles)
    s = np.sin(angles)
    # (n_dir, n_pts)
    proj = (c[:, None] * xs[None, :]) + (s[:, None] * ys[None, :])
    support = np.max(proj, axis=1)

    if prev_support is None:
        return xs, ys, support, 1.0

    prev_support = np.asarray(prev_support, dtype=float).reshape(n)
    tol = float(tol)
    with np.errstate(divide="ignore", invalid="ignore"):
        needed = np.where(support < (prev_support - tol), prev_support / np.maximum(support, 1e-30), 1.0)
    scale = float(np.max(needed))
    if scale <= 1.0:
        new_prev = np.maximum(prev_support, support)
        return xs, ys, new_prev, 1.0

    xs2 = xs * scale
    ys2 = ys * scale
    new_prev = np.maximum(prev_support, support * scale)
    return xs2, ys2, new_prev, scale


def superellipse_xy(theta: np.ndarray, a: float, b: float, n: float) -> Tuple[np.ndarray, np.ndarray]:
    """Superellipse parameterization.

    Uses the standard exponent form:
      x = a * sign(cosθ) * |cosθ|^(2/n)
      y = b * sign(sinθ) * |sinθ|^(2/n)

    Where n=2 is an ellipse and n→∞ approaches a rectangle with half-width a and
    half-height b.
    """
    theta = np.asarray(theta, dtype=float)
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be > 0")
    if n < 2:
        raise ValueError("n must be >= 2")

    c = np.cos(theta)
    s = np.sin(theta)
    p = 2.0 / float(n)
    x = float(a) * np.sign(c) * (np.abs(c) ** p)
    y = float(b) * np.sign(s) * (np.abs(s) ** p)
    return x, y


def superellipse_points(n_points: int, a: float, b: float, n: float) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a superellipse perimeter with `n_points` points (closed, no duplicate endpoint)."""
    if n_points < 8:
        raise ValueError("n_points must be >= 8")
    theta = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    return superellipse_xy(theta, a, b, n)


def _rounded_rectangle_first_quadrant_point(
    dx: float, dy: float, a: float, b: float, rc: float
) -> Tuple[float, float]:
    """Intersection of a ray with a rounded rectangle in the first quadrant.

    The ray starts at the origin with direction (dx,dy) where dx,dy >= 0 and
    not both zero. Returns the boundary point (x,y) in the first quadrant.
    """
    if dx <= 0 and dy <= 0:
        raise ValueError("ray direction must be non-zero")
    rc = float(np.clip(float(rc), 0.0, float(min(a, b))))

    if rc <= 0:
        # Sharp rectangle: choose whichever side we hit first.
        tx = float("inf") if dx <= 0 else (float(a) / float(dx))
        ty = float("inf") if dy <= 0 else (float(b) / float(dy))
        t = float(min(tx, ty))
        return float(t * dx), float(t * dy)

    # Try straight sides first: x=a (y <= b-rc) or y=b (x <= a-rc).
    if dx > 0:
        t = float(a) / float(dx)
        y = t * float(dy)
        if y <= float(b) - rc + 1e-12:
            return float(a), float(y)
    if dy > 0:
        t = float(b) / float(dy)
        x = t * float(dx)
        if x <= float(a) - rc + 1e-12:
            return float(x), float(b)

    # Otherwise, we intersect the corner arc centered at (a-rc, b-rc) with radius rc.
    # For a ray starting at the origin, the *outer* boundary point corresponds to the
    # farther (exiting) intersection with the circle, i.e. the larger positive root.
    cx = float(a) - rc
    cy = float(b) - rc
    norm = float(np.hypot(dx, dy))
    ux = float(dx) / norm
    uy = float(dy) / norm

    dot = ux * cx + uy * cy
    c2 = cx * cx + cy * cy
    disc = dot * dot - (c2 - rc * rc)
    disc = max(disc, 0.0)
    sqrt_disc = float(np.sqrt(disc))
    t1 = dot - sqrt_disc
    t2 = dot + sqrt_disc
    t = max(t1, t2)
    if t <= 0:
        raise ValueError("rounded-rectangle ray intersection failed (no positive root)")
    return float(t * ux), float(t * uy)


def rounded_rectangle_xy(theta: np.ndarray, a: float, b: float, corner_radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rounded rectangle boundary sampled at angles `theta` by ray intersection."""
    theta = np.asarray(theta, dtype=float)
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be > 0")
    if corner_radius < 0:
        raise ValueError("corner_radius must be >= 0")
    rc = float(np.clip(float(corner_radius), 0.0, float(min(a, b))))

    c = np.cos(theta)
    s = np.sin(theta)
    sx = np.sign(c)
    sy = np.sign(s)

    x = np.empty_like(theta)
    y = np.empty_like(theta)
    for i, (ci, si) in enumerate(zip(c, s)):
        dx = float(abs(ci))
        dy = float(abs(si))
        px, py = _rounded_rectangle_first_quadrant_point(dx, dy, float(a), float(b), rc)
        x[i] = float(sx[i]) * px
        y[i] = float(sy[i]) * py

    return x, y


def theta_with_corner_refinement(
    *,
    n_total: int,
    a: float,
    b: float,
    rc: float,
    corner_points: int = 0,
) -> np.ndarray:
    """Generate `n_total` sampling angles for a rounded rectangle.

    Key design requirement: keep principal planes (0, π/2, π, 3π/2) as explicit samples
    and allow corner refinement without creating pathological sub-mm segments.

    Implementation detail:
    - We allocate `n_total` samples across quadrants with 180° symmetry.
    - Within each quadrant, we distribute points along the *physical perimeter*
      (vertical side → corner arc → horizontal side), then convert those points to
      ray angles `theta=atan2(y,x)`. This avoids over-refining corners when the
      corner region spans only a small range of ray angles.
    """
    n_total = int(n_total)
    if n_total < 8:
        raise ValueError("n_total must be >= 8")
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be > 0")

    rc = float(np.clip(float(rc), 0.0, float(min(a, b))))
    corner_points = max(0, int(corner_points))

    base = n_total // 4
    rem = n_total - 4 * base

    # Prefer 180° symmetry in the allocation (rectangles are invariant under π rotation).
    # rem ∈ {0,1,2,3} for division by 4.
    n_q = [base, base, base, base]
    if rem >= 1:
        n_q[0] += 1
    if rem >= 2:
        n_q[2] += 1
    if rem >= 3:
        n_q[1] += 1

    # Build each quadrant by reflecting the first-quadrant prototype.
    # Important: for a!=b, Q2/Q4 are *reflections* (not simple +π/2 shifts).
    def _proto_interior(n_interior: int) -> np.ndarray:
        """First-quadrant angles in (0, π/2), excluding endpoints, length=n_interior."""
        n_interior = int(n_interior)
        if n_interior <= 0:
            return np.array([], dtype=float)

        # Degenerate corner radius: no arc; fall back to uniform sampling.
        if rc <= 0 or corner_points <= 0:
            return np.linspace(0.0, np.pi / 2, n_interior + 2, endpoint=True)[1:-1]

        # Allocate explicit corner arc sampling in *arc-length* (phi) rather than ray-angle.
        n_arc = int(min(int(corner_points), max(0, n_interior - 2)))
        n_side = int(n_interior - 2 - n_arc)
        side1_len = float(b) - float(rc)  # x=a, y: 0 -> b-rc
        side2_len = float(a) - float(rc)  # y=b, x: a-rc -> 0
        denom = max(1e-12, side1_len + side2_len)
        n_side1 = int(round(n_side * (side1_len / denom))) if n_side > 0 else 0
        n_side1 = int(np.clip(n_side1, 0, n_side))
        n_side2 = int(n_side - n_side1)

        # Build points in the first quadrant in increasing theta order.
        # Startpoint (a,0) is NOT included here (handled by quadrant start).
        pts: list[tuple[float, float]] = []

        # Side 1 interior (exclude endpoints).
        if n_side1 > 0 and side1_len > 0:
            ys = np.linspace(0.0, side1_len, n_side1 + 2, endpoint=True)[1:-1]
            pts.extend([(float(a), float(y)) for y in ys])

        # Tangent point side->arc.
        pts.append((float(a), float(side1_len)))

        # Arc interior points (exclude endpoints).
        if n_arc > 0:
            cx = float(a) - float(rc)
            cy = float(b) - float(rc)
            phi = np.linspace(0.0, np.pi / 2, n_arc + 2, endpoint=True)[1:-1]
            for ph in phi:
                x = cx + float(rc) * float(np.cos(ph))
                y = cy + float(rc) * float(np.sin(ph))
                pts.append((float(x), float(y)))

        # Tangent point arc->side.
        pts.append((float(side2_len), float(b)))

        # Side 2 interior (exclude endpoint at (0,b)).
        if n_side2 > 0 and side2_len > 0:
            xs = np.linspace(side2_len, 0.0, n_side2 + 2, endpoint=True)[1:-1]
            pts.extend([(float(x), float(b)) for x in xs])

        # Convert to theta and keep strict interior (0, π/2).
        theta = np.array([float(np.arctan2(y, x)) for x, y in pts], dtype=float)
        theta = np.clip(theta, 1e-12, (np.pi / 2) - 1e-12)

        # Ensure exact count.
        if theta.size != n_interior:
            return np.linspace(0.0, np.pi / 2, n_interior + 2, endpoint=True)[1:-1]
        return theta

    # Build each quadrant as a half-open interval, explicitly including the
    # quadrant start angles (0, π/2, π, 3π/2) exactly once. This keeps principal
    # plane cuts (e.g. x=0) aligned with polygon vertices when sampling.
    q0i = _proto_interior(n_q[0] - 1)
    q1i = _proto_interior(n_q[1] - 1)
    q2i = _proto_interior(n_q[2] - 1)
    q3i = _proto_interior(n_q[3] - 1)

    q0 = np.concatenate([np.array([0.0], dtype=float), q0i])
    q1 = np.concatenate([np.array([np.pi / 2], dtype=float), (np.pi - q1i[::-1])])
    q2 = np.concatenate([np.array([np.pi], dtype=float), (np.pi + q2i)])
    q3 = np.concatenate([np.array([3 * np.pi / 2], dtype=float), ((2 * np.pi) - q3i[::-1])])

    theta = np.concatenate([q0, q1, q2, q3])
    if len(theta) != n_total:
        return np.linspace(0.0, 2 * np.pi, n_total, endpoint=False)
    return theta


def _blend_u(t: float, *, rate: float, fixed_part: float, end_part: float) -> float:
    """Blend parameter u∈[0,1] with an initial fixed circular segment and optional early completion."""
    tt = float(t)
    if tt <= fixed_part:
        return 0.0
    if tt >= end_part:
        return 1.0
    u = (tt - float(fixed_part)) / (float(end_part) - float(fixed_part))
    return float(u ** float(rate))


def superellipse_exponent_for_corner_radius(
    a: float,
    b: float,
    corner_radius: float,
    *,
    n_max: float = 100.0,
) -> float:
    """Heuristic mapping: larger corner_radius => smaller superellipse exponent.

    Note: `MorphTargetShape.RECTANGLE` uses `rounded_rectangle_xy` (true fillet
    radius by ray intersection) and does not use this heuristic. This helper is
    retained for cases where a "roundness" control is desired via superellipse
    exponent (n=2 ellipse → n→∞ rectangle).
    """
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be > 0")
    if corner_radius < 0:
        raise ValueError("corner_radius must be >= 0")
    if n_max < 2:
        raise ValueError("n_max must be >= 2")

    r_max = float(min(a, b))
    if r_max <= 0:
        return 2.0

    # Normalize corner radius to [0,1], where 1 means "very round" (ellipse-like).
    f = float(np.clip(corner_radius / r_max, 0.0, 1.0))
    # Smoothly map: f=0 -> n_max (sharp), f=1 -> 2 (ellipse)
    # Use a cubic to keep most resolution near sharp corners.
    return float(2.0 + (1.0 - f) ** 3 * (float(n_max) - 2.0))


def superformula_xy(
    theta: np.ndarray,
    *,
    a: float,
    b: float,
    m1: float,
    m2: float,
    n1: float,
    n2: float,
    n3: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Superformula (Gielis) curve in polar form.

    r(θ) = ( |cos(m1 θ/4)/a|^n2 + |sin(m2 θ/4)/b|^n3 )^(-1/n1)
    """
    theta = np.asarray(theta, dtype=float)
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be > 0")
    if n1 <= 0 or n2 <= 0 or n3 <= 0:
        raise ValueError("n1, n2, n3 must be > 0")

    c = np.cos(float(m1) * theta / 4.0) / float(a)
    s = np.sin(float(m2) * theta / 4.0) / float(b)
    term = (np.abs(c) ** float(n2)) + (np.abs(s) ** float(n3))
    term = np.maximum(term, 1e-30)
    r = term ** (-1.0 / float(n1))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def _scale_to_width_aspect(
    x: np.ndarray, y: np.ndarray, *, width: float, aspect_ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale an arbitrary closed curve to a target width and aspect ratio."""
    if width <= 0:
        raise ValueError("width must be > 0")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be > 0")
    height = float(width) / float(aspect_ratio)

    max_x = float(np.max(np.abs(x)))
    max_y = float(np.max(np.abs(y)))
    if max_x <= 0 or max_y <= 0:
        raise ValueError("cannot scale degenerate curve")

    sx = (float(width) / 2.0) / max_x
    sy = (float(height) / 2.0) / max_y
    return x * sx, y * sy


def polygon_self_intersection_pairs(x: np.ndarray, y: np.ndarray) -> list[tuple[int, int]]:
    """Return intersecting segment index pairs for a closed polygon (O(n^2)).

    Segments are (i -> i+1), with the final segment (n-1 -> 0).
    Adjacent segments (sharing a vertex) are ignored.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D arrays of the same shape")
    n = int(x.size)
    if n < 4:
        return []

    def _orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def _on_segment(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
        return (
            min(ax, bx) - 1e-14 <= cx <= max(ax, bx) + 1e-14
            and min(ay, by) - 1e-14 <= cy <= max(ay, by) + 1e-14
        )

    def _segments_intersect(i: int, j: int) -> bool:
        a1x, a1y = float(x[i]), float(y[i])
        a2x, a2y = float(x[(i + 1) % n]), float(y[(i + 1) % n])
        b1x, b1y = float(x[j]), float(y[j])
        b2x, b2y = float(x[(j + 1) % n]), float(y[(j + 1) % n])

        o1 = _orient(a1x, a1y, a2x, a2y, b1x, b1y)
        o2 = _orient(a1x, a1y, a2x, a2y, b2x, b2y)
        o3 = _orient(b1x, b1y, b2x, b2y, a1x, a1y)
        o4 = _orient(b1x, b1y, b2x, b2y, a2x, a2y)

        # Proper intersection.
        if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
            return True

        # Colinear / endpoint touches.
        if abs(o1) <= 1e-14 and _on_segment(a1x, a1y, a2x, a2y, b1x, b1y):
            return True
        if abs(o2) <= 1e-14 and _on_segment(a1x, a1y, a2x, a2y, b2x, b2y):
            return True
        if abs(o3) <= 1e-14 and _on_segment(b1x, b1y, b2x, b2y, a1x, a1y):
            return True
        if abs(o4) <= 1e-14 and _on_segment(b1x, b1y, b2x, b2y, a2x, a2y):
            return True
        return False

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            # Skip adjacent segments and the (first,last) adjacency.
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue
            if _segments_intersect(i, j):
                pairs.append((i, j))
    return pairs


def require_simple_polygon_xy(x: np.ndarray, y: np.ndarray, *, name: str = "polygon") -> None:
    pairs = polygon_self_intersection_pairs(x, y)
    if pairs:
        raise ValueError(f"{name} self-intersects at segment pairs {pairs[:3]} (showing up to 3)")


def theta_for_morph(
    *,
    n_total: int,
    mouth_radius: float,
    morph: Optional[MorphConfig],
    corner_resolution: int = 0,
) -> np.ndarray:
    """Return sampling angles for morphed cross-sections, including corner refinement."""
    n_total = int(n_total)
    if n_total < 8:
        raise ValueError("n_total must be >= 8")
    theta = np.linspace(0.0, 2 * np.pi, n_total, endpoint=False)
    if morph is None or morph.target_shape != MorphTargetShape.RECTANGLE:
        return theta
    if float(getattr(morph, "corner_radius", 0.0)) <= 0:
        return theta
    cp = int(corner_resolution)
    if cp <= 0:
        return theta
    a = float(getattr(morph, "target_width", 0.0) or (2.0 * float(mouth_radius))) / 2.0
    b = float(getattr(morph, "target_height", 0.0) or (2.0 * float(mouth_radius))) / 2.0
    return theta_with_corner_refinement(
        n_total=n_total,
        a=float(a),
        b=float(b),
        rc=float(getattr(morph, "corner_radius", 0.0)),
        corner_points=cp,
    )


def morphed_sections_xy(
    *,
    x: np.ndarray,
    r: np.ndarray,
    length: float,
    theta: np.ndarray,
    throat_radius: float,
    mouth_radius: float,
    morph: MorphConfig,
    enforce_n_directions_default: int,
    name_prefix: str = "morph cross-section",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute enforced, validity-checked cross-section polygons for each axial slice.

    Returns
    -------
    (xs, ys)
        Arrays of shape (n_slices, n_theta) containing the boundary points for each slice.
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if x.shape != r.shape or x.ndim != 1:
        raise ValueError("x and r must be 1D arrays with the same shape")
    if theta.ndim != 1 or theta.size < 8:
        raise ValueError("theta must be a 1D array with >= 8 samples")
    if length <= 0:
        raise ValueError("length must be positive")

    morph.validate()

    xs_all = np.zeros((x.size, theta.size), dtype=float)
    ys_all = np.zeros((x.size, theta.size), dtype=float)

    prev_x = None
    prev_y = None
    prev_support = None
    for i, (xi, ri) in enumerate(zip(x, r)):
        t = float(xi) / float(length)
        xs, ys = morphed_cross_section_xy(
            theta,
            t=t,
            radius=float(ri),
            throat_radius=float(throat_radius),
            mouth_radius=float(mouth_radius),
            morph=morph,
        )
        if getattr(morph, "enforce_noncontracting", False):
            mode = str(getattr(morph, "enforce_mode", "axes"))
            if mode == "directions":
                xs, ys, prev_support, _s = apply_noncontracting_direction_scaling(
                    xs,
                    ys,
                    prev_support=prev_support,
                    n_directions=int(getattr(morph, "enforce_n_directions", int(enforce_n_directions_default))),
                    tol=float(getattr(morph, "enforce_tol", 0.0)),
                )
            else:
                xs, ys, prev_x, prev_y, _sx, _sy = apply_noncontracting_axis_scaling(
                    xs,
                    ys,
                    prev_x=prev_x,
                    prev_y=prev_y,
                    axes=tuple(getattr(morph, "enforce_axes", ("y",))),
                    tol=float(getattr(morph, "enforce_tol", 0.0)),
                )
        require_simple_polygon_xy(xs, ys, name=f"{name_prefix} t={t:.3f}")
        xs_all[i] = np.asarray(xs, dtype=float)
        ys_all[i] = np.asarray(ys, dtype=float)

    return xs_all, ys_all


def morphed_cross_section_xy(
    theta: np.ndarray,
    *,
    t: float,
    radius: float,
    throat_radius: Optional[float] = None,
    mouth_radius: float,
    morph: Optional[MorphConfig],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return cross-section boundary points for a slice at normalized position `t`.

    Parameters
    ----------
    theta : np.ndarray
        Angles in radians at which to sample the boundary.
    t : float
        Normalized position along length: 0=throat, 1=mouth.
    radius : float
        Equivalent circular radius for this slice (from the chosen profile).
    throat_radius : float, optional
        Required for `morph.profile_mode="radial"` to map the scalar radius to a
        normalized expansion factor.
    mouth_radius : float
        Equivalent circular radius at the mouth (used for scaling target width/height).
    morph : MorphConfig | None
        Morph settings; None or KEEP returns the circular cross-section.

    Notes
    -----
    `morph.profile_mode` controls how morph progress is parameterized:
    - `"blend"` uses axial position `t` (with `fixed_part/end_part/rate`) and blends
      boundary points between a circle and the target shape.
    - `"radial"` uses a normalized radius progress `s=(r-throat_r)/(mouth_r-throat_r)`,
      so the morph timing follows the chosen axial profile even when it is non-linear.
    - `"axes"` also uses radius progress `s`, but grows width/height monotonically
      from the throat radius to the target mouth dimensions to avoid "waist" artifacts
      for high-aspect-ratio mouths.
    """
    theta = np.asarray(theta, dtype=float)
    r = float(radius)
    if r <= 0:
        raise ValueError("radius must be > 0")
    if mouth_radius <= 0:
        raise ValueError("mouth_radius must be > 0")

    if morph is None or morph.target_shape == MorphTargetShape.KEEP:
        return r * np.cos(theta), r * np.sin(theta)

    morph.validate()
    if morph.profile_mode == "axes":
        if throat_radius is None:
            raise ValueError("throat_radius is required when morph.profile_mode='axes'")
        throat_r = float(throat_radius)
        if throat_r <= 0:
            raise ValueError("throat_radius must be > 0")
        denom = float(mouth_radius) - throat_r
        if denom <= 0:
            raise ValueError("mouth_radius must be > throat_radius for axes morphing")

        # Progress parameter derived from scalar profile radius (0 at throat, 1 at mouth).
        s = (float(r) - throat_r) / denom
        s = float(np.clip(s, 0.0, 1.0))
        timing_is_default = (
            float(morph.fixed_part) == 0.0
            and float(morph.end_part) == 1.0
            and float(morph.rate) == 3.0
        )
        u_shape = s if timing_is_default else _blend_u(s, rate=morph.rate, fixed_part=morph.fixed_part, end_part=morph.end_part)

        # Target half-dimensions at the mouth (defaults to an axisymmetric mouth).
        a_mouth = (morph.target_width / 2.0) if morph.target_width > 0 else float(mouth_radius)
        b_mouth = (morph.target_height / 2.0) if morph.target_height > 0 else float(mouth_radius)
        a_mouth = float(a_mouth)
        b_mouth = float(b_mouth)
        if not morph.allow_shrinkage:
            a_mouth = max(a_mouth, throat_r)
            b_mouth = max(b_mouth, throat_r)

        # Interpolate dimensions monotonically from throat circle to mouth dims.
        #
        # Important: use `s` (radius progress) for dimensional growth so the "fixed circular"
        # region (u_shape=0) can still expand without creating an unintended cylinder.
        a = throat_r + s * (a_mouth - throat_r)
        b = throat_r + s * (b_mouth - throat_r)
        a = float(a)
        b = float(b)

        # Use a circle baseline that never exceeds the intended minor axis, so we don't
        # "shrink" from a larger circle to a smaller vertical dimension near the mouth.
        r_circle = float(min(a, b))
        # Important: corner-refined sampling for a rounded-rectangle mouth can produce
        # strongly non-uniform `theta` (many samples cluster in a small ray-angle range).
        # This is desirable near the mouth, but can over-refine the circular throat
        # region (u_shape≈0) and explode element counts in quick iteration presets.
        #
        # Blend from uniform sampling at the throat toward the caller-provided sampling
        # as the morph shape blends in.
        theta_uniform = np.linspace(0.0, 2 * np.pi, theta.size, endpoint=False)
        theta_eff = (1.0 - float(u_shape)) * theta_uniform + float(u_shape) * theta

        x_circle = r_circle * np.cos(theta_eff)
        y_circle = r_circle * np.sin(theta_eff)

        # If the shape blend hasn't started, keep a circular cross-section (but respect r_circle).
        if u_shape <= 0.0:
            return x_circle, y_circle

        # Corner radius grows from 0 to the requested mouth corner radius as the shape blends in.
        corner_here = float(getattr(morph, "corner_radius", 0.0)) * float(u_shape)

        if morph.target_shape == MorphTargetShape.ELLIPSE:
            c = np.cos(theta)
            s0 = np.sin(theta)
            denom_e = (c / a) ** 2 + (s0 / b) ** 2
            denom_e = np.maximum(denom_e, 1e-30)
            r_target = 1.0 / np.sqrt(denom_e)
            x_target = r_target * np.cos(theta)
            y_target = r_target * np.sin(theta)
        elif morph.target_shape == MorphTargetShape.RECTANGLE:
            x_target, y_target = rounded_rectangle_xy(theta_eff, a, b, corner_radius=corner_here)
        elif morph.target_shape == MorphTargetShape.SUPERELLIPSE:
            n = float(morph.superellipse_n)  # validated
            c = np.cos(theta)
            s0 = np.sin(theta)
            denom_se = (np.abs(c) / a) ** n + (np.abs(s0) / b) ** n
            denom_se = np.maximum(denom_se, 1e-30)
            r_target = denom_se ** (-1.0 / n)
            x_target = r_target * np.cos(theta)
            y_target = r_target * np.sin(theta)
        elif morph.target_shape == MorphTargetShape.SUPERFORMULA:
            sf = morph.superformula
            if sf is None:
                raise ValueError("SUPERFORMULA target requires morph.superformula")
            sf.validate()
            a_sf, b_sf, m1, m2, n1, n2, n3 = sf.resolved()

            width_mouth = float(sf.width) if float(sf.width) > 0 else float(morph.target_width)
            if width_mouth <= 0:
                width_mouth = 2.0 * float(mouth_radius)
            aspect = float(sf.aspect_ratio)
            if float(morph.target_width) > 0 and float(morph.target_height) > 0 and float(sf.width) <= 0 and np.isclose(aspect, 1.0):
                aspect = float(morph.target_width) / float(morph.target_height)

            # Scale the superformula curve to match the *current* intended width (2a) and aspect.
            x0, y0 = superformula_xy(theta, a=a_sf, b=b_sf, m1=m1, m2=m2, n1=n1, n2=n2, n3=n3)
            x_target, y_target = _scale_to_width_aspect(x0, y0, width=2.0 * float(a), aspect_ratio=float(aspect))
        else:
            x_target, y_target = superellipse_xy(theta, a, b, 2.0)

        x = (1.0 - u_shape) * x_circle + u_shape * x_target
        y = (1.0 - u_shape) * y_circle + u_shape * y_target
        return x, y

    if morph.profile_mode == "radial":
        if throat_radius is None:
            raise ValueError("throat_radius is required when morph.profile_mode='radial'")
        throat_r = float(throat_radius)
        if throat_r <= 0:
            raise ValueError("throat_radius must be > 0")
        denom = float(mouth_radius) - throat_r
        if denom <= 0:
            raise ValueError("mouth_radius must be > throat_radius for radial morphing")

        # Normalized expansion factor, derived from the scalar profile radius.
        s = (float(r) - throat_r) / denom
        s = float(np.clip(s, 0.0, 1.0))
        timing_is_default = (
            float(morph.fixed_part) == 0.0
            and float(morph.end_part) == 1.0
            and float(morph.rate) == 3.0
        )
        u = s if timing_is_default else _blend_u(s, rate=morph.rate, fixed_part=morph.fixed_part, end_part=morph.end_part)

        # If the morph has not started yet, keep a circular cross-section but still
        # respect the axial profile radius `r` (avoid an unintended cylindrical segment).
        if u <= 0.0:
            return r * np.cos(theta), r * np.sin(theta)

        # Target half-dimensions at the mouth.
        a_mouth = (morph.target_width / 2.0) if morph.target_width > 0 else float(mouth_radius)
        b_mouth = (morph.target_height / 2.0) if morph.target_height > 0 else float(mouth_radius)
        a_mouth = float(a_mouth)
        b_mouth = float(b_mouth)

        # Scale the target to the current profile radius (like "blend" mode does).
        scale = float(r) / float(mouth_radius)
        a = float(a_mouth) * scale
        b = float(b_mouth) * scale

        # Directional support at the mouth for each direction angle in `theta`.
        # This produces a star-shaped radial function R_mouth(theta) that we can
        # interpolate monotonically along the axis.
        if morph.target_shape == MorphTargetShape.ELLIPSE:
            # Ray intersection for an ellipse aligned to axes (at this slice).
            c = np.cos(theta)
            s0 = np.sin(theta)
            denom_e = (c / a) ** 2 + (s0 / b) ** 2
            denom_e = np.maximum(denom_e, 1e-30)
            r_target = 1.0 / np.sqrt(denom_e)
        elif morph.target_shape == MorphTargetShape.RECTANGLE:
            x_m, y_m = rounded_rectangle_xy(theta, a, b, corner_radius=float(morph.corner_radius) * scale)
            r_target = np.sqrt(x_m**2 + y_m**2)
        elif morph.target_shape == MorphTargetShape.SUPERELLIPSE:
            n = float(morph.superellipse_n)  # validated
            c = np.cos(theta)
            s0 = np.sin(theta)
            denom_se = (np.abs(c) / a) ** n + (np.abs(s0) / b) ** n
            denom_se = np.maximum(denom_se, 1e-30)
            r_target = denom_se ** (-1.0 / n)
        elif morph.target_shape == MorphTargetShape.SUPERFORMULA:
            sf = morph.superformula
            if sf is None:
                raise ValueError("SUPERFORMULA target requires morph.superformula")
            sf.validate()
            a_sf, b_sf, m1, m2, n1, n2, n3 = sf.resolved()

            # Generate mouth boundary points (scaled) and compute support via max dot.
            width_mouth = float(sf.width) if float(sf.width) > 0 else float(morph.target_width)
            if width_mouth <= 0:
                width_mouth = 2.0 * float(mouth_radius)
            aspect = float(sf.aspect_ratio)
            if float(morph.target_width) > 0 and float(morph.target_height) > 0 and float(sf.width) <= 0 and np.isclose(aspect, 1.0):
                aspect = float(morph.target_width) / float(morph.target_height)

            phi = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
            x0, y0 = superformula_xy(phi, a=a_sf, b=b_sf, m1=m1, m2=m2, n1=n1, n2=n2, n3=n3)
            x_m, y_m = _scale_to_width_aspect(x0, y0, width=float(width_mouth) * scale, aspect_ratio=float(aspect))

            # Support in each direction theta: max(x cos + y sin).
            dirs = np.vstack([np.cos(theta), np.sin(theta)])  # (2, n_dir)
            pts = np.vstack([x_m, y_m])  # (2, n_pts)
            r_target = np.max(dirs.T @ pts, axis=1)
        else:
            r_target = np.full_like(theta, float(r))

        # If shrinkage is disallowed, ensure we never go below the current circular
        # radius at this slice (prevents any local contraction due to timing).
        if not morph.allow_shrinkage:
            r_target = np.maximum(r_target, float(r))

        R = (1.0 - u) * float(r) + u * r_target
        return R * np.cos(theta), R * np.sin(theta)

    u = _blend_u(float(t), rate=morph.rate, fixed_part=morph.fixed_part, end_part=morph.end_part)
    if u <= 0.0:
        return r * np.cos(theta), r * np.sin(theta)

    a_mouth = (morph.target_width / 2.0) if morph.target_width > 0 else float(mouth_radius)
    b_mouth = (morph.target_height / 2.0) if morph.target_height > 0 else float(mouth_radius)
    scale = r / float(mouth_radius)
    a = float(a_mouth) * scale
    b = float(b_mouth) * scale

    if not morph.allow_shrinkage:
        inscribed = float(min(a, b))
        if inscribed > 0 and inscribed < r:
            grow = r / inscribed
            a *= grow
            b *= grow

    if morph.target_shape == MorphTargetShape.ELLIPSE:
        c = np.cos(theta)
        s0 = np.sin(theta)
        denom_e = (c / a) ** 2 + (s0 / b) ** 2
        denom_e = np.maximum(denom_e, 1e-30)
        r_target = 1.0 / np.sqrt(denom_e)
        x_target = r_target * np.cos(theta)
        y_target = r_target * np.sin(theta)
        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)
        x = (1.0 - u) * x_circle + u * x_target
        y = (1.0 - u) * y_circle + u * y_target
        return x, y
    elif morph.target_shape == MorphTargetShape.RECTANGLE:
        x_target, y_target = rounded_rectangle_xy(
            theta,
            a,
            b,
            corner_radius=float(morph.corner_radius) * scale,
        )
        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)
        x = (1.0 - u) * x_circle + u * x_target
        y = (1.0 - u) * y_circle + u * y_target
        return x, y
    elif morph.target_shape == MorphTargetShape.SUPERELLIPSE:
        n = float(morph.superellipse_n)  # validated
        c = np.cos(theta)
        s0 = np.sin(theta)
        denom_se = (np.abs(c) / a) ** n + (np.abs(s0) / b) ** n
        denom_se = np.maximum(denom_se, 1e-30)
        r_target = denom_se ** (-1.0 / n)
        x_target = r_target * np.cos(theta)
        y_target = r_target * np.sin(theta)
        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)
        x = (1.0 - u) * x_circle + u * x_target
        y = (1.0 - u) * y_circle + u * y_target
        return x, y
    elif morph.target_shape == MorphTargetShape.SUPERFORMULA:
        sf = morph.superformula
        if sf is None:
            raise ValueError("SUPERFORMULA target requires morph.superformula")
        sf.validate()
        a_sf, b_sf, m1, m2, n1, n2, n3 = sf.resolved()

        width_mouth = float(sf.width) if float(sf.width) > 0 else float(morph.target_width)
        if width_mouth <= 0:
            width_mouth = 2.0 * float(mouth_radius)
        aspect = float(sf.aspect_ratio)
        if float(morph.target_width) > 0 and float(morph.target_height) > 0 and float(sf.width) <= 0 and np.isclose(aspect, 1.0):
            aspect = float(morph.target_width) / float(morph.target_height)

        width_here = width_mouth * scale
        x0, y0 = superformula_xy(theta, a=a_sf, b=b_sf, m1=m1, m2=m2, n1=n1, n2=n2, n3=n3)
        x_target, y_target = _scale_to_width_aspect(x0, y0, width=width_here, aspect_ratio=aspect)

        if not morph.allow_shrinkage:
            inscribed = float(min(np.max(np.abs(x_target)), np.max(np.abs(y_target))))
            if inscribed > 0 and inscribed < r:
                grow = r / inscribed
                x_target *= grow
                y_target *= grow

        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)
        x = (1.0 - u) * x_circle + u * x_target
        y = (1.0 - u) * y_circle + u * y_target
        return x, y
    else:
        # Default to ellipse-like blend.
        x_target, y_target = superellipse_xy(theta, a, b, 2.0)
        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)
        x = (1.0 - u) * x_circle + u * x_target
        y = (1.0 - u) * y_circle + u * y_target
        return x, y
