"""Waveguide numeric profile checks and tuning (pure; no Gmsh required)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from bempp_audio.mesh.profiles import (
    conical_profile,
    exponential_profile,
    tractrix_profile,
    tractrix_horn_profile,
    os_profile,
    hyperbolic_profile,
    cts_profile,
)
from bempp_audio.mesh.morph import (
    MorphTargetShape,
    MorphConfig,
    morphed_cross_section_xy,
    theta_with_corner_refinement,
    apply_noncontracting_axis_scaling,
    apply_noncontracting_direction_scaling,
)


@dataclass(frozen=True)
class WaveguideProfileCheck:
    """Numeric profile check results (pure; no Gmsh required)."""

    ok: bool
    min_dr: float
    n_r_violations: int
    min_support_delta: Optional[float]
    n_support_violations: int
    worst_direction_deg: Optional[float]
    min_area_delta: Optional[float] = None
    n_area_violations: int = 0
    max_scale_x: Optional[float] = None
    max_scale_y: Optional[float] = None
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MouthDiffractionProxy:
    """
    Pure-geometry proxy metric for likely mouth-edge diffraction severity.

    This is intentionally cheap (no Gmsh/BEM required) and meant for design-time
    screening and optimization/sweeps.
    """

    mouth_angle_deg: float
    tangency_error_deg: float
    curvature_radius_m: float
    risk_score_0_100: float
    notes: Tuple[str, ...] = ()


def mouth_diffraction_proxy(
    config: "WaveguideMeshConfig",
    *,
    n_axial: int = 250,
    fit_points: int = 9,
) -> MouthDiffractionProxy:
    """
    Compute a simple mouth-diffraction proxy from the meridian curve `r(x)`.

    What it measures
    ---------------
    - `mouth_angle_deg`: wall tangent angle relative to the axis at the mouth.
      `90°` means perfectly tangent to the baffle plane; smaller angles imply a
      "sharper" termination and higher diffraction risk.
    - `curvature_radius_m`: radius of curvature of the meridian at the mouth.
      Smaller radii indicate a tighter bend near the termination.

    Notes
    -----
    This is not a full diffraction model. It is a proxy that is useful for
    ranking candidates in parameter sweeps before running expensive BEM solves.
    """
    cfg = config
    cfg.validate()

    n_axial = int(n_axial)
    fit_points = int(fit_points)
    if n_axial < 10:
        raise ValueError("n_axial must be >= 10")
    if fit_points < 5 or fit_points > n_axial:
        raise ValueError("fit_points must be in [5, n_axial]")

    x = np.linspace(0.0, float(cfg.length), n_axial)
    throat_r = float(cfg.throat_diameter) / 2.0
    mouth_r = float(cfg.mouth_diameter) / 2.0

    if cfg.profile_type == "conical":
        r = conical_profile(x, throat_r, mouth_r, float(cfg.length))
    elif cfg.profile_type == "exponential":
        r = exponential_profile(
            x,
            throat_r,
            mouth_r,
            float(cfg.length),
            flare_constant=cfg.flare_constant if cfg.flare_constant > 0 else None,
        )
    elif cfg.profile_type == "tractrix":
        r = tractrix_profile(x, throat_r, mouth_r, float(cfg.length))
    elif cfg.profile_type == "tractrix_horn":
        r = tractrix_horn_profile(x, throat_r, mouth_r, float(cfg.length))
    elif cfg.profile_type in ("os", "oblate_spheroidal"):
        r = os_profile(
            x,
            throat_r,
            mouth_r,
            float(cfg.length),
            opening_angle_deg=getattr(cfg, "os_opening_angle_deg", None),
        )
    elif cfg.profile_type == "hyperbolic":
        r = hyperbolic_profile(x, throat_r, mouth_r, float(cfg.length), cfg.hyperbolic_sharpness)
    elif cfg.profile_type == "cts":
        r = cts_profile(
            x,
            throat_r,
            mouth_r,
            float(cfg.length),
            throat_blend=cfg.cts_throat_blend,
            transition=cfg.cts_transition,
            driver_exit_angle_deg=cfg.cts_driver_exit_angle_deg,
            throat_angle_deg=cfg.cts_throat_angle_deg,
            tangency=cfg.cts_tangency,
            mouth_roll=getattr(cfg, "cts_mouth_roll", 0.0),
            curvature_regularizer=getattr(cfg, "cts_curvature_regularizer", 1.0),
            mid_curvature=getattr(cfg, "cts_mid_curvature", 0.0),
        )
    else:
        raise ValueError(f"Unknown profile type: {cfg.profile_type}")

    # Fit a quadratic to the final segment to estimate slope and curvature at the mouth.
    xs = np.asarray(x[-fit_points:], dtype=float)
    rs = np.asarray(r[-fit_points:], dtype=float)
    L = float(cfg.length)

    # Shift x for conditioning.
    x0 = xs - float(L)
    # r ≈ a*x0^2 + b*x0 + c
    a, b, _ = np.polyfit(x0, rs, deg=2)
    dr_dx = float(b)  # derivative at x0=0
    d2r_dx2 = float(2.0 * a)

    mouth_angle = float(np.degrees(np.arctan(max(0.0, dr_dx))))
    tangency_error = float(max(0.0, 90.0 - mouth_angle))

    # Curvature κ = |r''| / (1 + r'^2)^(3/2)
    denom = float((1.0 + dr_dx * dr_dx) ** 1.5)
    kappa = float(abs(d2r_dx2) / denom) if denom > 0 else 0.0
    curvature_radius = float(1.0 / kappa) if kappa > 1e-18 else float("inf")

    # Simple heuristic risk score.
    # - Tangency error dominates (bigger error -> bigger risk).
    # - Tight curvature (< ~50mm) adds penalty.
    radius_mm = curvature_radius * 1e3
    curv_pen = 0.0
    if np.isfinite(radius_mm) and radius_mm > 0:
        curv_pen = float(max(0.0, (50.0 - radius_mm) / 50.0)) * 30.0
    risk = float(min(100.0, 1.2 * tangency_error + curv_pen))

    notes: list[str] = []
    if tangency_error > 10.0:
        notes.append("Large tangency error at mouth; expect stronger edge diffraction.")
    if np.isfinite(radius_mm) and radius_mm < 25.0:
        notes.append("Tight curvature near mouth termination; watch for diffraction/ringing.")

    return MouthDiffractionProxy(
        mouth_angle_deg=mouth_angle,
        tangency_error_deg=tangency_error,
        curvature_radius_m=curvature_radius,
        risk_score_0_100=risk,
        notes=tuple(notes),
    )


def check_profile(
    config: "WaveguideMeshConfig",
    *,
    n_axial: int = 200,
    n_directions: int = 36,
    tol: float = 0.0,
    theta_mode: str = "dense",
) -> WaveguideProfileCheck:
    """
    Check that a waveguide profile is expanding (monotone non-decreasing).

    Checks
    ------
    1) Axisymmetric radius `r(x)` is non-decreasing along the axis.
    2) If `config.morph` is enabled, also checks:
       - the *support radius* in multiple lateral directions is non-decreasing
       - the cross-sectional polygon area is non-decreasing
    """
    cfg = config
    cfg.validate()

    effective_tol = max(float(tol), 1e-12)

    n_axial = int(n_axial)
    n_directions = int(n_directions)
    if n_axial < 3:
        raise ValueError("n_axial must be >= 3")
    if n_directions < 4:
        raise ValueError("n_directions must be >= 4")
    if str(theta_mode) not in ("dense", "mesh"):
        raise ValueError("theta_mode must be 'dense' or 'mesh'")

    x = np.linspace(0.0, float(cfg.length), n_axial)

    throat_r = float(cfg.throat_diameter) / 2.0
    mouth_r = float(cfg.mouth_diameter) / 2.0

    if cfg.profile_type == "conical":
        r = conical_profile(x, throat_r, mouth_r, float(cfg.length))
    elif cfg.profile_type == "exponential":
        r = exponential_profile(
            x,
            throat_r,
            mouth_r,
            float(cfg.length),
            flare_constant=cfg.flare_constant if cfg.flare_constant > 0 else None,
        )
    elif cfg.profile_type == "tractrix":
        r = tractrix_profile(x, throat_r, mouth_r, float(cfg.length))
    elif cfg.profile_type == "tractrix_horn":
        r = tractrix_horn_profile(x, throat_r, mouth_r, float(cfg.length))
    elif cfg.profile_type in ("os", "oblate_spheroidal"):
        r = os_profile(
            x,
            throat_r,
            mouth_r,
            float(cfg.length),
            opening_angle_deg=getattr(cfg, "os_opening_angle_deg", None),
        )
    elif cfg.profile_type == "hyperbolic":
        r = hyperbolic_profile(x, throat_r, mouth_r, float(cfg.length), cfg.hyperbolic_sharpness)
    elif cfg.profile_type == "cts":
        r = cts_profile(
            x,
            throat_r,
            mouth_r,
            float(cfg.length),
            throat_blend=cfg.cts_throat_blend,
            transition=cfg.cts_transition,
            driver_exit_angle_deg=cfg.cts_driver_exit_angle_deg,
            throat_angle_deg=cfg.cts_throat_angle_deg,
            tangency=cfg.cts_tangency,
            mouth_roll=getattr(cfg, "cts_mouth_roll", 0.0),
            curvature_regularizer=getattr(cfg, "cts_curvature_regularizer", 1.0),
            mid_curvature=getattr(cfg, "cts_mid_curvature", 0.0),
        )
    else:
        raise ValueError(f"Unknown profile type: {cfg.profile_type}")

    dr = np.diff(r)
    n_r_bad = int(np.sum(dr < -effective_tol))
    min_dr = float(np.min(dr)) if dr.size else 0.0

    min_support_delta: Optional[float] = None
    n_support_bad = 0
    worst_deg: Optional[float] = None
    min_area_delta: Optional[float] = None
    n_area_bad = 0
    max_scale_x: Optional[float] = None
    max_scale_y: Optional[float] = None
    notes: list[str] = []

    morph = cfg.morph
    if morph is not None and morph.target_shape != MorphTargetShape.KEEP:
        if str(theta_mode) == "mesh":
            n_circ = int(cfg.n_circumferential)
            theta = np.linspace(0.0, 2 * np.pi, n_circ, endpoint=False)
            if (
                morph.target_shape == MorphTargetShape.RECTANGLE
                and float(getattr(morph, "corner_radius", 0.0)) > 0
                and int(getattr(cfg, "corner_resolution", 0)) > 0
                and n_circ >= 8
            ):
                a = float(getattr(morph, "target_width", 0.0) or (2.0 * float(mouth_r))) / 2.0
                b = float(getattr(morph, "target_height", 0.0) or (2.0 * float(mouth_r))) / 2.0
                theta = theta_with_corner_refinement(
                    n_total=n_circ,
                    a=float(a),
                    b=float(b),
                    rc=float(getattr(morph, "corner_radius", 0.0)),
                    corner_points=int(getattr(cfg, "corner_resolution", 0)),
                )
        else:
            theta = np.linspace(0.0, 2 * np.pi, 4 * n_directions, endpoint=False)  # boundary samples
        dirs = np.linspace(0.0, 2 * np.pi, n_directions, endpoint=False)  # support directions
        supports = np.zeros((n_directions, n_axial), dtype=float)
        areas = np.zeros(n_axial, dtype=float)

        prev_x = None
        prev_y = None
        prev_support = None
        sx_max = 1.0
        sy_max = 1.0
        s_iso_max = 1.0

        for i, (xi, ri) in enumerate(zip(x, r)):
            t = float(xi) / float(cfg.length)
            xs, ys = morphed_cross_section_xy(
                theta,
                t=t,
                radius=float(ri),
                throat_radius=float(throat_r),
                mouth_radius=mouth_r,
                morph=morph,
            )
            if getattr(morph, "enforce_noncontracting", False):
                mode = str(getattr(morph, "enforce_mode", "axes"))
                if mode == "directions":
                    xs, ys, prev_support, s_iso = apply_noncontracting_direction_scaling(
                        xs,
                        ys,
                        prev_support=prev_support,
                        n_directions=int(getattr(morph, "enforce_n_directions", n_directions)),
                        tol=float(getattr(morph, "enforce_tol", 0.0)),
                    )
                    s_iso_max = max(s_iso_max, float(s_iso))
                else:
                    xs, ys, prev_x, prev_y, sx, sy = apply_noncontracting_axis_scaling(
                        xs,
                        ys,
                        prev_x=prev_x,
                        prev_y=prev_y,
                        axes=tuple(getattr(morph, "enforce_axes", ("y",))),
                        tol=float(getattr(morph, "enforce_tol", 0.0)),
                    )
                    sx_max = max(sx_max, float(sx))
                    sy_max = max(sy_max, float(sy))
            for j, ang in enumerate(dirs):
                dx = float(np.cos(ang))
                dy = float(np.sin(ang))
                supports[j, i] = float(np.max(xs * dx + ys * dy))
            x0 = np.asarray(xs, dtype=float)
            y0 = np.asarray(ys, dtype=float)
            areas[i] = 0.5 * float(abs(np.dot(x0, np.roll(y0, -1)) - np.dot(y0, np.roll(x0, -1))))

        ds = np.diff(supports, axis=1)
        bad = ds < -effective_tol
        n_support_bad = int(np.sum(bad))
        if ds.size:
            min_support_delta = float(np.min(ds))
            j = int(np.unravel_index(int(np.argmin(ds)), ds.shape)[0])
            worst_deg = float(np.degrees(dirs[j]))
        if n_support_bad:
            notes.append(
                "Morph causes local contraction in at least one lateral direction; "
                "consider enabling `morph.enforce_noncontracting`, reducing shrinkage, or adjusting morph timing."
            )

        dA = np.diff(areas)
        n_area_bad = int(np.sum(dA < -effective_tol))
        if dA.size:
            min_area_delta = float(np.min(dA))
        if n_area_bad:
            notes.append("Cross-sectional area decreases locally; consider adjusting morph mapping or enforcement.")

        if getattr(morph, "enforce_noncontracting", False):
            mode = str(getattr(morph, "enforce_mode", "axes"))
            if mode == "directions":
                max_scale_x = float(s_iso_max)
                max_scale_y = float(s_iso_max)
            else:
                max_scale_x = float(sx_max)
                max_scale_y = float(sy_max)

    ok = (n_r_bad == 0) and (n_support_bad == 0) and (n_area_bad == 0)
    if n_r_bad:
        notes.append("Axisymmetric radius is not monotone increasing; adjust profile parameters.")

    return WaveguideProfileCheck(
        ok=ok,
        min_dr=min_dr,
        n_r_violations=n_r_bad,
        min_support_delta=min_support_delta,
        n_support_violations=n_support_bad,
        worst_direction_deg=worst_deg,
        min_area_delta=min_area_delta,
        n_area_violations=n_area_bad,
        max_scale_x=max_scale_x,
        max_scale_y=max_scale_y,
        notes=tuple(notes),
    )


def auto_tune_morph_for_expansion(
    config: "WaveguideMeshConfig",
    *,
    n_axial: int = 200,
    n_directions: int = 36,
    tol: float = 0.0,
    fixed_part_grid: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3),
    rate_grid: Tuple[float, ...] = (0.3, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0),
) -> Tuple["WaveguideMeshConfig", WaveguideProfileCheck]:
    """
    Try to reduce “vase/waist” behavior by tuning morph blend parameters.

    This keeps target dimensions fixed and only adjusts morph timing
    (`fixed_part`, `rate`). Returns the best configuration found, even if it
    still has violations.
    """
    cfg = config
    cfg.validate()
    morph = cfg.morph
    if morph is None or morph.target_shape == MorphTargetShape.KEEP:
        return cfg, check_profile(cfg, n_axial=n_axial, n_directions=n_directions, tol=tol)

    best_cfg = cfg
    best_report = check_profile(cfg, n_axial=n_axial, n_directions=n_directions, tol=tol)

    def score(rep: WaveguideProfileCheck) -> Tuple[int, int, float]:
        support_min = rep.min_support_delta if rep.min_support_delta is not None else 0.0
        return (rep.n_support_violations, rep.n_r_violations, -support_min)

    best_score = score(best_report)

    for fixed in fixed_part_grid:
        for rate in rate_grid:
            tuned_morph = MorphConfig(
                target_shape=morph.target_shape,
                target_width=morph.target_width,
                target_height=morph.target_height,
                corner_radius=morph.corner_radius,
                fixed_part=float(fixed),
                end_part=morph.end_part,
                rate=float(rate),
                allow_shrinkage=morph.allow_shrinkage,
                profile_mode=morph.profile_mode,
                enforce_noncontracting=morph.enforce_noncontracting,
                enforce_mode=morph.enforce_mode,
                enforce_axes=morph.enforce_axes,
                enforce_n_directions=morph.enforce_n_directions,
                enforce_tol=morph.enforce_tol,
                superellipse_n=morph.superellipse_n,
                superformula=morph.superformula,
            )
            tuned = type(cfg)(**{**cfg.__dict__, "morph": tuned_morph})
            rep = check_profile(tuned, n_axial=n_axial, n_directions=n_directions, tol=tol)
            sc = score(rep)
            if sc < best_score:
                best_score = sc
                best_cfg = tuned
                best_report = rep
                if best_report.ok:
                    return best_cfg, best_report

    return best_cfg, best_report
