"""Waveguide profile export utilities (pure; no Gmsh required)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from bempp_audio.mesh.morph import (
    MorphTargetShape,
    apply_noncontracting_axis_scaling,
    apply_noncontracting_direction_scaling,
    morphed_cross_section_xy,
    theta_for_morph,
)
from bempp_audio.mesh.profiles import (
    conical_profile,
    cts_profile,
    exponential_profile,
    hyperbolic_profile,
    os_profile,
    tractrix_horn_profile,
    tractrix_profile,
)
from bempp_audio.mesh.waveguide import WaveguideMeshConfig


@dataclass(frozen=True)
class ProfileExportRow:
    x: float
    z: float
    r_profile: float

    x_xz_raw: float
    y_yz_raw: float
    x_support_raw: float
    y_support_raw: float
    area_raw: float

    x_xz_enforced: Optional[float]
    y_yz_enforced: Optional[float]
    x_support_enforced: Optional[float]
    y_support_enforced: Optional[float]
    area_enforced: Optional[float]

    scale_x: Optional[float]
    scale_y: Optional[float]
    scale_iso: Optional[float]


def _shoelace_area(xs: np.ndarray, ys: np.ndarray) -> float:
    xs = np.asarray(xs, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    if xs.size < 3:
        return 0.0
    return 0.5 * float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))


def _profile_radius(cfg: WaveguideMeshConfig, x: np.ndarray) -> np.ndarray:
    throat_r = float(cfg.throat_diameter) / 2.0
    mouth_r = float(cfg.mouth_diameter) / 2.0
    length = float(cfg.length)

    if cfg.profile_type == "conical":
        return conical_profile(x, throat_r, mouth_r, length)
    if cfg.profile_type == "exponential":
        return exponential_profile(
            x,
            throat_r,
            mouth_r,
            length,
            flare_constant=cfg.flare_constant if cfg.flare_constant > 0 else None,
        )
    if cfg.profile_type == "tractrix":
        return tractrix_profile(x, throat_r, mouth_r, length)
    if cfg.profile_type == "tractrix_horn":
        return tractrix_horn_profile(x, throat_r, mouth_r, length)
    if cfg.profile_type in ("os", "oblate_spheroidal"):
        return os_profile(x, throat_r, mouth_r, length, opening_angle_deg=cfg.os_opening_angle_deg)
    if cfg.profile_type == "hyperbolic":
        return hyperbolic_profile(x, throat_r, mouth_r, length, cfg.hyperbolic_sharpness)
    if cfg.profile_type == "cts":
        return cts_profile(
            x,
            throat_r,
            mouth_r,
            length,
            throat_blend=cfg.cts_throat_blend,
            transition=cfg.cts_transition,
            driver_exit_angle_deg=cfg.cts_driver_exit_angle_deg,
            throat_angle_deg=cfg.cts_throat_angle_deg,
            tangency=cfg.cts_tangency,
        )
    raise ValueError(f"Unknown profile_type={cfg.profile_type!r}")


def export_profile_csv(
    cfg: WaveguideMeshConfig,
    *,
    out_csv: str | Path,
    n_axial: int = 200,
    theta_samples: Optional[int] = None,
    include_enforced: bool = True,
) -> Path:
    """Export raw/enforced cross-section extents along the axis to a CSV file.

    The CSV is intended for diagnosing:
    - short-axis contraction (e.g. in XZ and YZ principal cuts)
    - whether enforcement scaling changes mouth dimensions
    - termination slope near the baffle plane (z=0)
    """
    cfg.validate()

    length = float(cfg.length)
    throat_r = float(cfg.throat_diameter) / 2.0
    mouth_r = float(cfg.mouth_diameter) / 2.0

    n_axial = int(n_axial)
    if n_axial < 3:
        raise ValueError("n_axial must be >= 3")

    x = np.linspace(0.0, length, n_axial)
    z = -length + x
    r = _profile_radius(cfg, x)

    n_theta = int(theta_samples) if theta_samples is not None else int(cfg.n_circumferential)
    theta = theta_for_morph(
        n_total=max(n_theta, 8),
        mouth_radius=mouth_r,
        morph=cfg.morph,
        corner_resolution=int(cfg.corner_resolution),
    )

    morph = cfg.morph
    do_morph = morph is not None and morph.target_shape != MorphTargetShape.KEEP

    prev_x = None
    prev_y = None
    prev_support = None

    rows: list[ProfileExportRow] = []
    for xi, zi, ri in zip(x, z, r):
        t = float(xi) / length

        if do_morph:
            xs_raw, ys_raw = morphed_cross_section_xy(
                theta,
                t=t,
                radius=float(ri),
                throat_radius=throat_r,
                mouth_radius=mouth_r,
                morph=morph,
            )
        else:
            xs_raw = float(ri) * np.cos(theta)
            ys_raw = float(ri) * np.sin(theta)

        # Principal-cut radii are best evaluated at exact cardinal angles.
        theta_xz = np.array([0.0, float(np.pi)], dtype=float)
        theta_yz = np.array([float(np.pi / 2), float(3 * np.pi / 2)], dtype=float)
        if do_morph:
            xz_raw, _ = morphed_cross_section_xy(
                theta_xz,
                t=t,
                radius=float(ri),
                throat_radius=throat_r,
                mouth_radius=mouth_r,
                morph=morph,
            )
            _, yz_raw = morphed_cross_section_xy(
                theta_yz,
                t=t,
                radius=float(ri),
                throat_radius=throat_r,
                mouth_radius=mouth_r,
                morph=morph,
            )
        else:
            xz_raw = float(ri) * np.cos(theta_xz)
            yz_raw = float(ri) * np.sin(theta_yz)

        x_xz_raw = float(np.max(np.abs(xz_raw)))
        y_yz_raw = float(np.max(np.abs(yz_raw)))
        x_support_raw = float(np.max(np.abs(xs_raw)))
        y_support_raw = float(np.max(np.abs(ys_raw)))
        area_raw = _shoelace_area(xs_raw, ys_raw)

        x_xz_enforced = None
        y_yz_enforced = None
        x_support_enforced = None
        y_support_enforced = None
        area_enforced = None
        scale_x = None
        scale_y = None
        scale_iso = None

        if include_enforced and do_morph and bool(getattr(morph, "enforce_noncontracting", False)):
            mode = str(getattr(morph, "enforce_mode", "axes"))
            if mode == "directions":
                xs2, ys2, prev_support, s_iso = apply_noncontracting_direction_scaling(
                    xs_raw,
                    ys_raw,
                    prev_support=prev_support,
                    n_directions=int(getattr(morph, "enforce_n_directions", int(cfg.n_circumferential))),
                    tol=float(getattr(morph, "enforce_tol", 0.0)),
                )
                scale_iso = float(s_iso)
            else:
                xs2, ys2, prev_x, prev_y, sx, sy = apply_noncontracting_axis_scaling(
                    xs_raw,
                    ys_raw,
                    prev_x=prev_x,
                    prev_y=prev_y,
                    axes=tuple(getattr(morph, "enforce_axes", ("y",))),
                    tol=float(getattr(morph, "enforce_tol", 0.0)),
                )
                scale_x = float(sx)
                scale_y = float(sy)

            # Apply same scale to the principal-cut values.
            if mode == "directions":
                x_xz_enforced = x_xz_raw * float(scale_iso)
                y_yz_enforced = y_yz_raw * float(scale_iso)
            else:
                x_xz_enforced = x_xz_raw * float(scale_x)
                y_yz_enforced = y_yz_raw * float(scale_y)

            x_support_enforced = float(np.max(np.abs(xs2)))
            y_support_enforced = float(np.max(np.abs(ys2)))
            area_enforced = _shoelace_area(xs2, ys2)

        rows.append(
            ProfileExportRow(
                x=float(xi),
                z=float(zi),
                r_profile=float(ri),
                x_xz_raw=x_xz_raw,
                y_yz_raw=y_yz_raw,
                x_support_raw=x_support_raw,
                y_support_raw=y_support_raw,
                area_raw=area_raw,
                x_xz_enforced=x_xz_enforced,
                y_yz_enforced=y_yz_enforced,
                x_support_enforced=x_support_enforced,
                y_support_enforced=y_support_enforced,
                area_enforced=area_enforced,
                scale_x=scale_x,
                scale_y=scale_y,
                scale_iso=scale_iso,
            )
        )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "x",
                "z",
                "r_profile",
                "x_xz_raw",
                "y_yz_raw",
                "x_support_raw",
                "y_support_raw",
                "area_raw",
                "x_xz_enforced",
                "y_yz_enforced",
                "x_support_enforced",
                "y_support_enforced",
                "area_enforced",
                "scale_x",
                "scale_y",
                "scale_iso",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row.x,
                    row.z,
                    row.r_profile,
                    row.x_xz_raw,
                    row.y_yz_raw,
                    row.x_support_raw,
                    row.y_support_raw,
                    row.area_raw,
                    row.x_xz_enforced,
                    row.y_yz_enforced,
                    row.x_support_enforced,
                    row.y_support_enforced,
                    row.area_enforced,
                    row.scale_x,
                    row.scale_y,
                    row.scale_iso,
                ]
            )

    return out_path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export waveguide profile metrics to CSV (pure math).")
    p.add_argument("--config-json", type=str, required=True, help="WaveguideMeshConfig JSON (WaveguideMeshConfig.to_json).")
    p.add_argument("--out", type=str, required=True, help="Output CSV path.")
    p.add_argument("--n-axial", type=int, default=200, help="Axial samples.")
    p.add_argument("--theta-samples", type=int, default=None, help="Boundary theta samples (default: cfg.n_circumferential).")
    p.add_argument("--no-enforced", action="store_true", help="Disable enforcement scaling columns.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ns = _parse_args(argv)
    cfg = WaveguideMeshConfig.from_json(ns.config_json)
    export_profile_csv(
        cfg,
        out_csv=ns.out,
        n_axial=int(ns.n_axial),
        theta_samples=int(ns.theta_samples) if ns.theta_samples is not None else None,
        include_enforced=not bool(ns.no_enforced),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

