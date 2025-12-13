from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from bempp_audio.mesh.morph import (
    MorphTargetShape,
    morphed_cross_section_xy,
    theta_for_morph,
    morphed_sections_xy,
)
from bempp_audio.mesh.profiles import (
    conical_profile,
    cts_profile,
    exponential_profile,
    hyperbolic_profile,
    tractrix_profile,
    tractrix_horn_profile,
    os_profile,
)
from bempp_audio.mesh.waveguide import WaveguideMeshConfig


def _profile_radius(cfg: WaveguideMeshConfig, x: np.ndarray) -> np.ndarray:
    throat_r = float(cfg.throat_diameter) / 2.0
    mouth_r = float(cfg.mouth_diameter) / 2.0
    L = float(cfg.length)

    if cfg.profile_type == "conical":
        return conical_profile(x, throat_r, mouth_r, L)
    if cfg.profile_type == "exponential":
        return exponential_profile(
            x,
            throat_r,
            mouth_r,
            L,
            flare_constant=cfg.flare_constant if float(cfg.flare_constant) > 0 else None,
        )
    if cfg.profile_type == "tractrix":
        return tractrix_profile(x, throat_r, mouth_r, L)
    if cfg.profile_type == "tractrix_horn":
        return tractrix_horn_profile(x, throat_r, mouth_r, L)
    if cfg.profile_type in ("os", "oblate_spheroidal"):
        return os_profile(x, throat_r, mouth_r, L, opening_angle_deg=cfg.os_opening_angle_deg)
    if cfg.profile_type == "hyperbolic":
        return hyperbolic_profile(x, throat_r, mouth_r, L, cfg.hyperbolic_sharpness)
    if cfg.profile_type == "cts":
        return cts_profile(
            x,
            throat_r,
            mouth_r,
            L,
            throat_blend=cfg.cts_throat_blend,
            transition=cfg.cts_transition,
            driver_exit_angle_deg=cfg.cts_driver_exit_angle_deg,
            throat_angle_deg=cfg.cts_throat_angle_deg,
            tangency=cfg.cts_tangency,
        )
    raise ValueError(f"Unknown profile_type: {cfg.profile_type}")


def _theta_for_mesh(cfg: WaveguideMeshConfig) -> np.ndarray:
    n_circ = int(cfg.n_circumferential)
    morph = cfg.morph
    return theta_for_morph(
        n_total=n_circ,
        mouth_radius=float(cfg.mouth_diameter) / 2.0,
        morph=morph,
        corner_resolution=int(getattr(cfg, "corner_resolution", 0)),
    )


def _line_intersections(poly_x: np.ndarray, poly_y: np.ndarray, *, axis: str) -> np.ndarray:
    """Intersect a closed polyline with x=0 (axis='x') or y=0 (axis='y')."""
    x = np.asarray(poly_x, dtype=float)
    y = np.asarray(poly_y, dtype=float)
    if x.shape != y.shape or x.size < 3:
        raise ValueError("poly_x and poly_y must have same shape and >=3 points")

    if axis == "x":
        a0 = x
        a1 = np.roll(x, -1)
        b0 = y
        b1 = np.roll(y, -1)
    elif axis == "y":
        a0 = y
        a1 = np.roll(y, -1)
        b0 = x
        b1 = np.roll(x, -1)
    else:
        raise ValueError("axis must be 'x' or 'y'")

    hits: list[float] = []
    for u0, u1, v0, v1 in zip(a0, a1, b0, b1):
        # If the segment endpoint lies on the axis, include it.
        if abs(u0) <= 1e-14:
            hits.append(float(v0))
            continue
        if abs(u1) <= 1e-14:
            hits.append(float(v1))
            continue
        # Proper crossing.
        if u0 * u1 < 0:
            t = u0 / (u0 - u1)
            hits.append(float(v0 + t * (v1 - v0)))

    if not hits:
        return np.array([], dtype=float)
    # Deduplicate numerically (axis-aligned vertices can be reported twice).
    hits_arr = np.array(sorted(hits), dtype=float)
    keep = [hits_arr[0]]
    for v in hits_arr[1:]:
        if abs(v - keep[-1]) > 1e-12:
            keep.append(v)
    return np.array(keep, dtype=float)


def export_profiles(
    cfg: WaveguideMeshConfig,
    *,
    out_dir: Path,
    n_axial: int = 200,
    theta_mode: str = "mesh",
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    throat_r = float(cfg.throat_diameter) / 2.0
    mouth_r = float(cfg.mouth_diameter) / 2.0
    L = float(cfg.length)

    x_ax = np.linspace(0.0, L, int(n_axial))
    z_ax = -L + x_ax
    r_ax = _profile_radius(cfg, x_ax)

    if theta_mode == "mesh":
        theta = _theta_for_mesh(cfg)
    elif theta_mode == "dense":
        theta = np.linspace(0.0, 2 * np.pi, max(720, int(cfg.n_circumferential) * 10), endpoint=False)
    else:
        raise ValueError("theta_mode must be 'mesh' or 'dense'")

    yz_rows = []
    xz_rows = []
    last_xs = None
    last_ys = None

    morph = cfg.morph
    if morph is None or morph.target_shape == MorphTargetShape.KEEP:
        xs_all = np.array([float(ri) * np.cos(theta) for ri in r_ax], dtype=float)
        ys_all = np.array([float(ri) * np.sin(theta) for ri in r_ax], dtype=float)
    else:
        xs_all, ys_all = morphed_sections_xy(
            x=x_ax,
            r=r_ax,
            length=float(L),
            theta=theta,
            throat_radius=float(throat_r),
            mouth_radius=float(mouth_r),
            morph=morph,
            enforce_n_directions_default=int(getattr(cfg, "n_circumferential", 36)),
            name_prefix="profile_debug cross-section",
        )

    for zi, xs, ys in zip(z_ax, xs_all, ys_all):

        # YZ profile: intersect polygon with x=0.
        y_hits = _line_intersections(xs, ys, axis="x")
        if y_hits.size:
            yz_rows.append((float(zi), float(y_hits.min()), float(y_hits.max()), int(y_hits.size)))
        else:
            yz_rows.append((float(zi), float("nan"), float("nan"), 0))

        # XZ profile: intersect polygon with y=0.
        x_hits = _line_intersections(xs, ys, axis="y")
        if x_hits.size:
            xz_rows.append((float(zi), float(x_hits.min()), float(x_hits.max()), int(x_hits.size)))
        else:
            xz_rows.append((float(zi), float("nan"), float("nan"), 0))

        last_xs = np.asarray(xs, dtype=float)
        last_ys = np.asarray(ys, dtype=float)

    if last_xs is None or last_ys is None:
        raise RuntimeError("no axial samples generated")

    mouth_xy_csv = out_dir / "mouth_xy.csv"
    np.savetxt(mouth_xy_csv, np.c_[last_xs, last_ys], delimiter=",", header="x,y", comments="")

    yz_csv = out_dir / "yz_profile.csv"
    xz_csv = out_dir / "xz_profile.csv"
    np.savetxt(yz_csv, np.array(yz_rows), delimiter=",", header="z,y_min,y_max,n_hits", comments="")
    np.savetxt(xz_csv, np.array(xz_rows), delimiter=",", header="z,x_min,x_max,n_hits", comments="")

    png_path = out_dir / "profiles.png"
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting; install it or use the CSV outputs") from e

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    x_plot = np.r_[last_xs, last_xs[:1]]
    y_plot = np.r_[last_ys, last_ys[:1]]
    axs[0].plot(x_plot, y_plot, "-", lw=1.5)
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_title("Mouth XY boundary")
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    axs[0].grid(True, alpha=0.25)

    yz = np.array(yz_rows, dtype=float)
    axs[1].plot(yz[:, 0], yz[:, 1], "-", lw=1.5, label="y_min (x=0 cut)")
    axs[1].plot(yz[:, 0], yz[:, 2], "-", lw=1.5, label="y_max (x=0 cut)")
    axs[1].set_title("YZ profile (x=0 cut)")
    axs[1].set_xlabel("z [m]")
    axs[1].set_ylabel("y [m]")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(fontsize=8)

    xz = np.array(xz_rows, dtype=float)
    axs[2].plot(xz[:, 0], xz[:, 1], "-", lw=1.5, label="x_min (y=0 cut)")
    axs[2].plot(xz[:, 0], xz[:, 2], "-", lw=1.5, label="x_max (y=0 cut)")
    axs[2].set_title("XZ profile (y=0 cut)")
    axs[2].set_xlabel("z [m]")
    axs[2].set_ylabel("x [m]")
    axs[2].grid(True, alpha=0.25)
    axs[2].legend(fontsize=8)

    fig.suptitle(
        f"profile_type={cfg.profile_type}, n_circ={int(cfg.n_circumferential)}, theta={theta_mode}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    return mouth_xy_csv, yz_csv, png_path


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export mouth XY and x=0/y=0 cut profiles for WaveguideMeshConfig.")
    p.add_argument("--config", type=str, required=True, help="Path to WaveguideMeshConfig JSON (from to_json()).")
    p.add_argument("--out", type=str, default="profile_debug_out", help="Output directory.")
    p.add_argument("--n-axial", type=int, default=200, help="Number of axial samples.")
    p.add_argument("--theta", type=str, default="mesh", choices=["mesh", "dense"], help="Angle sampling mode.")
    args = p.parse_args(list(argv) if argv is not None else None)

    cfg = WaveguideMeshConfig.from_json(args.config)
    cfg.validate()
    out_dir = Path(args.out)
    mouth_xy_csv, yz_csv, png_path = export_profiles(cfg, out_dir=out_dir, n_axial=int(args.n_axial), theta_mode=str(args.theta))
    print(f"Wrote {mouth_xy_csv}")
    print(f"Wrote {yz_csv}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
