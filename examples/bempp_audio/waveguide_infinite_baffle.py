"""
Waveguide on infinite baffle example.

Demonstrates:
- CTS (Conical-to-Tangent Smoothstep) waveguide profile
- Infinite baffle half-space radiation model
- OSRC preconditioning for high-frequency efficiency
- 3D HTML mesh visualization

CTS Profile (3-section blending):
- Throat blend: driver exit angle -> conical angle (smooth acoustic coupling)
- Conical section: constant wall angle (maintains beamwidth above Fc)
- Mouth blend: conical -> horizontal (tangent to baffle, reduces diffraction)

OSRC Preconditioning:
- Uses On-Surface Radiation Condition approximation of the NtD map
- Significantly reduces GMRES iterations at high frequencies (ka > 5)
- For this geometry (300mm mouth), crossover is ~1.8 kHz
- Expect 3-10x fewer iterations above crossover frequency

Usage:
    python waveguide_infinite_baffle.py

Environment Variables:
    BEMPPAUDIO_MESH_PRESET   Mesh resolution: ultra-fast, super-fast, fast, standard, slow (default: ultra-fast)
    BEMPPAUDIO_SOLVER_PRESET Solver options preset: ultra-fast, super-fast, fast, standard, slow (default: matches mesh preset)
    BEMPPAUDIO_LOCK_THROAT   Lock throat rim discretization: 1 or 0 (default: 1)
    BEMPPAUDIO_THROAT_CIRCLE_POINTS  Axisymmetric throat rim point count (default: derived from n_circumferential)
    BEMPPAUDIO_USE_OSRC      Enable OSRC preconditioning: 1 or 0 (default: 1)
    BEMPPAUDIO_OSRC_NPADE    OSRC Padé order: 1-4 (default: 2)
    BEMPPAUDIO_N_WORKERS     Parallel workers: 1, 2, ... or "auto" (default: auto)
    BEMPPAUDIO_F1            Start frequency (Hz, default: 200)
    BEMPPAUDIO_F2            End frequency (Hz, default: 20000)
    BEMPPAUDIO_NUM_FREQS     Number of frequencies (default: 32)
    BEMPPAUDIO_OUT_DIR       Output directory for results (default: logs)
    BEMPP_DEVICE_INTERFACE   Backend: numba or opencl (default: numba)

Benchmarking OSRC benefit:
    # With OSRC (default)
    python waveguide_infinite_baffle.py

    # Without OSRC (for comparison)
    BEMPPAUDIO_USE_OSRC=0 python waveguide_infinite_baffle.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


def main() -> int:
    _ensure_repo_on_path()

    # Use numba backend (more reliable than OpenCL on many systems)
    os.environ.setdefault("BEMPP_DEVICE_INTERFACE", "numba")

    from bempp_audio import Loudspeaker
    from bempp_audio.progress import get_logger
    from bempp_audio.mesh.reporting import log_profile_check
    from bempp_audio.mesh.waveguide import WaveguideMeshConfig
    from bempp_audio.mesh.profiles import cts_mouth_angle_deg
    from bempp_audio.mesh.morph import MorphConfig
    from bempp_audio.mesh.validation import MeshResolutionPresets

    logger = get_logger()
    out_dir = Path(os.environ.get("BEMPPAUDIO_OUT_DIR", "logs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(__file__).stem
    dashboard_path = out_dir / f"{stem}_dashboard.html"

    # Waveguide geometry (high expansion ratio for good tangency)
    throat_d = 0.0254  # 25.4mm throat
    mouth_d = 0.300    # 300mm mouth
    length = 0.070     # 70mm length

    # Morph to a rounded rectangle mouth (6mm corner radius)
    morph_width = 0.300    # 300mm wide mouth
    morph_height = 0.100   # 100mm tall mouth
    morph_corner_r = 0.006 # 6mm corner radius

    # CTS profile parameters
    throat_blend = 0.15     # Blend first 15% from driver exit angle
    transition = 0.75       # Start mouth blend at 75%
    tangency = 1.0          # Maximum mouth tangency
    driver_exit_angle = 10.0  # Typical compression driver exit angle

    # Compute resulting mouth angle
    mouth_angle = cts_mouth_angle_deg(
        throat_d / 2, mouth_d / 2, length,
        throat_blend=throat_blend,
        transition=transition,
        driver_exit_angle_deg=driver_exit_angle,
        tangency=tangency,
    )

    logger.info("CTS Waveguide Configuration:")
    logger.config(
        "Geometry",
        f"throat={throat_d*1000:.0f}mm, mouth={mouth_d*1000:.0f}mm, length={length*1000:.0f}mm",
    )
    logger.config(
        "Throat blend",
        f"0 → {throat_blend*100:.0f}% (driver angle: {driver_exit_angle:.0f}deg)",
    )
    logger.config("Conical section", f"{throat_blend*100:.0f}% → {transition*100:.0f}%")
    logger.config("Mouth blend", f"{transition*100:.0f}% → 100% (tangency: {tangency:.1f})")
    logger.config("Resulting mouth angle", f"{mouth_angle:.1f}deg (90deg = tangent to baffle)")
    logger.config(
        "Morph",
        "circle → rounded rectangle "
        f"({morph_width*1000:.0f}×{morph_height*1000:.0f}mm, r={morph_corner_r*1000:.0f}mm)",
    )

    # Mesh resolution presets (speed vs accuracy).
    mesh_preset = os.environ.get("BEMPPAUDIO_MESH_PRESET", "ultra-fast").strip()
    sizes = MeshResolutionPresets.get_preset(mesh_preset)
    h_throat = float(sizes["h_throat"])
    h_mouth = float(sizes["h_mouth"])
    n_axial = int(sizes.get("n_axial_slices", 20))
    n_circ = int(sizes.get("n_circumferential", 72))
    corner_resolution = int(sizes.get("corner_resolution", 8))
    logger.config(
        "Mesh preset",
        f"{mesh_preset} (h_throat={h_throat*1e3:.1f}mm, h_mouth={h_mouth*1e3:.1f}mm, "
        f"n_axial={n_axial}, n_circ={n_circ}, corner_res={corner_resolution})",
    )

    # Solver performance presets (speed vs accuracy). This only affects iterative
    # solver settings; geometry/mesh are controlled by BEMPPAUDIO_MESH_PRESET.
    #
    # Keep this forgiving: examples should not crash just because a preset name
    # spelling differs (e.g. ultra_fast vs ultra-fast).
    def _norm_preset(value: str) -> str:
        return str(value).strip().lower().replace("_", "-").replace(" ", "-")

    solver_preset = os.environ.get("BEMPPAUDIO_SOLVER_PRESET", mesh_preset)
    solver_preset_key = _norm_preset(solver_preset)
    solver_map = {
        "ultra-fast": {"tol": 3e-4, "maxiter": 400},
        "super-fast": {"tol": 1e-4, "maxiter": 600},
        "fast": {"tol": 3e-5, "maxiter": 800},
        "standard": {"tol": 1e-5, "maxiter": 1000},
        "slow": {"tol": 3e-6, "maxiter": 1500},
    }
    if solver_preset_key not in solver_map:
        # Fall back to a sensible default rather than aborting an expensive run.
        logger.warning(
            f"Unknown BEMPPAUDIO_SOLVER_PRESET '{solver_preset}'; falling back to 'ultra-fast'. "
            f"Available: {list(solver_map.keys())}"
        )
        solver_preset_key = "ultra-fast"
    solver_cfg = solver_map[solver_preset_key]
    logger.config("Solver preset", f"{solver_preset_key} (tol={solver_cfg['tol']:.0e}, maxiter={solver_cfg['maxiter']})")

    # OSRC preconditioning parameters
    # OSRC provides significant speedup when ka > 5 (k = 2*pi*f/c, a = mouth_radius)
    # For this geometry: f_crossover = 5 * c / (2*pi*a) where a = mouth_d/2
    # Set BEMPPAUDIO_USE_OSRC=0 to disable for comparison benchmarks
    c_sound = 343.0  # m/s
    ka_crossover = 5.0
    f_crossover = ka_crossover * c_sound / (3.14159 * mouth_d)
    use_osrc = os.environ.get("BEMPPAUDIO_USE_OSRC", "1").strip() != "0"
    osrc_npade = int(os.environ.get("BEMPPAUDIO_OSRC_NPADE", "2").strip())
    logger.config(
        "OSRC preconditioner",
        f"{'enabled' if use_osrc else 'disabled'} (npade={osrc_npade}, "
        f"ka=5 crossover at {f_crossover:.0f} Hz)",
    )

    # Create mesh config
    lock_throat = os.environ.get("BEMPPAUDIO_LOCK_THROAT", "1").strip() != "0"
    throat_circle_points_env = os.environ.get("BEMPPAUDIO_THROAT_CIRCLE_POINTS")
    throat_circle_points = int(throat_circle_points_env) if throat_circle_points_env is not None else None

    waveguide_cfg = WaveguideMeshConfig(
        throat_diameter=throat_d,
        mouth_diameter=mouth_d,
        length=length,
        profile_type="cts",
        n_axial_slices=n_axial,
        n_circumferential=n_circ,
        corner_resolution=corner_resolution,
        h_throat=h_throat,
        h_mouth=h_mouth,
        throat_cap_refinement=1.0,
        lock_throat_boundary=bool(lock_throat),
        throat_circle_points=throat_circle_points,
        cts_throat_blend=throat_blend,
        cts_transition=transition,
        cts_driver_exit_angle_deg=driver_exit_angle,
        cts_tangency=tangency,
        morph=MorphConfig.rectangle(
            width=morph_width,
            height=morph_height,
            corner_radius=morph_corner_r,
            fixed_part=0.1,
            rate=0.6,
            allow_shrinkage=False,
            profile_mode="axes",
        ),
    )

    # Numeric sanity check: verify expansion (avoid “vase/waist” behavior).
    # For strong mouth aspect ratios, some directions can locally contract when
    # the morph blend changes faster than the geometric expansion.
    from bempp_audio.mesh import check_profile

    report = check_profile(waveguide_cfg, n_axial=200, n_directions=36, tol=0.0)
    log_profile_check("Profile check", report, logger=logger)
    if not report.ok:
        logger.warning(
            "Profile still shows contraction; consider adjusting target aspect ratio or morph timing."
        )

    # Build and solve
    # Note: .use_osrc() enables OSRC preconditioning which reduces GMRES iterations
    # at high frequencies. The benefit is most significant above the ka=5 crossover.
    speaker = Loudspeaker().waveguide_from_config(
        waveguide_cfg,
        throat_velocity_amplitude=0.01,
        export_mesh=False,
        show_mesh_quality=False,
    ).infinite_baffle().solver_options(tol=float(solver_cfg["tol"]), maxiter=int(solver_cfg["maxiter"]))

    if use_osrc:
        speaker = speaker.use_osrc(npade=osrc_npade)

    speaker = (
        speaker
        .solver_progress(gmres_log_every=2)
        .frequency_range(
            f1=float(os.environ.get("BEMPPAUDIO_F1", "200").strip()),
            f2=float(os.environ.get("BEMPPAUDIO_F2", "20000").strip()),
            num=int(os.environ.get("BEMPPAUDIO_NUM_FREQS", "32").strip()),
            spacing="log",
        )
        .preset_horn()
    )

    # Print mesh summary
    speaker.describe_mesh()

    # Export 3D mesh visualization
    try:
        from bempp_audio.viz import mesh_3d_html

        mesh = speaker.mesh
        if mesh is not None:
            wg = speaker.state.waveguide
            domain_names = {
                wg.throat_domain: "Throat (vibrating)",
                wg.wall_domain: "Walls (rigid)",
            }
            domain_colors = {
                wg.throat_domain: "#FF6B6B",
                wg.wall_domain: "#4ECDC4",
            }
            mesh_3d_html(
                mesh,
                filename=str(out_dir / "waveguide_cts_mesh.html"),
                title="CTS Waveguide Mesh",
                color_by_domain=True,
                domain_names=domain_names,
                domain_colors=domain_colors,
                show_edges=True,
            )
            logger.info(f"3D mesh exported: {out_dir / 'waveguide_cts_mesh.html'}")
    except Exception as e:
        logger.warning(f"3D export skipped: {e}")

    # Solve
    n_workers_env = os.environ.get("BEMPPAUDIO_N_WORKERS", "auto").strip().lower()
    n_workers = None if n_workers_env == "auto" else int(n_workers_env)
    response = speaker.solve(n_workers=n_workers)

    # -------------------------------------------------------------------------
    # Minimal data-path sanity checks (BEMPP_CL -> FrequencyResponse -> viz)
    # -------------------------------------------------------------------------
    if len(response.results) == 0:
        logger.error("Solve returned an empty FrequencyResponse; skipping visualization exports.")
        return 2

    try:
        freqs = response.frequencies
        logger.info(f"Response points: {len(freqs)} ({float(freqs[0]):.0f}–{float(freqs[-1]):.0f} Hz)")
    except Exception:
        logger.info(f"Response points: {len(response.results)}")

    # Evaluate a single on-axis point and far-field direction at the first frequency.
    # This guards against upstream breaks where the solve succeeds but field/far-field
    # evaluations (used by viz) fail due to missing operators/BC data.
    try:
        import numpy as np

        r0 = response.results[0]
        p = r0.pressure_at(np.array([[0.0], [0.0], [1.0]], dtype=float))
        if not (np.isfinite(p.real).all() and np.isfinite(p.imag).all()):
            logger.warning("Field evaluation returned non-finite values; plots may be unreliable.")
        ff = r0.directivity().far_field_at(np.array([[0.0], [0.0], [1.0]], dtype=float))
        if not (np.isfinite(ff.real).all() and np.isfinite(ff.imag).all()):
            logger.warning("Far-field evaluation returned non-finite values; DI/beamwidth panels may be unreliable.")
    except Exception as e:
        logger.warning(f"Viz sanity check failed (field/far-field eval): {e}")

    # Save summary plot only (single figure)
    try:
        from bempp_audio.viz import ReportBuilder

        ReportBuilder(response).title("CTS Waveguide (Infinite Baffle) — Summary").polar_options(
            max_angle_deg=90, normalize_angle_deg=0.0
        ).preset_waveguide_summary().save(str(out_dir / f"{stem}_summary.png"))

        ReportBuilder(response).title("CTS Waveguide (Infinite Baffle) — Designer").polar_options(
            max_angle_deg=90, normalize_angle_deg=10.0
        ).preset_waveguide_designer().save(str(out_dir / f"{stem}_designer.png"))
    except Exception as e:
        logger.warning(f"Plot export skipped: {e}")

    try:
        from bempp_audio.viz import save_driver_dashboard_html

        save_driver_dashboard_html(
            response,
            filename=str(dashboard_path),
            title="Waveguide Infinite Baffle — Dashboard",
            distance=1.0,
            max_angle=90.0,
            normalize_angle=10.0,
            show_progress=True,
        )
        logger.info(f"Dashboard exported: {dashboard_path}")
    except Exception as e:
        logger.warning(f"Dashboard export skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
