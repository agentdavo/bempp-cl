#!/usr/bin/env python3
"""
CTS rounded-rectangle waveguide (320×240mm, r=6mm) on box with a directivity objective.

Goal (objective-driven):
- Horizontal (-6 dB) beamwidth target: 90° over 1–16 kHz
- Optional vertical target via env var

Artifacts (default `logs/`):
- `<stem>_summary.png`
- `<stem>_dashboard.html`
- `<stem>_objective.json`

Quick probe example:

  BEMPPAUDIO_PROBE_5F=1 BEMPPAUDIO_N_WORKERS=8 \\
    python examples/bempp_audio/waveguide_cts_rect_320x240_objective.py

Environment variables
---------------------
Core:
  BEMPPAUDIO_OUT_DIR         Output directory (default: logs)
  BEMPPAUDIO_N_WORKERS       Parallel workers: 1, 2, ... or "auto" (default: auto)
  BEMPPAUDIO_MESH_PRESET     Mesh preset (default: super_fast)
  BEMPPAUDIO_SOLVER_PRESET   Solver preset (default: matches mesh preset)
  BEMPPAUDIO_USE_OSRC        "1"/"0" (default: 0)
  BEMPPAUDIO_OSRC_NPADE      int (default: 2)
  BEMPPAUDIO_EXPORT_MESH     "1"/"0" export mesh HTML (default: 1)
  BEMPPAUDIO_EXPORT_FRD      "1"/"0" export FRD files (default: 1)
  BEMPPAUDIO_FRD_ANGLES      "spl" | "polar" | "0,10,20,..." (default: spl)
  BEMPPAUDIO_N_CIRC          Override circumferential samples (default: preset)
  BEMPPAUDIO_CORNER_RES      Override corner arc samples (default: preset)
  BEMPPAUDIO_MORPH_FIXED_MM  Keep circle for first N mm of expansion (default: 5)
  BEMPPAUDIO_MORPH_RATE      Morph blend exponent (default: 3)

Waveguide geometry:
  BEMPPAUDIO_WG_LEN_M        Waveguide depth/length (meters, default: 0.18)

Box geometry (default: sized for 320×240mm mouth with 100mm margin):
  BEMPPAUDIO_BOX_WIDTH_M     Box width (meters, default: 0.52)
  BEMPPAUDIO_BOX_HEIGHT_M    Box height (meters, default: 0.44)
  BEMPPAUDIO_BOX_DEPTH_M     Box depth (meters, default: 0.30)

Directivity objective:
  BEMPPAUDIO_OBJ_F_LO        Lower band edge (Hz, default: 1000)
  BEMPPAUDIO_OBJ_F_HI        Upper band edge (Hz, default: 16000)
  BEMPPAUDIO_TARGET_BW_H_DEG Horizontal -6dB beamwidth target (deg, default: 90)
  BEMPPAUDIO_TARGET_BW_V_DEG Vertical -6dB beamwidth target (deg, default: 90)
  BEMPPAUDIO_OBJ_DI_MODE     "proxy" (fast) or "balloon" (slow, accurate)

Frequency sampling:
  BEMPPAUDIO_NUM_FREQS       Number of freq points (default: 28)
  BEMPPAUDIO_PROBE_5F        If "1": override to 5 log-spaced freqs in [OBJ_F_LO, OBJ_F_HI]
  BEMPPAUDIO_PROBE_MESH_PRESET Mesh preset used for probe mode when presets are not explicitly set (default: standard)

Cabinet chamfers (all edges, symmetric 45-degree, millimeters):
  BEMPPAUDIO_CHAMFER_ALL_MM  Apply same chamfer to all four front-face edges (default: 0, disabled)
  BEMPPAUDIO_CHAMFER_TOP_MM  Chamfer on top edge only (overrides CHAMFER_ALL)
  BEMPPAUDIO_CHAMFER_BOTTOM_MM  Chamfer on bottom edge only
  BEMPPAUDIO_CHAMFER_LEFT_MM    Chamfer on left edge only
  BEMPPAUDIO_CHAMFER_RIGHT_MM   Chamfer on right edge only
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
import sys


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


def main() -> int:
    _ensure_repo_on_path()
    os.environ.setdefault("BEMPP_DEVICE_INTERFACE", "numba")

    from bempp_audio import Loudspeaker
    from bempp_audio.mesh.validation import MeshResolutionValidator, MeshResolutionPresets
    from bempp_audio.mesh.cabinet import ChamferSpec
    from bempp_audio.results import BeamwidthTarget, DirectivityObjectiveConfig, evaluate_directivity_objective
    from bempp_audio.viz import ReportBuilder, save_driver_dashboard_html
    from bempp_audio.progress import get_logger

    logger = get_logger()

    out_dir = Path(os.environ.get("BEMPPAUDIO_OUT_DIR", "logs")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(__file__).stem

    mesh_preset = os.environ.get("BEMPPAUDIO_MESH_PRESET", "super_fast").strip()
    solver_preset = os.environ.get("BEMPPAUDIO_SOLVER_PRESET", mesh_preset).strip()
    n_workers_env = os.environ.get("BEMPPAUDIO_N_WORKERS", "auto").strip().lower()
    n_workers = None if n_workers_env == "auto" else int(n_workers_env)
    export_mesh = os.environ.get("BEMPPAUDIO_EXPORT_MESH", "1").strip() != "0"

    use_osrc = os.environ.get("BEMPPAUDIO_USE_OSRC", "0").strip() != "0"
    osrc_npade = int(os.environ.get("BEMPPAUDIO_OSRC_NPADE", "2").strip())

    n_circ_env = os.environ.get("BEMPPAUDIO_N_CIRC")
    n_circ = int(n_circ_env) if n_circ_env is not None else 36
    corner_res_env = os.environ.get("BEMPPAUDIO_CORNER_RES")
    corner_res = int(corner_res_env) if corner_res_env is not None else 0
    morph_fixed_mm = float(os.environ.get("BEMPPAUDIO_MORPH_FIXED_MM", "5").strip())
    morph_rate = float(os.environ.get("BEMPPAUDIO_MORPH_RATE", "3").strip())

    # Fixed target geometry for this example
    throat_d = 0.0254
    mouth_w = 0.320
    mouth_h = 0.240
    corner_r = 0.006
    length_m = float(os.environ.get("BEMPPAUDIO_WG_LEN_M", "0.18").strip())

    # Box dimensions (default: 100mm margin around mouth, 300mm deep)
    box_width = float(os.environ.get("BEMPPAUDIO_BOX_WIDTH_M", "0.52").strip())
    box_height = float(os.environ.get("BEMPPAUDIO_BOX_HEIGHT_M", "0.44").strip())
    box_depth = float(os.environ.get("BEMPPAUDIO_BOX_DEPTH_M", "0.30").strip())

    f_lo = float(os.environ.get("BEMPPAUDIO_OBJ_F_LO", "1000").strip())
    f_hi = float(os.environ.get("BEMPPAUDIO_OBJ_F_HI", "16000").strip())
    bw_h = float(os.environ.get("BEMPPAUDIO_TARGET_BW_H_DEG", "90").strip())
    bw_v = float(os.environ.get("BEMPPAUDIO_TARGET_BW_V_DEG", "90").strip())

    num_freqs = int(os.environ.get("BEMPPAUDIO_NUM_FREQS", "28").strip())
    probe_5f = os.environ.get("BEMPPAUDIO_PROBE_5F", "0").strip() == "1"

    # Probe mode: default to a higher-resolution mesh/solver preset for the 5-point
    # "am I in the right ballpark?" sweep unless the user explicitly set presets.
    if probe_5f:
        probe_preset = os.environ.get("BEMPPAUDIO_PROBE_MESH_PRESET", "standard").strip()
        if "BEMPPAUDIO_MESH_PRESET" not in os.environ:
            mesh_preset = probe_preset
        if "BEMPPAUDIO_SOLVER_PRESET" not in os.environ:
            solver_preset = mesh_preset

    # Mesh preset sanity: print the implied throat resolution vs band top.
    try:
        preset = MeshResolutionPresets.get_preset(mesh_preset)
        MeshResolutionValidator.validate_element_size(preset["h_throat"], freq_max_hz=f_hi, verbose=True)
    except Exception:
        pass

    from bempp_audio.mesh import MorphConfig

    # Morph config for rounded-rectangle mouth
    morph = MorphConfig.rectangle(
        width=float(mouth_w),
        height=float(mouth_h),
        corner_radius=float(corner_r),
        profile_mode="axes",
    )

    # Cabinet chamfers (optional, symmetric 45-degree chamfers on front-face edges)
    def _parse_chamfer_mm(env_name: str, fallback_mm: float = 0.0) -> ChamferSpec | None:
        val_str = os.environ.get(env_name, "").strip()
        if not val_str:
            if fallback_mm > 0:
                return ChamferSpec.symmetric(fallback_mm * 1e-3)
            return None
        val = float(val_str)
        if val <= 0:
            return None
        return ChamferSpec.symmetric(val * 1e-3)

    chamfer_all_mm = float(os.environ.get("BEMPPAUDIO_CHAMFER_ALL_MM", "0").strip())
    chamfer_top = _parse_chamfer_mm("BEMPPAUDIO_CHAMFER_TOP_MM", chamfer_all_mm)
    chamfer_bottom = _parse_chamfer_mm("BEMPPAUDIO_CHAMFER_BOTTOM_MM", chamfer_all_mm)
    chamfer_left = _parse_chamfer_mm("BEMPPAUDIO_CHAMFER_LEFT_MM", chamfer_all_mm)
    chamfer_right = _parse_chamfer_mm("BEMPPAUDIO_CHAMFER_RIGHT_MM", chamfer_all_mm)

    has_chamfers = any(c is not None for c in [chamfer_top, chamfer_bottom, chamfer_left, chamfer_right])
    if has_chamfers:
        chamfer_info = []
        for name, spec in [("top", chamfer_top), ("bottom", chamfer_bottom), ("left", chamfer_left), ("right", chamfer_right)]:
            if spec is not None:
                d1, d2 = spec.distances()
                chamfer_info.append(f"{name}={d1*1e3:.1f}mm")
        logger.info(f"Cabinet chamfers: {', '.join(chamfer_info)}")

    speaker = (
        Loudspeaker()
        .performance_preset(solver_preset, mode="horn")
        .waveguide_on_box(
            throat_diameter=throat_d,
            mouth_diameter=float(max(mouth_w, mouth_h)),  # axisymmetric envelope; morph sets final shape
            waveguide_length=length_m,
            box_width=box_width,
            box_height=box_height,
            box_depth=box_depth,
            profile="cts",
            mesh_preset=None,  # use state.mesh_preset from performance_preset
            throat_velocity_amplitude=0.01,
            n_circumferential=int(n_circ),
            morph=morph,
            morph_fixed_part=float(max(0.0, min(0.99, (morph_fixed_mm * 1e-3) / max(length_m, 1e-9)))),
            morph_rate=float(morph_rate),
            lock_throat_boundary=True,
            export_mesh=bool(export_mesh),
            export_mesh_path=str(out_dir / f"{stem}_mesh.html"),
            cts_driver_exit_angle_deg=10.0,
            cts_throat_blend=0.15,
            cts_transition=0.80,
            cts_tangency=1.0,
            cts_mouth_roll=0.6,
            cts_mid_curvature=0.2,
            cts_curvature_regularizer=1.0,
            show_mesh_quality=True,
            # Cabinet chamfers (front-face perimeter edges)
            cabinet_chamfer_top=chamfer_top,
            cabinet_chamfer_bottom=chamfer_bottom,
            cabinet_chamfer_left=chamfer_left,
            cabinet_chamfer_right=chamfer_right,
        )
        .free_space()
        .frequency_range(f_lo, f_hi, num=(5 if probe_5f else num_freqs), spacing="log")
        .preset_horn(distance=1.0, max_angle=90.0, resolution_deg=5.0, normalize_angle=10.0)
    )

    if use_osrc:
        speaker = speaker.use_osrc(npade=osrc_npade)

    logger.info(
        f"Waveguide on box: throat=25.4mm mouth=320x240mm r=6mm L={length_m * 1e3:.0f}mm | "
        f"box={box_width * 1e3:.0f}x{box_height * 1e3:.0f}x{box_depth * 1e3:.0f}mm | "
        f"objective: BW_H={bw_h:.0f}° BW_V={bw_v:.0f}° @ -6dB over {f_lo:.0f}–{f_hi:.0f} Hz"
    )
    logger.info(f"Solve: mesh_preset={mesh_preset} solver_preset={solver_preset} n_workers={n_workers_env}")

    response = speaker.solve(n_workers=n_workers)
    if not response.results:
        logger.error("Empty response (no results); aborting.")
        return 2

    export_frd = os.environ.get("BEMPPAUDIO_EXPORT_FRD", "1").strip() != "0"
    if export_frd:
        try:
            import numpy as np

            frd_dir = out_dir / f"{stem}_frd"
            frd_dir.mkdir(parents=True, exist_ok=True)

            frd_angles_spec = os.environ.get("BEMPPAUDIO_FRD_ANGLES", "spl").strip().lower()
            if frd_angles_spec in ("spl", "spl_angles"):
                angles = [float(x) for x in speaker.plot_settings["angles"]]
            elif frd_angles_spec in ("polar", "polar_angles"):
                angles = np.linspace(float(speaker.state.polar_start), float(speaker.state.polar_end), int(speaker.state.polar_num))
                angles = [float(a) for a in angles]
            else:
                angles = [float(x.strip()) for x in frd_angles_spec.split(",") if x.strip()]
                if not angles:
                    angles = [0.0]

            distance_m = float(speaker.plot_settings["distance"])

            def _angle_token(angle_deg: float) -> str:
                a = float(angle_deg)
                if abs(a - round(a)) < 1e-9:
                    return f"{int(round(a)):+d}"
                s = f"{a:+.2f}".rstrip("0").rstrip(".")
                return s.replace(".", "p")

            for plane, tag in (("horizontal", "hor"), ("vertical", "ver")):
                sweep = response.polar_sweep(angles_deg=angles, distance_m=distance_m, plane=plane, show_progress=False)
                freqs = sweep.frequencies_hz
                spl_db = sweep.spl_db(ref_pa=20e-6)
                phase_deg = np.degrees(np.unwrap(np.angle(sweep.pressure), axis=0))

                for j, angle in enumerate(sweep.angles_deg):
                    filename = frd_dir / f"{stem}-{tag}_deg{_angle_token(float(angle))}.txt"
                    with open(filename, "w") as f:
                        f.write("Frequency\tSPL_dB\tPhase_deg\n")
                        for freq, mag, ph in zip(freqs, spl_db[:, j], phase_deg[:, j]):
                            f.write(f"{float(freq):.4f}\t{float(mag):.4f}\t{float(ph):.4f}\n")

            logger.success(f"Saved FRD: {frd_dir}")
        except Exception as e:
            logger.warning(f"FRD export skipped: {e}")

    cfg = DirectivityObjectiveConfig(
        beamwidth_level_db=-6.0,
        f_lo_hz=f_lo,
        f_hi_hz=f_hi,
        target_h=BeamwidthTarget(kind="constant", value_deg=bw_h),
        target_v=BeamwidthTarget(kind="constant", value_deg=bw_v),
        di_mode=os.environ.get("BEMPPAUDIO_OBJ_DI_MODE", "proxy").strip().lower(),
    )
    obj = evaluate_directivity_objective(response.results, cfg=cfg, plane_h="xz", plane_v="yz")
    (out_dir / f"{stem}_objective.json").write_text(json.dumps(asdict(obj), indent=2) + "\n")
    logger.info(
        f"Directivity objective: J={obj.J:.3f} "
        f"(BW_RMSE_H={obj.bw_rmse_h_deg:.1f}° BW_RMSE_V={obj.bw_rmse_v_deg:.1f}° "
        f"DI_ripple_H={obj.di_ripple_h_db:.1f}dB DI_ripple_V={obj.di_ripple_v_db:.1f}dB)"
    )

    try:
        (
            ReportBuilder(response)
            .title("CTS Rect Waveguide 320×240 (r=6mm) on Box — Summary")
            .polar_options(max_angle_deg=90.0, normalize_angle_deg=10.0)
            .preset_waveguide_designer()
            .save(str(out_dir / f"{stem}_summary.png"))
        )
    except Exception as e:
        logger.warning(f"Summary export skipped (optional deps missing?): {e}")

    try:
        save_driver_dashboard_html(
            response,
            filename=str(out_dir / f"{stem}_dashboard.html"),
            title="CTS Rect Waveguide 320×240 (r=6mm) on Box — Directivity Objective",
            distance=1.0,
            max_angle=90.0,
            normalize_angle=10.0,
            show_progress=True,
        )
    except Exception as e:
        logger.warning(f"Dashboard export skipped (optional deps missing?): {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
