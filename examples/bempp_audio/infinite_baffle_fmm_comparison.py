"""
Compare dense vs FMM-backed BEM solves on a simple infinite-baffle case.

This script runs the same piston-on-infinite-baffle simulation twice:
- Dense operator application (default)
- FMM-enabled operator application (if available)

It writes two summary charts for side-by-side comparison.

Usage:
  python examples/bempp_audio/infinite_baffle_fmm_comparison.py

Environment:
  BEMPP_DEVICE_INTERFACE=numba|opencl
  BEMPPAUDIO_FMM_ORDER=5
  BEMPPAUDIO_N_WORKERS=1|auto
  BEMPPAUDIO_NUM_FREQS=20
  BEMPPAUDIO_F1=200
  BEMPPAUDIO_F2=20000
  BEMPPAUDIO_OUT_DIR=logs
"""

from __future__ import annotations

import os
import time
from pathlib import Path
import sys


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


def _run_case(*, use_fmm: bool, fmm_order: int, n_workers: int | None, f1: float, f2: float, num_freqs: int) -> tuple[object, float]:
    from bempp_audio import Loudspeaker

    speaker = (
        Loudspeaker()
        .circular_piston(radius=0.05, element_size=0.012)
        .infinite_baffle()
        .velocity(mode="piston", amplitude=0.01)
        .frequency_range(float(f1), float(f2), num=int(num_freqs), spacing="log")
        .preset_infinite_baffle(resolution_deg=10.0, normalize_angle=0.0)
        .solver_options(use_fmm=use_fmm, fmm_expansion_order=fmm_order)
    )

    mesh = speaker.mesh
    if mesh is not None:
        info = mesh.info()
        print(f"Mesh: elements={info.n_elements} vertices={info.n_vertices}")

    t0 = time.perf_counter()
    response = speaker.solve(n_workers=n_workers)
    dt = time.perf_counter() - t0
    return response, float(dt)


def main() -> int:
    _ensure_repo_on_path()

    from bempp_audio.viz import ReportBuilder

    os.environ.setdefault("BEMPP_DEVICE_INTERFACE", "numba")
    fmm_order = int(os.environ.get("BEMPPAUDIO_FMM_ORDER", "5"))
    n_workers_env = os.environ.get("BEMPPAUDIO_N_WORKERS", "1").strip().lower()
    n_workers = None if n_workers_env == "auto" else int(n_workers_env)
    num_freqs = int(os.environ.get("BEMPPAUDIO_NUM_FREQS", "20").strip())
    f1 = float(os.environ.get("BEMPPAUDIO_F1", "200").strip())
    f2 = float(os.environ.get("BEMPPAUDIO_F2", "20000").strip())

    out_dir = Path(os.environ.get("BEMPPAUDIO_OUT_DIR", "logs")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Infinite Baffle: Dense vs FMM comparison ===")
    print(f"Output directory: {out_dir}")
    print(f"FMM expansion order: {fmm_order}")
    print(f"Workers: {n_workers_env}  Freqs: {num_freqs} ({f1:.0f}–{f2:.0f} Hz)")

    dense_resp, dense_s = _run_case(
        use_fmm=False, fmm_order=fmm_order, n_workers=n_workers, f1=f1, f2=f2, num_freqs=num_freqs
    )
    dense_png = out_dir / "infinite_baffle_dense_summary.png"
    ReportBuilder(dense_resp).title(f"Infinite Baffle (dense) — {dense_s:.1f}s").polar_options(
        max_angle_deg=90, normalize_angle_deg=0.0
    ).preset_waveguide_summary().save(str(dense_png))

    ReportBuilder(dense_resp).title(f"Infinite Baffle (dense) — Designer").polar_options(
        max_angle_deg=90, normalize_angle_deg=10.0
    ).preset_waveguide_designer().save(str(out_dir / "infinite_baffle_dense_designer.png"))
    print(f"Saved: {dense_png}")

    try:
        from bempp_audio.viz import save_driver_dashboard_html

        save_driver_dashboard_html(
            dense_resp,
            filename=str(out_dir / "infinite_baffle_dense_dashboard.html"),
            title="Infinite Baffle (dense) — Dashboard",
            distance=1.0,
            max_angle=90.0,
            normalize_angle=10.0,
            show_progress=False,
        )
    except Exception as e:
        print(f"Dense dashboard export skipped: {e}")

    try:
        fmm_resp, fmm_s = _run_case(
            use_fmm=True, fmm_order=fmm_order, n_workers=n_workers, f1=f1, f2=f2, num_freqs=num_freqs
        )
    except Exception as e:
        print(f"FMM run failed or unavailable: {e}")
        return 0

    fmm_png = out_dir / "infinite_baffle_fmm_summary.png"
    ReportBuilder(fmm_resp).title(f"Infinite Baffle (FMM) — {fmm_s:.1f}s").polar_options(
        max_angle_deg=90, normalize_angle_deg=0.0
    ).preset_waveguide_summary().save(str(fmm_png))

    ReportBuilder(fmm_resp).title("Infinite Baffle (FMM) — Designer").polar_options(
        max_angle_deg=90, normalize_angle_deg=10.0
    ).preset_waveguide_designer().save(str(out_dir / "infinite_baffle_fmm_designer.png"))
    print(f"Saved: {fmm_png}")

    try:
        from bempp_audio.viz import save_driver_dashboard_html

        save_driver_dashboard_html(
            fmm_resp,
            filename=str(out_dir / "infinite_baffle_fmm_dashboard.html"),
            title="Infinite Baffle (FMM) — Dashboard",
            distance=1.0,
            max_angle=90.0,
            normalize_angle=10.0,
            show_progress=False,
        )
    except Exception as e:
        print(f"FMM dashboard export skipped: {e}")

    print(f"Timing: dense={dense_s:.1f}s, fmm={fmm_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
