#!/usr/bin/env python3
"""
Ultra-minimal piston example for `bempp_audio`.

This is tuned to finish quickly in constrained environments:
- coarse mesh (`element_size`)
- single frequency solve
- `n_workers=1` to avoid multiprocessing restrictions

Plot generation is optional (requires matplotlib/plotly).
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

    # Avoid OpenCL kernel build issues on some systems (including WSL2/PoCL).
    os.environ.setdefault("BEMPP_DEVICE_INTERFACE", "numba")

    from bempp_audio import Loudspeaker
    from bempp_audio.viz import ReportBuilder

    out_dir = Path(os.environ.get("BEMPPAUDIO_OUT_DIR", "logs")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(__file__).stem

    radius_m = 0.05  # 50 mm
    element_size_m = 0.01  # coarse for a fast demo
    response = (
        Loudspeaker()
        .circular_piston(radius=radius_m, element_size=element_size_m)
        .infinite_baffle()
        .single_frequency(1000.0)
        .solve(n_workers=1)
    )

    try:
        ReportBuilder(response).title("Piston (Ultra Minimal) — Summary").polar_options(
            max_angle_deg=90, normalize_angle_deg=0.0
        ).preset_waveguide_summary().save(str(out_dir / f"{stem}_summary.png"))

        ReportBuilder(response).title("Piston (Ultra Minimal) — Designer").polar_options(
            max_angle_deg=90, normalize_angle_deg=10.0
        ).preset_waveguide_designer().save(str(out_dir / f"{stem}_designer.png"))
    except Exception as e:
        print(f"Plot export skipped (optional deps missing?): {e}")

    try:
        from bempp_audio.viz import save_driver_dashboard_html

        save_driver_dashboard_html(
            response,
            filename=str(out_dir / f"{stem}_dashboard.html"),
            title="Piston (Ultra Minimal) — Dashboard",
            distance=1.0,
            max_angle=90.0,
            normalize_angle=10.0,
            show_progress=False,
        )
    except Exception as e:
        print(f"Dashboard export skipped (optional deps missing?): {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
