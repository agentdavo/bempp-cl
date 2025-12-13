#!/usr/bin/env python3
"""
Waveguide designer report example.

This example builds a CTS waveguide, runs a frequency sweep, then generates a
waveguide-designer report using `ReportBuilder` (normalized to 10° by default).
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
    os.environ.setdefault("BEMPP_DEVICE_INTERFACE", "numba")

    from bempp_audio import Loudspeaker
    from bempp_audio.viz import ReportBuilder

    out_dir = Path(os.environ.get("BEMPPAUDIO_OUT_DIR", "logs")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(__file__).stem

    n_workers_env = os.environ.get("BEMPPAUDIO_N_WORKERS", "1").strip().lower()
    n_workers = None if n_workers_env == "auto" else int(n_workers_env)
    num_freqs = int(os.environ.get("BEMPPAUDIO_NUM_FREQS", "30").strip())

    speaker = (
        Loudspeaker()
        .waveguide(
            throat_diameter=0.025,
            mouth_diameter=0.200,
            length=0.120,
            profile="cts",
            throat_velocity_amplitude=0.01,
            export_mesh=False,
        )
        .infinite_baffle()
        .frequency_range(200, 20000, num=num_freqs, spacing="log")
        .preset_horn()
    )

    response = speaker.solve(n_workers=n_workers)

    try:
        (
            ReportBuilder(response)
            .title("Waveguide Designer Report (CTS)")
            .polar_options(max_angle_deg=90.0, normalize_angle_deg=10.0)
            .preset_waveguide_designer()
            .save(str(out_dir / f"{stem}_summary.png"))
        )
    except Exception as e:
        print(f"Report export skipped (optional deps missing?): {e}")

    try:
        from bempp_audio.viz import save_driver_dashboard_html

        save_driver_dashboard_html(
            response,
            filename=str(out_dir / f"{stem}_dashboard.html"),
            title="Waveguide Designer Dashboard (CTS)",
            distance=1.0,
            max_angle=90.0,
            normalize_angle=10.0,
            show_progress=True,
        )
    except Exception as e:
        print(f"Dashboard export skipped (optional deps missing?): {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
