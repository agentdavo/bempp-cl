#!/usr/bin/env python3
"""
CI-friendly minimal waveguide solve (single frequency).

This script is intended to validate that the end-to-end waveguide -> BEM solve
pipeline works in a predictable, single-process mode.

Behavior:
- If `gmsh` or `bempp_cl.api` are not importable, exits 0 (skipped).
- Builds a small axisymmetric waveguide (no mouth morphing).
- Runs a single-frequency solve with `n_workers=1` and `BEMPP_DEVICE_INTERFACE=numba`.
- Evaluates one on-axis pressure point to ensure field evaluation works.
- Optionally writes a dashboard HTML if plotly is available and `--dashboard` is set.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def main() -> int:
    _ensure_repo_on_path()

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.environ.get("BEMPPAUDIO_OUT_DIR", "logs"), help="Output directory (default: logs)")
    p.add_argument("--frequency", type=float, default=1000.0, help="Solve frequency in Hz (default: 1000)")
    p.add_argument("--dashboard", action="store_true", help="Also write a Plotly dashboard (requires plotly)")
    p.add_argument("--use-osrc", action="store_true", help="Enable OSRC preconditioning (default off for smoke)")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("BEMPP_DEVICE_INTERFACE", "numba")

    from bempp_audio._optional import optional_import

    gmsh, has_gmsh = optional_import("gmsh")
    bempp, has_bempp = optional_import("bempp_cl.api")
    if not has_gmsh or not has_bempp:
        print("SKIP: requires gmsh and bempp_cl.api")
        return 0

    from bempp_audio import Loudspeaker

    speaker = (
        Loudspeaker()
        .waveguide(
            throat_diameter=0.025,
            mouth_diameter=0.08,
            length=0.04,
            profile="exponential",
            throat_element_size=0.008,
            mouth_element_size=0.020,
            throat_velocity_amplitude=0.01,
            n_axial_slices=8,
            n_circumferential=24,
            export_mesh=False,
            show_mesh_quality=False,
        )
        .infinite_baffle()
        .single_frequency(float(args.frequency))
        .preset_horn()
        .solver_options(tol=1e-3, maxiter=250)
    )

    if args.use_osrc:
        speaker = speaker.use_osrc(npade=2)

    response = speaker.solve(n_workers=1, show_progress=True)
    if len(response.results) != 1:
        raise RuntimeError(f"Expected 1 result, got {len(response.results)}")

    # Field eval sanity: 1m on-axis.
    p0 = response.results[0].pressure_at(np.array([[0.0], [0.0], [1.0]], dtype=float))
    if not (np.isfinite(p0.real).all() and np.isfinite(p0.imag).all()):
        raise RuntimeError("Non-finite pressure_at result")

    print(f"OK: waveguide solve at {args.frequency:.0f} Hz, |p|={float(np.abs(p0[0])):g}")

    if args.dashboard:
        plotly, has_plotly = optional_import("plotly")
        if not has_plotly:
            print("SKIP: --dashboard requested but plotly not available")
            return 0
        from bempp_audio.viz import save_driver_dashboard_html

        out = out_dir / "smoke_waveguide_dashboard.html"
        save_driver_dashboard_html(
            response,
            filename=str(out),
            title="Smoke Waveguide Dashboard",
            distance=1.0,
            max_angle=60.0,
            normalize_angle=0.0,
            show_progress=False,
        )
        print(f"Wrote: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
