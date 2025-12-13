#!/usr/bin/env python3
"""
Benchmark 3.0-Ti (v1) modal validation (spherical-cap shell).

This example:
1) Generates the Benchmark3TiConfig_v1 spherical-cap mesh
2) Applies a finite clamped annulus (36.5–37.25 mm)
3) Runs a modal analysis (first N modes) if DOLFINx/SLEPc are available
4) Exports modes to XDMF for visualization

If DOLFINx cannot be imported (common in some WSL2/MPI setups), the script exits
gracefully after writing the surface mesh and printing next steps.
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

    import numpy as np

    from bempp_audio.fea import Benchmark3TiConfig_v1, DomeMesher

    cfg = Benchmark3TiConfig_v1()
    geom = cfg.dome_geometry()

    out_dir = Path(os.environ.get("BEMPPAUDIO_OUT_DIR", "logs")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(__file__).stem
    surface_xdmf = out_dir / f"{stem}_surface.xdmf"
    modes_xdmf = out_dir / f"{stem}_modes.xdmf"

    # Generate surface mesh (gmsh required)
    mesh = DomeMesher(geom).generate(element_size_m=0.8e-3)

    # Export surface mesh to XDMF (meshio)
    try:
        import meshio

        m = meshio.Mesh(points=mesh.vertices, cells=[("triangle", mesh.triangles)])
        meshio.write(str(surface_xdmf), m)
        print(f"Surface mesh written: {surface_xdmf}")
    except Exception as e:
        print(f"Mesh export failed: {e}")
        return 1

    # Try to import dolfinx (may fail in some MPI setups).
    try:
        import dolfinx
        from dolfinx.io import XDMFFile
        from mpi4py import MPI
    except Exception as e:
        print(f"\nDOLFINx not available (modal solve skipped): {e}")
        print("Next: run this on a working DOLFINx MPI environment to compute modal frequencies.")
        return 0

    # Read mesh into DOLFINx
    with XDMFFile(MPI.COMM_WORLD, str(surface_xdmf), "r") as xf:
        dfx_mesh = xf.read_mesh(name="Grid")

    from bempp_audio.fea import SphericalCapMindlinReissnerShellSolver

    solver = SphericalCapMindlinReissnerShellSolver(
        dfx_mesh,
        sphere_radius_m=cfg.sphere_radius_m,
        material=cfg.material,
        degree=1,
        shear_quadrature_degree=1,
    )
    solver.add_clamped_annulus(r_inner_m=cfg.clamp_inner_radius_m, r_outer_m=cfg.clamp_outer_radius_m)

    modal = solver.solve_modes(n_modes=8)
    print("\nBenchmark3TiConfig_v1 modes (spherical-cap MR shell):")
    for i, f in enumerate(modal.frequencies_hz):
        print(f"  mode {i}: {float(f):.1f} Hz")

    try:
        solver.write_modes_xdmf(str(modes_xdmf), modal)
        print(f"Modes written: {modes_xdmf}")
    except Exception as e:
        print(f"Mode export skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
