#!/usr/bin/env python3
"""
Mindlin–Reissner plate modal analysis demo (Phase 7.2).

This example solves a small modal problem for a flat plate using DOLFINx +
SLEPc via `MindlinReissnerPlateSolver`.

Notes:
- Requires `python3-dolfinx-complex` and `python3-slepc4py-64-complex` (Ubuntu 24.04).
- Uses a simple unit-square mesh with clamped boundaries for portability.
  For circular-plate validation, compare against `estimate_first_bending_mode`
  separately (geometry/BCs differ).
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


def main() -> int:
    _ensure_repo_on_path()

    try:
        from mpi4py import MPI
        from dolfinx.mesh import create_unit_square
    except Exception as e:
        print(f"DOLFINx not available: {e}")
        return 0

    from bempp_audio.fea import MindlinReissnerPlateSolver, TITANIUM

    m = create_unit_square(MPI.COMM_WORLD, 30, 30)
    solver = MindlinReissnerPlateSolver(m, material=TITANIUM, degree=1, shear_quadrature_degree=1)

    # Clamped boundary: mark all exterior facets with marker=1.
    tdim = m.topology.dim
    m.topology.create_connectivity(tdim - 1, tdim)
    import numpy as np
    from dolfinx.mesh import exterior_facet_indices, meshtags

    facets = exterior_facet_indices(m.topology)
    tags = meshtags(m, tdim - 1, facets, np.full(len(facets), 1, dtype=np.int32))
    solver.set_facet_tags(tags)
    solver.add_clamped_bc(marker=1)

    modal = solver.solve_modes(num_modes=6)
    if m.comm.rank == 0:
        print("Computed modal frequencies (Hz):")
        for f in modal.frequencies_hz:
            print(f"  {f:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

