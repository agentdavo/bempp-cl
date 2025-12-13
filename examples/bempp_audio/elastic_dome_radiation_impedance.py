#!/usr/bin/env python3
"""
Compute radiation impedance of a (shell) dome using a 2D surface mesh + Helmholtz BEM.

This demonstrates the intended pipeline:
  (1) Build a thin-shell dome surface mesh (2D manifold embedded in 3D)
  (2) Define a (possibly non-uniform) normal-velocity distribution on the surface
  (3) Solve exterior acoustics via BEM and compute radiation impedance

Run from repo root:
  PYTHONPATH=. python examples/bempp_audio/elastic_dome_radiation_impedance.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

import numpy as np

from bempp_audio._optional import optional_import
from bempp_audio.fea import DomeGeometry, DomeMesher
from bempp_audio.fea.fem_bem_coupling import compute_radiation_impedance


def _element_centroids(grid) -> np.ndarray:
    vertices = np.asarray(grid.vertices, dtype=float)  # (3, n_vertices)
    elements = np.asarray(grid.elements, dtype=int)  # (3, n_elements)
    p0 = vertices[:, elements[0, :]]
    p1 = vertices[:, elements[1, :]]
    p2 = vertices[:, elements[2, :]]
    centroids = (p0 + p1 + p2) / 3.0
    return centroids.T


def main() -> None:
    gmsh, has_gmsh = optional_import("gmsh")
    bempp, has_bempp = optional_import("bempp_cl.api")
    if not has_gmsh or not has_bempp:
        raise SystemExit("Requires `gmsh` and `bempp_cl` (run inside the project venv).")

    geo = DomeGeometry.spherical(base_diameter_m=0.035, dome_height_m=0.008)
    mesh = DomeMesher(geo).generate(element_size_m=0.0015)
    grid = mesh.to_bempp_grid()

    centroids = _element_centroids(grid)
    r = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)
    r0 = float(geo.base_radius_m)

    # Example "elastic-ish" axisymmetric velocity shape: highest at center, reduced at rim.
    # Replace this with vn = j*omega*w_n from a shell FEM solution.
    vn_shape = np.clip(1.0 - (r / r0) ** 2, 0.0, 1.0)
    vn = vn_shape.astype(np.complex128)

    freqs = [200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
    print(f"Dome: base_diameter={geo.base_diameter_m*1e3:.1f} mm, height={geo.dome_height_m*1e3:.1f} mm")
    print(f"Mesh: elements={grid.number_of_elements}, vertices={grid.number_of_vertices}")
    print("f(Hz)   Z_rad (Pa·s/m^3) [real, imag]")

    for f in freqs:
        z = compute_radiation_impedance(grid, float(f), vn)
        print(f"{f:6.0f}  {np.real(z): .6e}  {np.imag(z): .6e}")


if __name__ == "__main__":
    main()
