#!/usr/bin/env python3
"""
Generate and preview the canonical Benchmark 3.0-Ti dome mesh.

This is a geometry + meshing smoke test (not a physics validation).
It exports an interactive Plotly HTML mesh for quick inspection and prints
derived parameters (sphere radius, clamp band radii) for reproducibility.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


def _mesh_to_plotly_html(vertices, triangles, out_html: Path, title: str) -> None:
    import numpy as np
    import plotly.graph_objects as go

    v = np.asarray(vertices, dtype=float)
    t = np.asarray(triangles, dtype=int)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=t[:, 0],
                j=t[:, 1],
                k=t[:, 2],
                opacity=0.7,
                color="lightsteelblue",
                flatshading=True,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs=True)


def main() -> int:
    _ensure_repo_on_path()

    import numpy as np

    from bempp_audio.fea import Benchmark3TiConfig_v1, DomeMesher, recommended_mesh_size

    cfg = Benchmark3TiConfig_v1()
    geom = cfg.dome_geometry()

    print("Benchmark3TiConfig_v1")
    print(f"  rise:               {cfg.rise_m*1e3:.3f} mm")
    print(f"  clamp outer radius: {cfg.clamp_outer_radius_m*1e3:.3f} mm")
    print(f"  clamp inner radius: {cfg.clamp_inner_radius_m*1e3:.3f} mm")
    print(f"  sphere radius:      {cfg.sphere_radius_m*1e3:.3f} mm")
    print(f"  material:           {cfg.material.name}, t={cfg.material.thickness_m*1e6:.1f} µm")

    h = recommended_mesh_size(max_frequency_hz=20000.0, material=cfg.material, elements_per_wavelength=6)
    h = float(min(h, 1.0e-3))  # keep the preview mesh modest by default
    mesh = DomeMesher(geom).generate(element_size_m=h)

    mask = cfg.clamp_vertex_mask_for_mesh(mesh)
    n_clamp = int(np.count_nonzero(mask))
    print(f"  mesh:               {mesh.num_vertices} vertices, {mesh.num_triangles} tris (h={h*1e3:.3f} mm)")
    print(f"  clamp vertices:     {n_clamp} ({100.0*n_clamp/max(1,mesh.num_vertices):.1f}%)")

    out_dir = Path(os.environ.get("BEMPPAUDIO_OUT_DIR", "logs")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / f"{Path(__file__).stem}_mesh.html"
    _mesh_to_plotly_html(mesh.vertices, mesh.triangles, out_html, title="Benchmark 3.0-Ti (v1) dome mesh")
    print(f"  preview:            {out_html}")

    # Print a quick quality summary (does not fail the script).
    report = mesh.validate_quality(material=cfg.material, max_frequency_hz=20000.0)
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
