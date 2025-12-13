from __future__ import annotations

import numpy as np
import pytest

from bempp_audio._optional import optional_import


def test_mesh_quality_plotly_exports(tmp_path):
    bempp, ok = optional_import("bempp_cl.api")
    if not ok:
        pytest.skip("requires bempp_cl.api")

    try:
        import plotly  # noqa: F401
    except Exception:
        pytest.skip("requires plotly")

    from bempp_audio.mesh.loudspeaker_mesh import LoudspeakerMesh
    from bempp_audio.viz import save_edge_length_histogram_html, save_element_quality_colormap_html

    vertices = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    elements = np.array([[0, 1], [1, 3], [2, 2]], dtype=int)
    grid = bempp.Grid(vertices, elements, domain_indices=np.asarray([1, 1], dtype=int))
    mesh = LoudspeakerMesh(grid)

    p1 = tmp_path / "edges.html"
    p2 = tmp_path / "quality.html"

    save_edge_length_histogram_html(mesh, p1)
    save_element_quality_colormap_html(mesh, p2, metric="aspect_ratio")

    assert p1.exists()
    assert p2.exists()

