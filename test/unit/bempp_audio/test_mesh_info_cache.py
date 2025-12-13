from __future__ import annotations

import numpy as np
import pytest

from bempp_audio._optional import optional_import


def test_mesh_info_is_cached():
    bempp, ok = optional_import("bempp_cl.api")
    if not ok:
        pytest.skip("requires bempp_cl.api")

    from bempp_audio.mesh.loudspeaker_mesh import LoudspeakerMesh

    vertices = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    elements = np.array([[0], [1], [2]], dtype=int)
    grid = bempp.Grid(vertices, elements, domain_indices=np.array([1], dtype=int))
    mesh = LoudspeakerMesh(grid)

    assert getattr(mesh, "_info_cache", None) is None
    info1 = mesh.info()
    info2 = mesh.info()
    assert info1 is info2
    assert getattr(mesh, "_info_cache", None) is info1

