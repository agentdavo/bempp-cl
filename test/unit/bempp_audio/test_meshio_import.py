from __future__ import annotations

import numpy as np


def test_loudspeaker_mesh_from_meshio_triangle(tmp_path):
    import meshio

    from bempp_audio.mesh import LoudspeakerMesh

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    cells = [("triangle", np.array([[0, 1, 2]], dtype=int))]
    mesh = meshio.Mesh(points=points, cells=cells)

    path = tmp_path / "tri.vtk"
    meshio.write(path, mesh)

    bempp_mesh = LoudspeakerMesh.from_meshio(str(path))
    assert bempp_mesh.grid.number_of_elements == 1

