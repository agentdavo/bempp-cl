from __future__ import annotations

import os
import numpy as np
import pytest

from bempp_audio._optional import optional_import


@pytest.mark.skipif(os.environ.get("BEMPPAUDIO_RUN_SLOW") != "1", reason="set BEMPPAUDIO_RUN_SLOW=1 to enable")
def test_waveguide_morph_rectangle_loft_mouth_extents_and_domains():
    gmsh, has_gmsh = optional_import("gmsh")
    bempp, has_bempp = optional_import("bempp_cl.api")
    if not has_gmsh or not has_bempp:
        pytest.skip("requires gmsh and bempp_cl")

    from bempp_audio.mesh.waveguide import WaveguideMeshConfig, WaveguideMeshGenerator
    from bempp_audio.mesh.morph import MorphConfig, MorphTargetShape

    target_w = 0.12
    target_h = 0.06

    cfg = WaveguideMeshConfig(
        throat_diameter=0.02,
        mouth_diameter=0.10,
        length=0.05,
        profile_type="conical",
        n_axial_slices=16,
        n_circumferential=48,
        h_throat=0.01,
        h_mouth=0.02,
        morph=MorphConfig(
            target_shape=MorphTargetShape.RECTANGLE,
            target_width=target_w,
            target_height=target_h,
            corner_radius=0.005,
            allow_shrinkage=True,
        ),
    )

    mesh, throat_mask = WaveguideMeshGenerator(cfg).generate()
    assert mesh.grid.number_of_elements > 0
    assert int(throat_mask.sum()) > 0

    domains = set(mesh.grid.domain_indices.tolist())
    assert domains == {1, 2}

    V = mesh.grid.vertices
    z = V[2]
    mouth = np.isclose(z, 0.0, atol=1e-8)
    assert mouth.any(), "expected vertices on the mouth plane (z=0)"

    x = V[0, mouth]
    y = V[1, mouth]
    mouth_w = float(x.max() - x.min())
    mouth_h = float(y.max() - y.min())

    assert abs(mouth_w - target_w) < 0.005
    assert abs(mouth_h - target_h) < 0.005

