from __future__ import annotations

import os
import numpy as np
import pytest

from bempp_audio._optional import optional_import


@pytest.mark.skipif(os.environ.get("BEMPPAUDIO_RUN_SLOW") != "1", reason="set BEMPPAUDIO_RUN_SLOW=1 to enable")
def test_unified_enclosure_morphed_mouth_cutout_reaches_target_extents():
    gmsh, has_gmsh = optional_import("gmsh")
    bempp, has_bempp = optional_import("bempp_cl.api")
    if not has_gmsh or not has_bempp:
        pytest.skip("requires gmsh and bempp_cl")

    from bempp_audio.mesh.unified_enclosure import UnifiedMeshConfig, create_waveguide_on_box_unified
    from bempp_audio.mesh.morph import MorphConfig, MorphTargetShape

    target_w = 0.10
    target_h = 0.05

    cfg = UnifiedMeshConfig(
        throat_diameter=0.02,
        mouth_diameter=0.08,  # base circular mouth (smaller than target width)
        waveguide_length=0.04,
        box_width=0.20,
        box_height=0.15,
        box_depth=0.15,
        n_axial_slices=6,
        n_circumferential=24,
        h_throat=0.015,
        h_mouth=0.03,
        h_box=0.04,
        morph=MorphConfig(
            target_shape=MorphTargetShape.RECTANGLE,
            target_width=target_w,
            target_height=target_h,
            corner_radius=0.004,
            allow_shrinkage=True,
        ),
    )

    mesh = create_waveguide_on_box_unified(cfg)
    domains = set(mesh.grid.domain_indices.tolist())
    assert domains == {1, 2, 3}

    V = mesh.grid.vertices
    mouth = np.isclose(V[2], 0.0, atol=1e-8)
    assert mouth.any()

    x = V[0, mouth]
    y = V[1, mouth]

    p_right = np.array([float(cfg.mount_x + target_w / 2), float(cfg.mount_y)])
    p_top = np.array([float(cfg.mount_x), float(cfg.mount_y + target_h / 2)])

    d_right = np.sqrt((x - p_right[0]) ** 2 + (y - p_right[1]) ** 2).min()
    d_top = np.sqrt((x - p_top[0]) ** 2 + (y - p_top[1]) ** 2).min()

    assert float(d_right) < 0.008
    assert float(d_top) < 0.008
