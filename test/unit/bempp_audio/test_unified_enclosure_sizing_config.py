from __future__ import annotations

from bempp_audio.mesh.unified_enclosure import UnifiedMeshConfig


def test_unified_mesh_config_sizing_defaults():
    cfg = UnifiedMeshConfig(
        throat_diameter=0.025,
        mouth_diameter=0.15,
        waveguide_length=0.1,
        box_width=0.3,
        box_height=0.4,
        box_depth=0.25,
    )
    assert cfg.h_baffle == cfg.h_mouth
    assert cfg.h_sides == cfg.h_box
    assert cfg.h_back == cfg.h_sides * 1.5

