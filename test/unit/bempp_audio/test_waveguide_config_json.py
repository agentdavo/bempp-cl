from __future__ import annotations

import json

from bempp_audio.mesh.morph import MorphConfig
from bempp_audio.mesh.waveguide import WaveguideMeshConfig


def test_waveguide_mesh_config_json_roundtrip(tmp_path):
    cfg = WaveguideMeshConfig(
        throat_diameter=0.025,
        mouth_diameter=0.150,
        length=0.100,
        profile_type="exponential",
        n_axial_slices=20,
        n_circumferential=24,
        meridian_curve="polyline",
        mouth_edge_refine=True,
        mouth_edge_size=0.004,
        mouth_edge_dist_min=0.01,
        mouth_edge_dist_max=0.03,
        mouth_edge_sampling=50,
        morph=MorphConfig.rectangle(width=0.12, height=0.06, corner_radius=0.006, allow_shrinkage=True),
    )

    path = tmp_path / "waveguide_config.json"
    cfg.to_json(path)

    data = json.loads(path.read_text())
    assert data["throat_diameter"] == 0.025
    assert data["meridian_curve"] == "polyline"
    assert bool(data["mouth_edge_refine"]) is True
    assert abs(float(data["mouth_edge_size"]) - 0.004) < 1e-12
    assert data["morph"]["target_shape"] == "RECTANGLE"
    assert abs(float(data["morph"]["corner_radius"]) - 0.006) < 1e-12

    cfg2 = WaveguideMeshConfig.from_json(path)
    assert cfg2.throat_diameter == cfg.throat_diameter
    assert cfg2.meridian_curve == "polyline"
    assert bool(cfg2.mouth_edge_refine) is True
    assert abs(float(cfg2.mouth_edge_size or 0.0) - 0.004) < 1e-12
    assert cfg2.morph is not None
    assert cfg2.morph.target_shape.name == "RECTANGLE"
    assert abs(float(cfg2.morph.corner_radius) - 0.006) < 1e-12
