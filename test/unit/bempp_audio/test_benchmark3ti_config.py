from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.fea import Benchmark3TiConfig_v1, Benchmark3TiFormerConfig_v1, Benchmark3TiSurroundConfig_v1, sweep_benchmark3ti_v1


def test_benchmark3ti_sphere_radius_matches_geometry_formula():
    cfg = Benchmark3TiConfig_v1()
    # Expected (a^2 + h^2)/(2h) with a=37.25mm and h=14.5mm ≈ 55.1mm.
    assert cfg.sphere_radius_m * 1e3 == pytest.approx(55.1, rel=2e-3)


def test_benchmark3ti_clamp_band_mask_selects_annulus():
    cfg = Benchmark3TiConfig_v1()
    # Create vertices at radii: 0, inner-ε, inner+ε, mid, outer-ε, outer+ε.
    eps = 1e-6
    radii = np.array(
        [
            0.0,
            cfg.clamp_inner_radius_m - eps,
            cfg.clamp_inner_radius_m + eps,
            0.5 * (cfg.clamp_inner_radius_m + cfg.clamp_outer_radius_m),
            cfg.clamp_outer_radius_m - eps,
            cfg.clamp_outer_radius_m + eps,
        ],
        dtype=float,
    )
    vertices = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=1)
    mask = cfg.clamp_vertex_mask(vertices)
    assert mask.tolist() == [False, False, True, True, True, False]


def test_benchmark3ti_former_added_mass_is_positive():
    cfg = Benchmark3TiFormerConfig_v1()
    assert cfg.former_mass_kg > 0
    assert cfg.adhesive_mass_kg > 0
    assert cfg.added_mass_kg == pytest.approx(cfg.former_mass_kg + cfg.adhesive_mass_kg)


def test_benchmark3ti_surround_mask_selects_ring():
    cfg = Benchmark3TiSurroundConfig_v1()
    eps = 1e-6
    radii = np.array(
        [
            cfg.surround_inner_radius_m - eps,
            cfg.surround_inner_radius_m + eps,
            0.5 * (cfg.surround_inner_radius_m + cfg.surround_outer_radius_m),
            cfg.surround_outer_radius_m - eps,
            cfg.surround_outer_radius_m + eps,
        ],
        dtype=float,
    )
    vertices = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=1)
    mask = cfg.surround_vertex_mask(vertices)
    assert mask.tolist() == [False, True, True, True, False]


def test_benchmark3ti_sweep_generates_expected_count():
    cfgs = sweep_benchmark3ti_v1(rise_m=[10e-3, 14.5e-3], thickness_m=[40e-6, 50e-6], structural_damping=[0.001, 0.002])
    assert len(cfgs) == 2 * 1 * 2 * 2
