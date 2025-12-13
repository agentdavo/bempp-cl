from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.viz.data.normalization import normalize_spl_map_by_angle, symmetric_polar_map


def test_normalize_spl_map_by_angle_uses_nearest_angle_and_subtracts_reference():
    angles = np.array([0.0, 5.0, 10.0])
    freqs = np.array([100.0, 200.0])

    # SPL map shape (n_angles, n_freqs)
    spl = np.array(
        [
            [90.0, 91.0],  # 0°
            [80.0, 81.0],  # 5°
            [70.0, 71.0],  # 10°
        ]
    )

    norm = normalize_spl_map_by_angle(
        angles_deg=angles,
        frequencies_hz=freqs,
        spl_map_db=spl,
        normalize_angle_deg=6.0,  # nearest is 5°
    )

    assert norm.norm_angle_used_deg == pytest.approx(5.0)
    assert norm.norm_angle_index == 1
    assert norm.spl_ref_db.tolist() == pytest.approx([80.0, 81.0])

    expected = spl - np.array([80.0, 81.0])[None, :]
    assert norm.spl_map_db == pytest.approx(expected)


def test_symmetric_polar_map_mirrors_without_duplicating_zero_angle():
    angles = np.array([0.0, 5.0, 10.0])
    spl = np.array(
        [
            [0.0, 0.0],
            [-1.0, -2.0],
            [-3.0, -4.0],
        ]
    )
    angles_sym, spl_sym = symmetric_polar_map(angles_deg=angles, spl_map_db=spl)

    assert angles_sym.tolist() == pytest.approx([-10.0, -5.0, 0.0, 5.0, 10.0])
    assert spl_sym.shape == (5, 2)
    assert spl_sym[2, :].tolist() == pytest.approx([0.0, 0.0])  # 0° appears once

