from __future__ import annotations

import numpy as np

from bempp_audio.analysis.hom_proxy import disk_sample_points, _metrics_from_magnitude


def test_disk_sample_points_shapes():
    pts, radii, thetas = disk_sample_points(center=(0.0, 0.0, 0.0), radius=0.1, n_r=5, n_theta=16)
    assert pts.shape[0] == 3
    assert pts.shape[1] == radii.shape[0] == thetas.shape[0]
    assert np.isfinite(pts).all()


def test_metrics_uniform_field_is_zeroish():
    _, radii, _ = disk_sample_points(center=(0.0, 0.0, 0.0), radius=0.1, n_r=6, n_theta=24)
    mag = np.ones_like(radii) * 2.0
    m = _metrics_from_magnitude(mag, radii)
    assert m.cv_mag == 0.0
    assert m.rms_rel == 0.0
    assert m.azimuthal_rms_rel == 0.0
    assert m.radial_rms_rel == 0.0


def test_metrics_detect_radial_gradient():
    _, radii, _ = disk_sample_points(center=(0.0, 0.0, 0.0), radius=0.1, n_r=6, n_theta=24)
    mag = 1.0 + 2.0 * (radii / radii.max())
    m = _metrics_from_magnitude(mag, radii)
    assert m.cv_mag > 0.0
    assert m.rms_rel > 0.0
    assert m.radial_rms_rel > 0.0

