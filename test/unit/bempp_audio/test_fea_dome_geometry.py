from __future__ import annotations

import numpy as np

from bempp_audio.fea.dome_geometry import DomeGeometry


def test_dome_geometry_profiles_follow_coordinate_convention():
    # Coordinate convention:
    # - apex at z=0 => r=0
    # - base at z=h => r=base_radius
    base_diameter_m = 0.035
    dome_height_m = 0.008

    for profile in ["spherical", "elliptical", "parabolic", "conical"]:
        geo = DomeGeometry(
            base_diameter_m=base_diameter_m,
            dome_height_m=dome_height_m,
            profile=profile,
        )

        assert np.isclose(geo.radius_at_z(0.0), 0.0, atol=1e-12)
        assert np.isclose(geo.radius_at_z(geo.dome_height_m), geo.base_radius_m, atol=1e-9)

        zs = np.linspace(0.0, geo.dome_height_m, 200)
        rs = np.array([geo.radius_at_z(float(z)) for z in zs])
        # Radius should be monotone non-decreasing as z goes from apex->base.
        assert np.min(np.diff(rs)) >= -1e-10


def test_dome_geometry_inverse_is_consistent():
    geo = DomeGeometry.spherical(base_diameter_m=0.035, dome_height_m=0.008)

    rs = np.linspace(0.0, geo.base_radius_m, 40)
    zs = np.array([geo.z_at_radius(float(r)) for r in rs])
    assert np.min(zs) >= -1e-12
    assert np.max(zs) <= geo.dome_height_m + 1e-12

    # Inverse round-trip away from the base clamp
    for r, z in zip(rs[:-1], zs[:-1]):
        assert np.isclose(geo.radius_at_z(float(z)), float(r), atol=1e-9)
