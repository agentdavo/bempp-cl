from __future__ import annotations

import numpy as np

from bempp_audio.mesh.profiles import (
    conical_profile,
    exponential_profile,
    tractrix_profile,
    hyperbolic_profile,
    cts_profile,
    cts_mouth_angle_deg,
    compute_exit_cone_angle_deg,
)


def test_profile_endpoints_match():
    x = np.linspace(0.0, 0.1, 11)
    r0 = 0.01
    r1 = 0.05

    for fn in (conical_profile, exponential_profile, tractrix_profile, hyperbolic_profile):
        r = fn(x, r0, r1, x[-1])
        assert np.isclose(r[0], r0)
        assert np.isclose(r[-1], r1)

    r = cts_profile(x, r0, r1, x[-1], transition=0.75)
    assert np.isclose(r[0], r0)
    assert np.isclose(r[-1], r1)


def test_profiles_monotone_non_decreasing():
    x = np.linspace(0.0, 0.1, 101)
    r = exponential_profile(x, 0.01, 0.05, x[-1])
    assert np.all(np.diff(r) >= -1e-12)

    r = cts_profile(x, 0.01, 0.05, x[-1], transition=0.75)
    assert np.all(np.diff(r) >= -1e-12)


def test_cts_tangency_endpoints():
    """CTS profile endpoints should match regardless of tangency."""
    x = np.linspace(0.0, 0.1, 101)
    r0 = 0.01
    r1 = 0.05

    for tangency in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = cts_profile(x, r0, r1, x[-1], transition=0.75, tangency=tangency)
        assert np.isclose(r[0], r0), f"tangency={tangency}: r(0) mismatch"
        assert np.isclose(r[-1], r1), f"tangency={tangency}: r(L) mismatch"


def test_cts_tangency_mouth_slope():
    """CTS tangency parameter should control wall angle at mouth.

    Wall angle is measured from z-axis:
    - tangency=0: conical exit (same angle throughout)
    - tangency=1: maximum achievable angle for natural profile shape

    Note: The achievable tangency is limited by geometry. Higher expansion ratios
    allow for more tangent mouths. The algorithm prioritizes natural-looking
    profiles over extreme tangency.
    """
    x = np.linspace(0.0, 0.1, 1001)  # Fine resolution for derivative
    r0 = 0.01
    r1 = 0.05
    L = x[-1]
    dx = x[1] - x[0]

    # tangency=1.0 should give a larger slope at mouth than tangency=0
    # The actual slope depends on the geometry (expansion ratio)
    r_t1 = cts_profile(x, r0, r1, L, transition=0.75, tangency=1.0)
    mouth_slope_t1 = (r_t1[-1] - r_t1[-2]) / dx

    r_t0 = cts_profile(x, r0, r1, L, transition=0.75, tangency=0.0)
    mouth_slope_t0 = (r_t0[-1] - r_t0[-2]) / dx

    # tangency=1.0 should have a steeper mouth than tangency=0.0
    assert mouth_slope_t1 > mouth_slope_t0, (
        f"tangency=1.0 mouth slope ({mouth_slope_t1:.2f}) should exceed "
        f"tangency=0.0 ({mouth_slope_t0:.2f})"
    )

    # tangency=0.0 should give conical (constant) slope
    mid_slope = (r_t0[500] - r_t0[499]) / dx
    assert abs(mouth_slope_t0 - mid_slope) < 0.01, f"tangency=0.0 should have constant slope"


def test_cts_tangency_monotone():
    """CTS profile should remain monotone for all tangency values."""
    x = np.linspace(0.0, 0.1, 101)
    r0 = 0.01
    r1 = 0.05

    for tangency in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = cts_profile(x, r0, r1, x[-1], transition=0.75, tangency=tangency)
        assert np.all(np.diff(r) >= -1e-12), f"tangency={tangency}: not monotone"


def test_cts_mouth_angle_deg():
    """cts_mouth_angle_deg should return correct mouth wall angle.

    Wall angle is measured from z-axis:
    - 0 deg = vertical wall (parallel to z-axis)
    - 90 deg = horizontal wall (tangent to XY baffle at z=0)

    Note: tangency=1.0 gives the MAXIMUM ACHIEVABLE angle for the geometry,
    which may be less than 90 deg due to geometric constraints.
    """
    r0 = 0.01
    r1 = 0.05
    L = 0.1

    # tangency=0.0 should give the conical angle
    angle_t0 = cts_mouth_angle_deg(r0, r1, L, transition=0.75, tangency=0.0)
    conical_angle = np.rad2deg(np.arctan((r1 - r0) / L))
    assert abs(angle_t0 - conical_angle) < 0.1, f"tangency=0.0 should give conical angle {conical_angle:.1f}, got {angle_t0}"

    # tangency=1.0 should give a larger angle than tangency=0 (max achievable)
    # The achievable angle depends on geometry (expansion ratio)
    angle_t1 = cts_mouth_angle_deg(r0, r1, L, transition=0.75, tangency=1.0)
    assert angle_t1 > angle_t0, f"tangency=1.0 should give larger angle than tangency=0"
    # For this geometry (5x expansion), max mouth angle ~55 deg
    # The angle should be significantly above conical but limited by geometry
    assert angle_t1 > conical_angle * 1.5, (
        f"tangency=1.0 should give significant angle increase above conical ({conical_angle:.1f}), got {angle_t1}"
    )

    # tangency=0.5 should be between tangency=0 and tangency=1
    angle_t05 = cts_mouth_angle_deg(r0, r1, L, transition=0.75, tangency=0.5)
    assert angle_t0 < angle_t05 < angle_t1, f"tangency=0.5 should be between t0 and t1, got {angle_t05}"


def test_compute_exit_cone_angle_deg():
    """compute_exit_cone_angle_deg should compute correct half-angle."""
    # Known geometry: 10mm throat -> 20mm mouth over 28.87mm length = 10 deg half-angle
    # tan(10 deg) = 0.1763, delta_r = 5mm, length = 5/0.1763 = 28.37mm
    throat_d = 0.010  # 10mm
    mouth_d = 0.020   # 20mm
    length = 0.02837  # ~28.37mm for 10 deg

    angle = compute_exit_cone_angle_deg(throat_d, mouth_d, length)
    assert abs(angle - 10.0) < 0.1, f"Expected ~10 deg, got {angle}"


def test_cts_throat_blend_endpoints():
    """CTS profile with throat_blend should still match endpoints."""
    x = np.linspace(0.0, 0.1, 101)
    r0 = 0.01
    r1 = 0.05
    L = x[-1]

    # Test with various throat_blend values
    for throat_blend in [0.0, 0.1, 0.2]:
        r = cts_profile(
            x, r0, r1, L,
            throat_blend=throat_blend,
            transition=0.75,
            tangency=1.0,
        )
        assert np.isclose(r[0], r0), f"throat_blend={throat_blend}: r(0) mismatch"
        assert np.isclose(r[-1], r1, rtol=1e-3), f"throat_blend={throat_blend}: r(L) mismatch"


def test_cts_throat_blend_with_driver_angle():
    """CTS profile should blend from driver exit angle to conical."""
    x = np.linspace(0.0, 0.1, 1001)
    r0 = 0.01
    r1 = 0.05
    L = x[-1]
    dx = x[1] - x[0]

    # Driver exit angle of 5 degrees (narrower than conical)
    # Conical would be arctan((r1-r0)/L) = arctan(0.04/0.1) = 21.8 deg
    r = cts_profile(
        x, r0, r1, L,
        throat_blend=0.15,
        transition=0.75,
        driver_exit_angle_deg=5.0,
        tangency=1.0,
    )

    # Check endpoints
    assert np.isclose(r[0], r0), "r(0) should match throat_r"
    assert np.isclose(r[-1], r1, rtol=1e-3), "r(L) should match mouth_r"

    # Check slope at throat should be close to tan(5 deg)
    throat_slope = (r[1] - r[0]) / dx
    expected_throat_slope = np.tan(np.deg2rad(5.0))
    assert abs(throat_slope - expected_throat_slope) < 0.05, \
        f"Throat slope should be ~{expected_throat_slope:.3f}, got {throat_slope:.3f}"

    # Profile should be monotone
    assert np.all(np.diff(r) >= -1e-12), "Profile should be monotone"


def test_cts_throat_blend_monotone():
    """CTS profile with throat blend should remain monotone."""
    x = np.linspace(0.0, 0.1, 101)
    r0 = 0.01
    r1 = 0.05
    L = x[-1]

    # Test with driver angle both smaller and larger than conical
    for driver_angle in [5.0, 15.0, 25.0]:
        for throat_blend in [0.1, 0.2]:
            r = cts_profile(
                x, r0, r1, L,
                throat_blend=throat_blend,
                transition=0.75,
                driver_exit_angle_deg=driver_angle,
                tangency=1.0,
            )
            assert np.all(np.diff(r) >= -1e-12), \
                f"driver_angle={driver_angle}, throat_blend={throat_blend}: not monotone"


def test_cts_3section_profile_shape():
    """Verify 3-section CTS profile has expected shape characteristics."""
    x = np.linspace(0.0, 0.1, 1001)
    r0 = 0.01
    r1 = 0.05
    L = x[-1]
    dx = x[1] - x[0]

    # Create profile with throat blend
    throat_blend = 0.15
    transition = 0.75
    driver_angle = 10.0  # degrees

    r = cts_profile(
        x, r0, r1, L,
        throat_blend=throat_blend,
        transition=transition,
        driver_exit_angle_deg=driver_angle,
        tangency=1.0,
    )

    # Compute local slope (wall angle) along the profile
    slopes = np.diff(r) / dx
    angles_deg = np.rad2deg(np.arctan(slopes))

    # At throat (t=0), angle should be close to driver_angle
    assert abs(angles_deg[0] - driver_angle) < 2.0, \
        f"Throat angle should be ~{driver_angle}, got {angles_deg[0]:.1f}"

    # In conical section (around t=0.5), angle should be relatively constant
    mid_idx = int(0.5 * len(angles_deg))
    mid_angles = angles_deg[mid_idx-50:mid_idx+50]
    angle_std = np.std(mid_angles)
    assert angle_std < 1.0, f"Conical section should have constant angle, std={angle_std:.2f}"

    # At mouth, angle should be larger than conical (if tangency > 0)
    mouth_angle = angles_deg[-1]
    conical_angle = np.rad2deg(np.arctan((r1 - r0) / L))
    assert mouth_angle > conical_angle + 5, \
        f"Mouth angle ({mouth_angle:.1f}) should be > conical ({conical_angle:.1f})"


def test_cts_mouth_roll_biases_termination_to_be_earlier():
    """mouth_roll should bias the mouth blend toward the final angle earlier.

    Since mouth_roll can also change the *final* mouth angle (due to the endpoint
    constraint), compare normalized blend progress relative to each profile's
    own (α_conical, α_mouth).
    """
    x = np.linspace(0.0, 0.1, 2001)
    r0 = 0.01
    r1 = 0.05
    L = x[-1]
    dx = x[1] - x[0]

    r_plain = cts_profile(x, r0, r1, L, transition=0.75, tangency=1.0, mouth_roll=0.0, throat_angle_deg=20.0)
    r_roll = cts_profile(x, r0, r1, L, transition=0.75, tangency=1.0, mouth_roll=0.8, throat_angle_deg=20.0)

    slopes_plain = np.diff(r_plain) / dx
    slopes_roll = np.diff(r_roll) / dx
    angles_plain = np.rad2deg(np.arctan(slopes_plain))
    angles_roll = np.rad2deg(np.arctan(slopes_roll))

    alpha_c_plain = 20.0
    alpha_c_roll = 20.0
    alpha_m_plain = float(angles_plain[-1])
    alpha_m_roll = float(angles_roll[-1])

    # Compare at mid mouth-blend (t≈0.875 for transition=0.75).
    idx = int(0.875 * len(angles_plain))

    def _progress(alpha_t: float, alpha_c: float, alpha_m: float) -> float:
        denom = max(1e-9, float(alpha_m - alpha_c))
        return float((alpha_t - alpha_c) / denom)

    p_plain = _progress(float(angles_plain[idx]), alpha_c_plain, alpha_m_plain)
    p_roll = _progress(float(angles_roll[idx]), alpha_c_roll, alpha_m_roll)
    assert p_roll >= p_plain - 1e-6


def test_cts_mid_curvature_keeps_endpoints_and_monotonic():
    x = np.linspace(0.0, 0.1, 501)
    r0 = 0.01
    r1 = 0.05
    L = x[-1]

    r = cts_profile(x, r0, r1, L, transition=0.75, tangency=1.0, mid_curvature=0.3)
    assert np.isclose(r[0], r0)
    assert np.isclose(r[-1], r1, rtol=1e-3)
    assert np.all(np.diff(r) >= -1e-12)
