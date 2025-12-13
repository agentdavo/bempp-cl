"""Tests for profile curvature utilities in bempp_audio.mesh.profiles."""

from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.mesh.profiles import (
    conical_profile,
    exponential_profile,
    compute_profile_curvature,
    compute_curvature_radius,
    check_min_curvature_radius,
    curvature_based_element_size,
)


class TestComputeProfileCurvature:
    """Tests for compute_profile_curvature."""

    def test_straight_line_zero_curvature(self):
        """A conical (linear) profile should have zero curvature."""
        x = np.linspace(0, 0.1, 100)
        r = conical_profile(x, 0.01, 0.05, 0.1)

        curvature = compute_profile_curvature(x, r)

        # Curvature should be near zero for a straight line
        assert np.allclose(curvature, 0, atol=1e-6)

    def test_exponential_nonzero_curvature(self):
        """An exponential profile should have non-zero curvature."""
        x = np.linspace(0, 0.1, 100)
        r = exponential_profile(x, 0.01, 0.05, 0.1)

        curvature = compute_profile_curvature(x, r)

        # Exponential profiles have positive curvature
        assert np.max(curvature) > 0

    def test_curvature_units(self):
        """Curvature should have units of 1/m."""
        x = np.linspace(0, 0.1, 100)
        r = exponential_profile(x, 0.01, 0.05, 0.1)

        curvature = compute_profile_curvature(x, r)

        # Curvature should be reasonable (not astronomical)
        # For typical waveguide profiles, curvature < 1000 1/m
        assert np.max(curvature) < 1000

    def test_short_array_returns_zeros(self):
        """Arrays with < 3 points should return zeros."""
        x = np.array([0, 0.1])
        r = np.array([0.01, 0.05])

        curvature = compute_profile_curvature(x, r)

        assert len(curvature) == 2
        assert np.allclose(curvature, 0)

    def test_same_length_output(self):
        """Output should have same length as input."""
        x = np.linspace(0, 0.1, 50)
        r = exponential_profile(x, 0.01, 0.05, 0.1)

        curvature = compute_profile_curvature(x, r)

        assert len(curvature) == len(x)


class TestComputeCurvatureRadius:
    """Tests for compute_curvature_radius."""

    def test_straight_line_large_radius(self):
        """A straight line should have very large radius of curvature."""
        x = np.linspace(0, 0.1, 100)
        r = conical_profile(x, 0.01, 0.05, 0.1)

        radii = compute_curvature_radius(x, r)

        # Large radius expected (capped at 1e6)
        assert np.min(radii) > 1e3

    def test_exponential_finite_radius(self):
        """An exponential profile should have finite radius of curvature."""
        x = np.linspace(0, 0.1, 100)
        r = exponential_profile(x, 0.01, 0.05, 0.1)

        radii = compute_curvature_radius(x, r)

        # Finite positive radius
        assert np.min(radii) > 0
        assert np.max(radii) < 1e6

    def test_minimum_radius_applied(self):
        """Minimum radius parameter should be applied."""
        x = np.linspace(0, 0.1, 100)
        r = exponential_profile(x, 0.01, 0.05, 0.1)

        radii = compute_curvature_radius(x, r, min_radius=0.001)

        assert np.min(radii) >= 0.001


class TestCheckMinCurvatureRadius:
    """Tests for check_min_curvature_radius."""

    def test_conical_passes_any_constraint(self):
        """A conical profile should pass any curvature constraint."""
        x = np.linspace(0, 0.1, 100)
        r = conical_profile(x, 0.01, 0.05, 0.1)

        ok, actual_min, location = check_min_curvature_radius(x, r, 0.001)

        assert ok is True
        assert actual_min > 0.001

    def test_returns_location_of_minimum(self):
        """Should return the x-position of minimum curvature radius."""
        x = np.linspace(0, 0.1, 100)
        r = exponential_profile(x, 0.01, 0.05, 0.1)

        ok, actual_min, location = check_min_curvature_radius(x, r, 0.001)

        # Location should be within bounds
        assert 0 <= location <= 0.1

    def test_fails_with_strict_constraint(self):
        """Should fail if constraint is stricter than actual minimum."""
        x = np.linspace(0, 0.1, 100)
        r = exponential_profile(x, 0.01, 0.05, 0.1)

        # Get actual minimum first
        _, actual_min, _ = check_min_curvature_radius(x, r, 0.0)

        # Now test with stricter constraint
        ok, _, _ = check_min_curvature_radius(x, r, actual_min * 10)

        assert ok is False


class TestCurvatureBasedElementSize:
    """Tests for curvature_based_element_size."""

    def test_straight_line_uses_h_max(self):
        """A straight line should use maximum element size."""
        x = np.linspace(0, 0.1, 100)
        r = conical_profile(x, 0.01, 0.05, 0.1)

        h = curvature_based_element_size(x, r, h_min=0.001, h_max=0.01)

        # Should be near h_max for straight line
        assert np.allclose(h, 0.01, atol=0.001)

    def test_curved_region_uses_smaller_size(self):
        """Curved regions should get smaller element sizes."""
        x = np.linspace(0, 0.1, 100)
        r = exponential_profile(x, 0.01, 0.05, 0.1)

        h = curvature_based_element_size(x, r, h_min=0.001, h_max=0.01)

        # Size should be between h_min and h_max
        assert np.min(h) >= 0.001
        assert np.max(h) <= 0.01

        # Should not be uniform (curved profile)
        assert np.std(h) > 0

    def test_size_bounded_by_h_min_h_max(self):
        """Element sizes should be bounded by h_min and h_max."""
        x = np.linspace(0, 0.1, 100)
        r = exponential_profile(x, 0.01, 0.05, 0.1)

        h = curvature_based_element_size(x, r, h_min=0.002, h_max=0.008)

        assert np.all(h >= 0.002)
        assert np.all(h <= 0.008)

    def test_curvature_factor_increases_refinement(self):
        """Higher curvature factor should increase refinement."""
        x = np.linspace(0, 0.1, 100)
        r = exponential_profile(x, 0.01, 0.05, 0.1)

        h_low = curvature_based_element_size(x, r, h_min=0.001, h_max=0.01, curvature_factor=0.5)
        h_high = curvature_based_element_size(x, r, h_min=0.001, h_max=0.01, curvature_factor=2.0)

        # Higher factor should give smaller mean element size
        assert np.mean(h_high) < np.mean(h_low)
