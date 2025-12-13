"""Tests for directivity sweep metrics in bempp_audio.results.directivity."""

from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.results.directivity import DirectivitySweepMetrics


class TestDirectivitySweepMetrics:
    """Tests for DirectivitySweepMetrics dataclass."""

    def test_di_ripple_calculation(self):
        """DI ripple should be peak-to-peak variation."""
        metrics = DirectivitySweepMetrics(
            frequencies=np.array([1000, 2000, 4000, 8000]),
            di_values=np.array([5.0, 6.0, 5.5, 7.0]),  # min=5, max=7
            beamwidth_values=np.array([90, 80, 70, 60]),
        )

        assert np.isclose(metrics.di_ripple_db, 2.0)

    def test_di_std_calculation(self):
        """DI std should be standard deviation."""
        metrics = DirectivitySweepMetrics(
            frequencies=np.array([1000, 2000, 4000, 8000]),
            di_values=np.array([5.0, 5.0, 5.0, 5.0]),  # uniform
            beamwidth_values=np.array([90, 80, 70, 60]),
        )

        assert np.isclose(metrics.di_std_db, 0.0)

    def test_beamwidth_monotonicity_perfect(self):
        """Perfectly decreasing beamwidth should have monotonicity 1.0."""
        metrics = DirectivitySweepMetrics(
            frequencies=np.array([1000, 2000, 4000, 8000]),
            di_values=np.array([5.0, 6.0, 7.0, 8.0]),
            beamwidth_values=np.array([90, 80, 70, 60]),  # Always decreasing
        )

        assert np.isclose(metrics.beamwidth_monotonicity, 1.0)

    def test_beamwidth_monotonicity_partial(self):
        """Partially monotonic beamwidth should have intermediate score."""
        metrics = DirectivitySweepMetrics(
            frequencies=np.array([1000, 2000, 4000, 8000]),
            di_values=np.array([5.0, 6.0, 7.0, 8.0]),
            beamwidth_values=np.array([90, 80, 85, 60]),  # One increase
        )

        # 3 steps: decrease, increase, decrease -> 2/3 monotonic
        assert np.isclose(metrics.beamwidth_monotonicity, 2 / 3)

    def test_beamwidth_monotonicity_single_point(self):
        """Single frequency should return 1.0 monotonicity."""
        metrics = DirectivitySweepMetrics(
            frequencies=np.array([1000]),
            di_values=np.array([5.0]),
            beamwidth_values=np.array([90]),
        )

        assert metrics.beamwidth_monotonicity == 1.0

    def test_beamwidth_ripple(self):
        """Beamwidth ripple should be peak-to-peak variation."""
        metrics = DirectivitySweepMetrics(
            frequencies=np.array([1000, 2000, 4000, 8000]),
            di_values=np.array([5.0, 6.0, 7.0, 8.0]),
            beamwidth_values=np.array([90, 60, 80, 50]),  # min=50, max=90
        )

        assert np.isclose(metrics.beamwidth_ripple_deg, 40.0)

    def test_coverage_angle_mean(self):
        """Coverage angle should be mean beamwidth."""
        metrics = DirectivitySweepMetrics(
            frequencies=np.array([1000, 2000, 4000, 8000]),
            di_values=np.array([5.0, 6.0, 7.0, 8.0]),
            beamwidth_values=np.array([100, 80, 60, 40]),
        )

        assert np.isclose(metrics.coverage_angle_mean_deg, 70.0)

    def test_is_smooth_passes(self):
        """Smooth directivity should pass criteria."""
        metrics = DirectivitySweepMetrics(
            frequencies=np.array([1000, 2000, 4000, 8000]),
            di_values=np.array([5.0, 5.5, 6.0, 6.5]),  # Ripple = 1.5 dB
            beamwidth_values=np.array([90, 85, 75, 65]),  # Monotonic
        )

        assert metrics.is_smooth(max_di_ripple_db=3.0, min_monotonicity=0.7)

    def test_is_smooth_fails_di_ripple(self):
        """High DI ripple should fail smoothness check."""
        metrics = DirectivitySweepMetrics(
            frequencies=np.array([1000, 2000, 4000, 8000]),
            di_values=np.array([3.0, 7.0, 4.0, 8.0]),  # Ripple = 5 dB
            beamwidth_values=np.array([90, 85, 75, 65]),
        )

        assert not metrics.is_smooth(max_di_ripple_db=3.0, min_monotonicity=0.7)

    def test_is_smooth_fails_monotonicity(self):
        """Non-monotonic beamwidth should fail smoothness check."""
        metrics = DirectivitySweepMetrics(
            frequencies=np.array([1000, 2000, 4000, 8000]),
            di_values=np.array([5.0, 5.5, 6.0, 6.5]),
            beamwidth_values=np.array([90, 100, 110, 120]),  # Increasing!
        )

        assert not metrics.is_smooth(max_di_ripple_db=3.0, min_monotonicity=0.7)

    def test_summary_format(self):
        """Summary should be a readable multi-line string."""
        metrics = DirectivitySweepMetrics(
            frequencies=np.array([1000, 2000, 4000, 8000]),
            di_values=np.array([5.0, 6.0, 7.0, 8.0]),
            beamwidth_values=np.array([90, 80, 70, 60]),
        )

        summary = metrics.summary()

        assert "Directivity Sweep Metrics" in summary
        assert "DI ripple" in summary
        assert "Beamwidth monotonicity" in summary
        assert "1000" in summary  # Frequency range
        assert "8000" in summary
