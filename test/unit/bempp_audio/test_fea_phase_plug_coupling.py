"""Tests for phase plug coupling and metrics in bempp_audio.fea."""

from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.fea.phase_plug_coupling import (
    ThroatExitData,
    PhasePlugMetrics,
    PressureFieldVisualization,
    throat_to_bem_monopole,
    throat_to_bem_piston,
    create_axial_sample_points,
    create_azimuthal_sample_points,
    summarize_metrics_sweep,
    AIR_DENSITY,
    SPEED_OF_SOUND,
)


class TestThroatExitData:
    """Tests for ThroatExitData dataclass."""

    def create_sample_data(self, n_points: int = 10) -> ThroatExitData:
        """Create sample throat data for testing."""
        # Create circular throat
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        r = 0.0125  # 25mm diameter throat
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full(n_points, 0.006)  # throat at z=6mm

        coords = np.column_stack([x, y, z])

        # Equal areas for each point (approximation)
        total_area = np.pi * r**2
        areas = np.full(n_points, total_area / n_points)

        # Uniform pressure and velocity
        pressure = np.full(n_points, 100.0 + 10j, dtype=complex)  # ~100 Pa
        velocity = np.full(n_points, 0.1 + 0.01j, dtype=complex)  # ~0.1 m/s

        return ThroatExitData(
            pressure=pressure,
            velocity=velocity,
            coordinates=coords,
            areas=areas,
            frequency_hz=1000.0,
            throat_radius_m=r,
        )

    def test_total_area(self):
        """Total area should equal sum of element areas."""
        data = self.create_sample_data()
        expected = np.sum(data.areas)
        assert np.isclose(data.total_area_m2, expected)

    def test_mean_pressure(self):
        """Mean pressure should be area-weighted average."""
        data = self.create_sample_data()
        # For uniform pressure, mean should equal the pressure value
        assert np.isclose(np.abs(data.mean_pressure), np.abs(data.pressure[0]), rtol=0.01)

    def test_mean_velocity(self):
        """Mean velocity should be area-weighted average."""
        data = self.create_sample_data()
        assert np.isclose(np.abs(data.mean_velocity), np.abs(data.velocity[0]), rtol=0.01)

    def test_volume_velocity(self):
        """Volume velocity should be integral of v·dS."""
        data = self.create_sample_data()
        expected = np.sum(data.velocity * data.areas)
        assert np.isclose(data.volume_velocity, expected)

    def test_acoustic_power_positive(self):
        """Acoustic power should be positive for positive power flow."""
        data = self.create_sample_data()
        # With pressure and velocity in phase, power should be positive
        assert data.acoustic_power_w > 0

    def test_acoustic_impedance(self):
        """Acoustic impedance should be pressure/volume velocity."""
        data = self.create_sample_data()
        z = data.acoustic_impedance
        assert np.isfinite(z)
        # Z = <p> / U, so |Z| should be reasonable
        assert np.abs(z) > 0

    def test_specific_impedance(self):
        """Specific impedance should be pressure/velocity."""
        data = self.create_sample_data()
        z_spec = data.specific_impedance
        assert np.isfinite(z_spec)
        # For p=100Pa, v=0.1m/s, z ≈ 1000 Pa·s/m
        assert np.abs(z_spec) > 100

    def test_non_uniform_pressure(self):
        """Test with non-uniform pressure distribution."""
        data = self.create_sample_data()
        # Modify to have varying pressure
        data.pressure[::2] *= 1.5  # Every other point has 50% higher pressure

        # Mean should still be computed correctly
        expected_mean = np.sum(data.pressure * data.areas) / data.total_area_m2
        assert np.isclose(data.mean_pressure, expected_mean)


class TestPhasePlugMetrics:
    """Tests for PhasePlugMetrics dataclass."""

    def test_metrics_creation(self):
        """Metrics should be creatable with all fields."""
        metrics = PhasePlugMetrics(
            frequency_hz=1000.0,
            dome_power_w=0.1,
            throat_power_w=0.095,
            transmission_efficiency=0.95,
            dome_impedance=1000 + 100j,
            throat_impedance=500 + 50j,
            impedance_transformation=2.0 + 0.1j,
            pressure_uniformity=0.98,
            phase_spread_deg=5.0,
        )

        assert metrics.frequency_hz == 1000.0
        assert metrics.transmission_efficiency == 0.95
        assert metrics.pressure_uniformity == 0.98


class TestPressureFieldVisualization:
    """Tests for PressureFieldVisualization class."""

    def test_pressure_magnitude(self):
        """Pressure magnitude should be absolute value."""
        coords = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
        pressure = np.array([100 + 10j, 50 - 50j, 0 + 100j])

        viz = PressureFieldVisualization(
            coordinates=coords,
            pressure=pressure,
            frequency_hz=1000.0,
        )

        expected = np.abs(pressure)
        np.testing.assert_array_almost_equal(viz.pressure_magnitude, expected)

    def test_pressure_phase(self):
        """Pressure phase should be in degrees."""
        coords = np.array([[0, 0, 0], [0.01, 0, 0]])
        pressure = np.array([1 + 0j, 0 + 1j])  # 0° and 90°

        viz = PressureFieldVisualization(
            coordinates=coords,
            pressure=pressure,
            frequency_hz=1000.0,
        )

        assert np.isclose(viz.pressure_phase_deg[0], 0.0, atol=1)
        assert np.isclose(viz.pressure_phase_deg[1], 90.0, atol=1)

    def test_spl_db(self):
        """SPL should be 20*log10(|p|/20µPa)."""
        coords = np.array([[0, 0, 0]])
        pressure = np.array([20e-6])  # Reference pressure = 0 dB

        viz = PressureFieldVisualization(
            coordinates=coords,
            pressure=pressure,
            frequency_hz=1000.0,
        )

        assert np.isclose(viz.spl_db[0], 0.0, atol=0.1)

    def test_spl_db_typical(self):
        """Test SPL for typical pressure levels."""
        coords = np.array([[0, 0, 0]])
        pressure = np.array([2.0])  # 2 Pa ≈ 100 dB

        viz = PressureFieldVisualization(
            coordinates=coords,
            pressure=pressure,
            frequency_hz=1000.0,
        )

        # 2 Pa = 100 dB SPL
        assert np.isclose(viz.spl_db[0], 100.0, atol=0.1)

    def test_to_dict(self):
        """to_dict should include all fields."""
        coords = np.array([[0, 0, 0], [0.01, 0, 0]])
        pressure = np.array([100 + 10j, 50 - 50j])

        viz = PressureFieldVisualization(
            coordinates=coords,
            pressure=pressure,
            frequency_hz=1000.0,
        )

        data = viz.to_dict()

        assert data['frequency_hz'] == 1000.0
        assert len(data['coordinates']) == 2
        assert len(data['pressure_real']) == 2
        assert len(data['pressure_imag']) == 2


class TestThroatToBEM:
    """Tests for throat to BEM coupling functions."""

    def create_throat_data(self) -> ThroatExitData:
        """Create sample throat data."""
        n_points = 8
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        r = 0.0125

        coords = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta),
            np.full(n_points, 0.006),
        ])

        total_area = np.pi * r**2
        areas = np.full(n_points, total_area / n_points)

        return ThroatExitData(
            pressure=np.full(n_points, 100.0, dtype=complex),
            velocity=np.full(n_points, 0.1, dtype=complex),
            coordinates=coords,
            areas=areas,
            frequency_hz=1000.0,
            throat_radius_m=r,
        )

    def test_monopole_position(self):
        """Monopole should be at throat center."""
        data = self.create_throat_data()
        position, strength = throat_to_bem_monopole(data)

        # Center should be at origin (x=0, y=0)
        assert np.isclose(position[0], 0.0, atol=1e-10)
        assert np.isclose(position[1], 0.0, atol=1e-10)
        # Z should be at throat position
        assert np.isclose(position[2], 0.006, atol=1e-6)

    def test_monopole_strength(self):
        """Monopole strength should equal volume velocity."""
        data = self.create_throat_data()
        position, strength = throat_to_bem_monopole(data)

        assert np.isclose(strength, data.volume_velocity)

    def test_piston_data(self):
        """throat_to_bem_piston should return correct data."""
        data = self.create_throat_data()
        result = throat_to_bem_piston(data)

        assert 'mean_velocity' in result
        assert 'velocity_distribution' in result
        assert 'throat_radius_m' in result
        assert 'throat_area_m2' in result

        assert np.isclose(result['mean_velocity'], data.mean_velocity)
        assert np.isclose(result['throat_radius_m'], data.throat_radius_m)


class TestSamplePointCreation:
    """Tests for sample point creation functions."""

    def test_axial_sample_points_shape(self):
        """Axial sample points should have correct shape."""
        points = create_axial_sample_points(
            r_max=0.01,
            z_min=0.0,
            z_max=0.01,
            n_r=10,
            n_z=20,
        )

        assert points.shape == (200, 3)  # 10 * 20 points

    def test_axial_sample_points_y_zero(self):
        """Axial sample points should be on y=0 plane."""
        points = create_axial_sample_points(
            r_max=0.01,
            z_min=0.0,
            z_max=0.01,
        )

        np.testing.assert_array_almost_equal(points[:, 1], 0.0)

    def test_axial_sample_points_bounds(self):
        """Axial sample points should respect bounds."""
        r_max = 0.015
        z_min = 0.001
        z_max = 0.008

        points = create_axial_sample_points(
            r_max=r_max,
            z_min=z_min,
            z_max=z_max,
        )

        # X should be in [0, r_max] (on y=0 plane, x = r)
        assert np.min(points[:, 0]) >= 0
        assert np.max(points[:, 0]) <= r_max + 1e-10

        # Z should be in [z_min, z_max]
        assert np.min(points[:, 2]) >= z_min - 1e-10
        assert np.max(points[:, 2]) <= z_max + 1e-10

    def test_azimuthal_sample_points_shape(self):
        """Azimuthal sample points should have correct shape."""
        points = create_azimuthal_sample_points(
            r=0.01,
            z_min=0.0,
            z_max=0.01,
            n_theta=18,
            n_z=25,
        )

        assert points.shape == (450, 3)  # 18 * 25 points

    def test_azimuthal_sample_points_radius(self):
        """Azimuthal sample points should be at correct radius."""
        r = 0.012

        points = create_azimuthal_sample_points(
            r=r,
            z_min=0.0,
            z_max=0.01,
        )

        # Check radius: sqrt(x² + y²) = r
        radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        np.testing.assert_array_almost_equal(radii, r, decimal=10)


class TestMetricsSweepSummary:
    """Tests for metrics sweep summary."""

    def test_summarize_empty_list(self):
        """Should handle empty list gracefully."""
        # With empty list, numpy operations will produce empty arrays
        # This tests that the function doesn't crash
        with pytest.raises((IndexError, ValueError)):
            summarize_metrics_sweep([])

    def test_summarize_single_frequency(self):
        """Should work with single frequency."""
        metrics = PhasePlugMetrics(
            frequency_hz=1000.0,
            dome_power_w=0.1,
            throat_power_w=0.09,
            transmission_efficiency=0.9,
            dome_impedance=1000 + 100j,
            throat_impedance=500 + 50j,
            impedance_transformation=2.0,
            pressure_uniformity=0.95,
            phase_spread_deg=10.0,
        )

        summary = summarize_metrics_sweep([metrics])

        assert len(summary['frequencies_hz']) == 1
        assert summary['mean_efficiency'] == 0.9
        assert summary['min_uniformity'] == 0.95
        assert summary['max_phase_spread'] == 10.0

    def test_summarize_multiple_frequencies(self):
        """Should compute correct summary statistics."""
        metrics_list = [
            PhasePlugMetrics(
                frequency_hz=1000.0,
                dome_power_w=0.1,
                throat_power_w=0.09,
                transmission_efficiency=0.9,
                dome_impedance=1000,
                throat_impedance=500,
                impedance_transformation=2.0,
                pressure_uniformity=0.98,
                phase_spread_deg=5.0,
            ),
            PhasePlugMetrics(
                frequency_hz=5000.0,
                dome_power_w=0.08,
                throat_power_w=0.064,
                transmission_efficiency=0.8,
                dome_impedance=800,
                throat_impedance=400,
                impedance_transformation=2.0,
                pressure_uniformity=0.85,
                phase_spread_deg=15.0,
            ),
        ]

        summary = summarize_metrics_sweep(metrics_list)

        assert len(summary['frequencies_hz']) == 2
        assert np.isclose(summary['mean_efficiency'], 0.85)  # (0.9 + 0.8) / 2
        assert np.isclose(summary['min_uniformity'], 0.85)
        assert np.isclose(summary['max_phase_spread'], 15.0)

    def test_summarize_keys(self):
        """Summary should contain all expected keys."""
        metrics = PhasePlugMetrics(
            frequency_hz=1000.0,
            dome_power_w=0.1,
            throat_power_w=0.09,
            transmission_efficiency=0.9,
            dome_impedance=1000,
            throat_impedance=500,
            impedance_transformation=2.0,
            pressure_uniformity=0.95,
            phase_spread_deg=10.0,
        )

        summary = summarize_metrics_sweep([metrics])

        expected_keys = [
            'frequencies_hz',
            'transmission_efficiency',
            'pressure_uniformity',
            'phase_spread_deg',
            'impedance_magnitude',
            'mean_efficiency',
            'min_uniformity',
            'max_phase_spread',
        ]

        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"


class TestAcousticPowerCalculation:
    """Tests for acoustic power calculation in ThroatExitData."""

    def test_power_formula(self):
        """Acoustic power should follow P = 0.5 * Re{∫ p v* dS}."""
        n = 4
        coords = np.zeros((n, 3))
        areas = np.full(n, 0.001)  # 1 cm² each

        # In-phase pressure and velocity
        pressure = np.full(n, 100.0, dtype=complex)  # 100 Pa
        velocity = np.full(n, 0.1, dtype=complex)  # 0.1 m/s

        data = ThroatExitData(
            pressure=pressure,
            velocity=velocity,
            coordinates=coords,
            areas=areas,
            frequency_hz=1000.0,
            throat_radius_m=0.01,
        )

        # Power = 0.5 * Re{p * v*} * A = 0.5 * 100 * 0.1 * 0.004 = 0.02 W
        expected = 0.5 * 100.0 * 0.1 * 0.004
        assert np.isclose(data.acoustic_power_w, expected)

    def test_power_with_phase_difference(self):
        """Power should reduce with phase difference."""
        n = 4
        coords = np.zeros((n, 3))
        areas = np.full(n, 0.001)

        # 90° phase difference between p and v
        pressure = np.full(n, 100.0, dtype=complex)
        velocity = np.full(n, 0.1j, dtype=complex)  # 90° phase

        data = ThroatExitData(
            pressure=pressure,
            velocity=velocity,
            coordinates=coords,
            areas=areas,
            frequency_hz=1000.0,
            throat_radius_m=0.01,
        )

        # Power should be zero for 90° phase difference (reactive)
        assert np.isclose(data.acoustic_power_w, 0.0, atol=1e-10)

    def test_power_negative_for_absorption(self):
        """Power should be negative if flow direction is reversed."""
        n = 4
        coords = np.zeros((n, 3))
        areas = np.full(n, 0.001)

        # Opposite phase (180°) = power flows opposite direction
        pressure = np.full(n, 100.0, dtype=complex)
        velocity = np.full(n, -0.1, dtype=complex)

        data = ThroatExitData(
            pressure=pressure,
            velocity=velocity,
            coordinates=coords,
            areas=areas,
            frequency_hz=1000.0,
            throat_radius_m=0.01,
        )

        assert data.acoustic_power_w < 0
