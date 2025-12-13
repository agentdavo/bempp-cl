"""Tests for elastic dome network integration in bempp_audio.fea."""

from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.fea.elastic_dome_network import (
    ElasticDomeProfile,
    RigidVsElasticComparison,
)


class TestElasticDomeProfile:
    """Tests for ElasticDomeProfile class."""

    def test_from_rigid_piston_has_unity_ratio(self):
        """Rigid piston should have ratio = 1 at all frequencies."""
        area = 0.001  # 1000 mm²
        freqs = np.array([100, 1000, 10000])

        profile = ElasticDomeProfile.from_rigid_piston(area, freqs)

        assert np.allclose(profile.volume_velocity_ratio, 1.0)
        assert np.allclose(profile.effective_area_m2, area)
        assert profile.physical_area_m2 == area

    def test_from_fem_results_computes_ratio(self):
        """FEM results should compute correct ratio."""
        freqs = np.array([1000, 5000, 10000])
        n_elements = 10
        element_areas = np.full(n_elements, 0.0001)  # 1 cm² each
        total_area = element_areas.sum()  # 10 cm²

        # Create velocity distributions with decreasing efficiency
        velocity_distributions = {
            1000.0: np.ones(n_elements, dtype=complex),  # Unity velocity
            5000.0: np.ones(n_elements, dtype=complex) * 0.8,  # 80%
            10000.0: np.ones(n_elements, dtype=complex) * 0.5,  # 50%
        }

        # Voice coil velocity of 1.0 at all frequencies
        voice_coil_velocity = np.ones(3, dtype=complex)

        profile = ElasticDomeProfile.from_fem_results(
            frequencies_hz=freqs,
            velocity_distributions=velocity_distributions,
            element_areas=element_areas,
            voice_coil_velocity=voice_coil_velocity,
            diaphragm_area_m2=total_area,
        )

        # Ratios should reflect velocity distribution averages
        assert np.isclose(profile.volume_velocity_ratio[0], 1.0, rtol=0.01)
        assert np.isclose(profile.volume_velocity_ratio[1], 0.8, rtol=0.01)
        assert np.isclose(profile.volume_velocity_ratio[2], 0.5, rtol=0.01)

    def test_interpolate_ratio(self):
        """Ratio should interpolate between frequencies."""
        freqs = np.array([1000, 10000])
        area = 0.001

        profile = ElasticDomeProfile(
            frequencies_hz=freqs,
            effective_area_m2=np.array([area, 0.0005]),
            volume_velocity_ratio=np.array([1.0, 0.5]),
            physical_area_m2=area,
        )

        # Interpolate at 5500 Hz (midpoint)
        ratio = profile.interpolate_ratio(5500)
        assert ratio == pytest.approx(0.75, abs=0.01)

    def test_interpolate_effective_area(self):
        """Effective area should interpolate between frequencies."""
        freqs = np.array([1000, 10000])

        profile = ElasticDomeProfile(
            frequencies_hz=freqs,
            effective_area_m2=np.array([0.001, 0.0005]),
            volume_velocity_ratio=np.array([1.0, 0.5]),
            physical_area_m2=0.001,
        )

        area = profile.interpolate_effective_area(5500)
        assert area == pytest.approx(0.00075, abs=0.00001)

    def test_breakup_frequencies_finds_crossing(self):
        """Should find frequency where ratio drops below threshold."""
        freqs = np.array([1000, 2000, 3000, 4000, 5000])
        # Ratio drops from 1.0 to 0.3 (below -3dB = 0.71)
        ratios = np.array([1.0, 0.9, 0.6, 0.4, 0.3])

        profile = ElasticDomeProfile(
            frequencies_hz=freqs,
            effective_area_m2=np.full(5, 0.001),
            volume_velocity_ratio=ratios,
            physical_area_m2=0.001,
        )

        breakup = profile.breakup_frequencies(threshold_db=-3.0)

        # Should find crossing between 2000 and 3000 Hz
        assert len(breakup) == 1
        assert 2000 < breakup[0] < 3000

    def test_no_breakup_when_always_above_threshold(self):
        """No breakup should be found if ratio stays high."""
        freqs = np.array([1000, 5000, 10000])
        ratios = np.array([1.0, 0.95, 0.9])  # All above 0.71 (-3dB)

        profile = ElasticDomeProfile(
            frequencies_hz=freqs,
            effective_area_m2=np.full(3, 0.001),
            volume_velocity_ratio=ratios,
            physical_area_m2=0.001,
        )

        breakup = profile.breakup_frequencies(threshold_db=-3.0)
        assert len(breakup) == 0


class TestRigidVsElasticComparison:
    """Tests for rigid vs elastic comparison."""

    def test_from_sweep_results_computes_difference(self):
        """Should compute SPL difference between models."""
        freqs = np.array([1000, 5000, 10000])

        rigid_results = {
            'frequencies': freqs,
            'volume_velocity': np.array([1.0, 1.0, 1.0], dtype=complex),
        }

        elastic_results = {
            'frequencies': freqs,
            'volume_velocity': np.array([1.0, 0.8, 0.5], dtype=complex),
        }

        comparison = RigidVsElasticComparison.from_sweep_results(
            rigid_results, elastic_results, threshold_db=3.0
        )

        assert len(comparison.frequencies_hz) == 3
        assert len(comparison.spl_difference_db) == 3

        # First frequency should have zero difference
        assert np.isclose(comparison.spl_difference_db[0], 0.0, atol=0.1)

        # Later frequencies should have negative difference (elastic < rigid)
        assert comparison.spl_difference_db[2] < -5  # 0.5 = -6 dB

    def test_breakup_frequency_detection(self):
        """Should detect frequency where difference exceeds threshold."""
        freqs = np.array([1000, 2000, 3000, 4000])

        rigid_results = {
            'frequencies': freqs,
            'volume_velocity': np.array([1.0, 1.0, 1.0, 1.0], dtype=complex),
        }

        # Elastic drops significantly at 3000 Hz
        elastic_results = {
            'frequencies': freqs,
            'volume_velocity': np.array([1.0, 0.9, 0.4, 0.2], dtype=complex),
        }

        comparison = RigidVsElasticComparison.from_sweep_results(
            rigid_results, elastic_results, threshold_db=3.0
        )

        assert comparison.breakup_frequency_hz is not None
        # Breakup should be around 3000 Hz (where ratio drops below threshold)
        assert 2000 < comparison.breakup_frequency_hz <= 3000

    def test_no_breakup_when_always_close(self):
        """No breakup if difference stays within threshold."""
        freqs = np.array([1000, 5000, 10000])

        rigid_results = {
            'frequencies': freqs,
            'volume_velocity': np.array([1.0, 1.0, 1.0], dtype=complex),
        }

        # Small deviation that stays within 3 dB
        elastic_results = {
            'frequencies': freqs,
            'volume_velocity': np.array([1.0, 0.9, 0.85], dtype=complex),
        }

        comparison = RigidVsElasticComparison.from_sweep_results(
            rigid_results, elastic_results, threshold_db=3.0
        )

        # All differences should be within threshold
        assert comparison.breakup_frequency_hz is None


class TestElasticDomeNetworkAdapter:
    """Tests for network adapter (mock network)."""

    def test_adapter_applies_correction(self):
        """Adapter should multiply velocity by ratio."""
        freqs = np.array([1000, 5000, 10000])

        profile = ElasticDomeProfile(
            frequencies_hz=freqs,
            effective_area_m2=np.array([0.001, 0.0008, 0.0005]),
            volume_velocity_ratio=np.array([1.0, 0.8, 0.5]),
            physical_area_m2=0.001,
        )

        # Mock network class
        class MockNetwork:
            @property
            def diaphragm_area_m2(self):
                return 0.001

            def solve_volume_velocity(self, freq, *args, **kwargs):
                return 1.0 + 0j  # Always returns 1.0

            def solve_with_metrics(self, freq, *args, **kwargs):
                return {'volume_velocity': 1.0 + 0j}

        from bempp_audio.fea.elastic_dome_network import ElasticDomeNetworkAdapter

        adapter = ElasticDomeNetworkAdapter(MockNetwork(), profile)

        # At 1000 Hz, ratio = 1.0, so output = 1.0
        u1 = adapter.solve_volume_velocity(1000)
        assert np.isclose(abs(u1), 1.0, rtol=0.01)

        # At 10000 Hz, ratio = 0.5, so output = 0.5
        u10k = adapter.solve_volume_velocity(10000)
        assert np.isclose(abs(u10k), 0.5, rtol=0.01)

    def test_adapter_metrics_include_ratio(self):
        """Metrics should include ratio and effective area."""
        freqs = np.array([1000])

        profile = ElasticDomeProfile(
            frequencies_hz=freqs,
            effective_area_m2=np.array([0.0008]),
            volume_velocity_ratio=np.array([0.8]),
            physical_area_m2=0.001,
        )

        class MockNetwork:
            @property
            def diaphragm_area_m2(self):
                return 0.001

            def solve_volume_velocity(self, freq, *args, **kwargs):
                return 1.0 + 0j

            def solve_with_metrics(self, freq, *args, **kwargs):
                return {'volume_velocity': 1.0 + 0j}

        from bempp_audio.fea.elastic_dome_network import ElasticDomeNetworkAdapter

        adapter = ElasticDomeNetworkAdapter(MockNetwork(), profile)
        metrics = adapter.solve_with_metrics(1000)

        assert 'velocity_ratio' in metrics
        assert 'effective_area_m2' in metrics
        assert 'volume_velocity_rigid' in metrics
        assert np.isclose(metrics['velocity_ratio'], 0.8, rtol=0.01)
        assert np.isclose(metrics['effective_area_m2'], 0.0008, rtol=0.01)

    def test_frequency_sweep(self):
        """Frequency sweep should apply corrections at each frequency."""
        freqs = np.array([1000, 2000, 3000])

        profile = ElasticDomeProfile(
            frequencies_hz=freqs,
            effective_area_m2=np.array([0.001, 0.0009, 0.0008]),
            volume_velocity_ratio=np.array([1.0, 0.9, 0.8]),
            physical_area_m2=0.001,
        )

        class MockNetwork:
            @property
            def diaphragm_area_m2(self):
                return 0.001

            def solve_volume_velocity(self, freq, *args, **kwargs):
                return 1.0 + 0j

            def solve_with_metrics(self, freq, *args, **kwargs):
                return {'volume_velocity': 1.0 + 0j}

        from bempp_audio.fea.elastic_dome_network import ElasticDomeNetworkAdapter

        adapter = ElasticDomeNetworkAdapter(MockNetwork(), profile)
        sweep = adapter.frequency_sweep(freqs)

        assert len(sweep['volume_velocity']) == 3
        assert len(sweep['velocity_ratio']) == 3
        assert np.allclose(np.abs(sweep['velocity_ratio']), [1.0, 0.9, 0.8], rtol=0.01)
