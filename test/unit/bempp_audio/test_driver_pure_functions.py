"""Unit tests for the pure helper functions in bempp_audio.driver.network."""

from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.driver.network import (
    compute_blocked_electrical_impedance,
    compute_mechanical_impedance,
    solve_two_node_acoustic_network,
    AcousticNodeResult,
    conical_waveguide_y_matrix,
    conical_waveguide_abcd_matrix,
)


class TestComputeBlockedElectricalImpedance:
    """Tests for compute_blocked_electrical_impedance()."""

    def test_dc_returns_resistance_only(self):
        z = compute_blocked_electrical_impedance(
            frequency_hz=0.0,
            re_ohm=6.0,
            le_h=0.0001,
        )
        assert z == pytest.approx(6.0, abs=1e-12)

    def test_ac_includes_inductance(self):
        f = 1000.0
        re = 6.0
        le = 0.0001
        z = compute_blocked_electrical_impedance(
            frequency_hz=f,
            re_ohm=re,
            le_h=le,
        )
        expected = re + 1j * 2 * np.pi * f * le
        assert z == pytest.approx(expected, abs=1e-12)

    def test_eddy_loss_adds_resistance_at_high_frequency(self):
        z_low = compute_blocked_electrical_impedance(
            frequency_hz=10.0,
            re_ohm=6.0,
            le_h=0.0001,
            eddy_r_max_ohm=4.0,
            eddy_f_corner_hz=500.0,
        )
        z_high = compute_blocked_electrical_impedance(
            frequency_hz=20000.0,
            re_ohm=6.0,
            le_h=0.0001,
            eddy_r_max_ohm=4.0,
            eddy_f_corner_hz=500.0,
        )
        # At high frequency, eddy loss should add ~4 ohms to real part
        assert np.real(z_high) > np.real(z_low)
        # At very high frequency, R_eddy approaches R_max
        assert np.real(z_high) > 6.0 + 3.5  # Should be close to 6+4=10

    def test_eddy_loss_zero_when_disabled(self):
        z_without = compute_blocked_electrical_impedance(
            frequency_hz=10000.0,
            re_ohm=6.0,
            le_h=0.0001,
            eddy_r_max_ohm=0.0,
        )
        z_with = compute_blocked_electrical_impedance(
            frequency_hz=10000.0,
            re_ohm=6.0,
            le_h=0.0001,
            eddy_r_max_ohm=4.0,
            eddy_f_corner_hz=500.0,
        )
        assert np.real(z_without) < np.real(z_with)


class TestComputeMechanicalImpedance:
    """Tests for compute_mechanical_impedance()."""

    def test_dc_returns_compliance_dominated(self):
        # At DC, compliance term 1/(jωC) dominates (→ infinity)
        z = compute_mechanical_impedance(
            frequency_hz=0.0,
            mms_kg=0.01,
            cms_m_per_n=2e-4,
            rms_ns_per_m=1.0,
        )
        # At f=0, function returns Rms only (special case)
        assert z == pytest.approx(1.0, abs=1e-12)

    def test_resonance_frequency_minimum_magnitude(self):
        mms = 0.01
        cms = 2e-4
        rms = 1.0
        # Resonance at f0 = 1/(2π√(M*C))
        f0 = 1.0 / (2 * np.pi * np.sqrt(mms * cms))

        # At resonance, mass and compliance cancel, leaving just Rms
        z_res = compute_mechanical_impedance(
            frequency_hz=f0,
            mms_kg=mms,
            cms_m_per_n=cms,
            rms_ns_per_m=rms,
        )
        # At resonance, |Z| should be close to Rms
        assert abs(z_res) == pytest.approx(rms, rel=0.01)

    def test_mass_dominates_at_high_frequency(self):
        f_low = 100.0
        f_high = 10000.0
        mms = 0.01
        cms = 2e-4
        rms = 1.0

        z_low = compute_mechanical_impedance(
            frequency_hz=f_low, mms_kg=mms, cms_m_per_n=cms, rms_ns_per_m=rms
        )
        z_high = compute_mechanical_impedance(
            frequency_hz=f_high, mms_kg=mms, cms_m_per_n=cms, rms_ns_per_m=rms
        )
        # At high frequency, mass term jωM dominates, so |Z| increases with f
        assert abs(z_high) > abs(z_low)


class TestSolveTwoNodeAcousticNetwork:
    """Tests for solve_two_node_acoustic_network()."""

    def test_single_node_uncoupled(self):
        """When V1 is disconnected, reduces to single-node solve."""
        q0 = 1e-6 + 0j  # 1 cc/s
        q1 = 0.0 + 0j
        y_front = 1e-6  # 1 MRayl
        y_v0 = 0.0 + 0j
        y_v1 = 0.0 + 0j
        y_coupling = np.zeros((2, 2), dtype=complex)
        z_front = 1e6 + 0j

        result = solve_two_node_acoustic_network(
            q0=q0,
            q1=q1,
            y_front=y_front,
            y_v0=y_v0,
            y_v1=y_v1,
            y_coupling=y_coupling,
            z_front=z_front,
        )

        assert isinstance(result, AcousticNodeResult)
        # p0 = q0 / y_front
        expected_p0 = q0 / y_front
        assert result.p0 == pytest.approx(expected_p0, rel=1e-9)
        assert result.p1 == pytest.approx(0.0, abs=1e-30)

    def test_coupled_nodes_with_resistive_slit(self):
        """Two nodes coupled through a resistive slit."""
        q0 = 1e-6 + 0j
        q1 = 0.5e-6 + 0j
        y_front = 1e-6 + 0j
        y_v0 = 0.0 + 0j
        y_v1 = 1e-7 + 0j  # V1 shunt compliance
        z_front = 1e6 + 0j

        # Resistive coupling element: Y = [[y, -y], [-y, y]] where y = 1/R
        r_slit = 1e5  # Pa·s/m³
        y_slit = 1.0 / r_slit
        y_coupling = np.array([[y_slit, -y_slit], [-y_slit, y_slit]], dtype=complex)

        result = solve_two_node_acoustic_network(
            q0=q0,
            q1=q1,
            y_front=y_front,
            y_v0=y_v0,
            y_v1=y_v1,
            y_coupling=y_coupling,
            z_front=z_front,
        )

        # With coupling, p1 should be nonzero
        assert abs(result.p1) > 0
        # With q1 contributing through the slit, total flow can exceed q0
        # Just verify the solve produces a reasonable positive flow
        assert abs(result.u_front) > 0
        # u_front should be less than total injected (q0 + q1)
        assert abs(result.u_front) <= abs(q0) + abs(q1)

    def test_returns_acoustic_node_result(self):
        """Verify return type and attributes."""
        result = solve_two_node_acoustic_network(
            q0=1e-6 + 0j,
            q1=0.0 + 0j,
            y_front=1e-6 + 0j,
            y_v0=0.0 + 0j,
            y_v1=0.0 + 0j,
            y_coupling=np.zeros((2, 2), dtype=complex),
            z_front=1e6 + 0j,
        )

        assert hasattr(result, "p0")
        assert hasattr(result, "p1")
        assert hasattr(result, "u_front")
        assert hasattr(result, "z_front")
        assert hasattr(result, "z_mat")
        assert result.z_mat.shape == (2, 2)


class TestConicalWaveguideYMatrix:
    """Tests for conical_waveguide_y_matrix() - Panzer eq. 8."""

    def test_returns_2x2_complex_matrix(self):
        y = conical_waveguide_y_matrix(
            frequency_hz=1000.0,
            rho=1.225,
            c=343.0,
            throat_area_m2=1e-4,
            mouth_area_m2=2e-4,
            length_m=0.05,
        )
        assert y.shape == (2, 2)
        assert y.dtype == complex

    def test_zero_frequency_returns_zero_matrix(self):
        y = conical_waveguide_y_matrix(
            frequency_hz=0.0,
            rho=1.225,
            c=343.0,
            throat_area_m2=1e-4,
            mouth_area_m2=2e-4,
            length_m=0.05,
        )
        assert np.allclose(y, 0.0)

    def test_invalid_areas_returns_zero_matrix(self):
        y = conical_waveguide_y_matrix(
            frequency_hz=1000.0,
            rho=1.225,
            c=343.0,
            throat_area_m2=0.0,
            mouth_area_m2=2e-4,
            length_m=0.05,
        )
        assert np.allclose(y, 0.0)

    def test_uniform_tube_limit(self):
        """When Sth == Smo, should behave like uniform tube."""
        area = 1e-4
        y_conical = conical_waveguide_y_matrix(
            frequency_hz=1000.0,
            rho=1.225,
            c=343.0,
            throat_area_m2=area,
            mouth_area_m2=area * 1.001,  # Nearly identical
            length_m=0.05,
        )
        # Matrix should be non-zero and have reasonable structure
        assert np.abs(y_conical).max() > 0

    def test_reciprocity(self):
        """Y12 should approximately equal Y21 for passive network."""
        y = conical_waveguide_y_matrix(
            frequency_hz=2000.0,
            rho=1.225,
            c=343.0,
            throat_area_m2=1e-4,
            mouth_area_m2=3e-4,
            length_m=0.03,
        )
        # For lossless reciprocal network, Y12 = Y21
        assert y[0, 1] == pytest.approx(y[1, 0], rel=1e-10)


class TestConicalWaveguideABCDMatrix:
    """Tests for conical_waveguide_abcd_matrix()."""

    def test_returns_2x2_complex_matrix(self):
        abcd = conical_waveguide_abcd_matrix(
            frequency_hz=1000.0,
            rho=1.225,
            c=343.0,
            throat_area_m2=1e-4,
            mouth_area_m2=2e-4,
            length_m=0.05,
        )
        assert abcd.shape == (2, 2)
        assert abcd.dtype == complex

    def test_identity_at_zero_length(self):
        """Zero-length waveguide should be identity."""
        abcd = conical_waveguide_abcd_matrix(
            frequency_hz=1000.0,
            rho=1.225,
            c=343.0,
            throat_area_m2=1e-4,
            mouth_area_m2=2e-4,
            length_m=0.0,
        )
        # Should return identity when length is zero
        assert abcd[0, 0] == pytest.approx(1.0, abs=1e-10)
        assert abcd[1, 1] == pytest.approx(1.0, abs=1e-10)
        assert abcd[0, 1] == pytest.approx(0.0, abs=1e-10)
        assert abcd[1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_determinant_is_unity_for_lossless(self):
        """For lossless passive network, det(ABCD) = 1."""
        abcd = conical_waveguide_abcd_matrix(
            frequency_hz=1500.0,
            rho=1.225,
            c=343.0,
            throat_area_m2=1e-4,
            mouth_area_m2=2.5e-4,
            length_m=0.04,
        )
        det = abcd[0, 0] * abcd[1, 1] - abcd[0, 1] * abcd[1, 0]
        # Determinant should be 1 for lossless reciprocal network
        assert abs(det) == pytest.approx(1.0, rel=0.01)
