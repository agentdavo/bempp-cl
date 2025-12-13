from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.driver import (
    AcousticMedium,
    CompressionDriverConfig,
    CompressionDriverExcitation,
    CompressionDriverNetwork,
    CompressionDriverNetworkOptions,
    DriverElectroMechConfig,
    FrontDuctConfig,
    RearVolumeConfig,
    plane_wave_tube_load_impedance,
)
from bempp_audio.driver.fitting import fit_vacuum_impedance
from bempp_audio.driver.validation import vacuum_electrical_impedance
from bempp_audio.driver.network import _kirchhoff_circular_duct_characteristics


def test_fit_vacuum_impedance_recovers_parameters_on_synthetic_curve():
    rng = np.random.default_rng(0)
    freqs = np.logspace(np.log10(50.0), np.log10(20000.0), 200)

    true = dict(
        re_ohm=6.2,
        le_h=1.7e-4,
        bl_tm=11.5,
        mms_kg=0.018,
        cms_m_per_n=1.2e-4,
        rms_ns_per_m=1.1,
        voice_coil_eddy_rmax_ohm=3.0,
        voice_coil_eddy_fcorner_hz=900.0,
    )

    z = np.array([vacuum_electrical_impedance(frequency_hz=float(f), **true) for f in freqs], dtype=complex)
    # Add small complex noise to avoid "perfect" conditioning in the unit test.
    noise = (0.01 * rng.standard_normal(z.shape)) + 1j * (0.01 * rng.standard_normal(z.shape))
    z_meas = z * (1.0 + noise)

    initial = dict(
        re_ohm=5.5,
        le_h=1.0e-4,
        bl_tm=9.0,
        mms_kg=0.012,
        cms_m_per_n=2.0e-4,
        rms_ns_per_m=0.7,
        voice_coil_eddy_rmax_ohm=1.0,
        voice_coil_eddy_fcorner_hz=1200.0,
    )

    fit = fit_vacuum_impedance(frequencies_hz=freqs, z_measured_ohm=z_meas, initial=initial, fit_eddy=True)
    assert fit.diagnostics.success

    # The vacuum fit is ill-conditioned (Bl, Mms, Cms trade off), so assert robust
    # parameters and that the fitted curve matches the synthetic measurement well.
    assert fit.x["re_ohm"] == pytest.approx(true["re_ohm"], rel=0.05)
    assert fit.x["le_h"] == pytest.approx(true["le_h"], rel=0.15)

    z_hat = np.array([vacuum_electrical_impedance(frequency_hz=float(f), **fit.x) for f in freqs], dtype=complex)
    rel_rms = float(np.sqrt(np.mean(np.abs(z_hat - z_meas) ** 2)) / np.sqrt(np.mean(np.abs(z_meas) ** 2)))
    assert rel_rms < 0.05


def test_callable_external_load_is_supported():
    medium = AcousticMedium(c=343.0, rho=1.225)
    cfg = CompressionDriverConfig(
        name="callable_load",
        driver=DriverElectroMechConfig(
            diaphragm_diameter_m=0.025,
            mms_kg=0.01,
            cms_m_per_n=2.0e-4,
            rms_ns_per_m=1.0,
            bl_tm=10.0,
            re_ohm=6.0,
            le_h=1.0e-4,
        ),
        rear_volume=RearVolumeConfig(volume_m3=200e-6),
        front_duct=FrontDuctConfig(diameter_m=0.025, length_m=0.02),
    )
    net = CompressionDriverNetwork(cfg, medium=medium)
    zc = plane_wave_tube_load_impedance(medium=medium, area_m2=float(net.exit_area_m2 or net.diaphragm_area_m2))

    def zload(_f: float) -> complex:
        return complex(zc)

    f = 1234.0
    ex = CompressionDriverExcitation(voltage_rms=2.83)
    m1 = net.solve_with_metrics(f, excitation=ex, z_external=complex(zc))
    m2 = net.solve_with_metrics(f, excitation=ex, z_external=zload)
    assert m2["electrical_impedance"] == pytest.approx(m1["electrical_impedance"], rel=1e-12, abs=1e-12)
    assert m2["volume_velocity"] == pytest.approx(m1["volume_velocity"], rel=1e-12, abs=1e-12)


def test_kirchhoff_losses_attenuate_wave_in_matched_duct():
    # Use a narrow/long duct at mid-high frequency to make attenuation obvious.
    medium = AcousticMedium(c=343.0, rho=1.225)
    diameter_m = 0.0015
    length_m = 0.15
    cfg = CompressionDriverConfig(
        name="lossy",
        driver=DriverElectroMechConfig(
            diaphragm_diameter_m=diameter_m,
            mms_kg=0.002,
            cms_m_per_n=5.0e-4,
            rms_ns_per_m=0.2,
            bl_tm=5.0,
            re_ohm=3.0,
            le_h=2.0e-4,
        ),
        rear_volume=None,
        front_duct=FrontDuctConfig(diameter_m=diameter_m, length_m=length_m),
        phase_plug=None,
        exit_cone=None,
    )

    opts = CompressionDriverNetworkOptions(duct_loss_model="Kirchhoff")
    net = CompressionDriverNetwork(cfg, medium=medium, options=opts)

    f_hz = 8000.0
    omega = 2 * np.pi * f_hz
    k_eff, zc_eff = _kirchhoff_circular_duct_characteristics(
        omega=float(omega),
        rho=float(medium.rho),
        c=float(medium.c),
        radius_m=0.5 * float(diameter_m),
        gamma=float(medium.gamma),
        prandtl=float(medium.prandtl),
        dynamic_viscosity_pa_s=float(medium.dynamic_viscosity_pa_s),
    )

    # Use matched load to isolate propagation attenuation.
    metrics = net.solve_with_metrics(f_hz, z_external=complex(zc_eff))
    u_diaphragm = metrics["volume_velocity_diaphragm"]
    u_throat = metrics["volume_velocity"]
    assert abs(u_diaphragm) > 0
    ratio = u_throat / u_diaphragm
    assert abs(ratio) < 1.0


def test_free_radiation_mode_is_applied_when_enabled():
    medium = AcousticMedium(c=343.0, rho=1.225)
    cfg = CompressionDriverConfig(
        name="free_rad",
        driver=DriverElectroMechConfig(
            diaphragm_diameter_m=0.025,
            mms_kg=0.01,
            cms_m_per_n=2.0e-4,
            rms_ns_per_m=1.0,
            bl_tm=10.0,
            re_ohm=6.0,
            le_h=1.0e-4,
        ),
        rear_volume=None,
        front_duct=None,
        phase_plug=None,
        exit_cone=None,
        front_radiation_mode="FreeRadiation",
        front_radiator_diameter_m=0.025,
    )
    net = CompressionDriverNetwork(cfg, medium=medium)
    m = net.solve_with_metrics(2000.0, z_external=np.inf + 0j)
    # If FreeRadiation is applied, z_external is ignored and the throat flow is nonzero.
    assert abs(m["volume_velocity"]) > 0
