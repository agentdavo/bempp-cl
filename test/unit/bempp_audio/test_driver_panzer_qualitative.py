from __future__ import annotations

import numpy as np

from bempp_audio.driver import (
    AcousticMedium,
    CompressionDriverConfig,
    CompressionDriverExcitation,
    CompressionDriverNetwork,
    DriverElectroMechConfig,
    FrontDuctConfig,
    RearVolumeConfig,
    plane_wave_tube_load_impedance,
)


def _peak_freq(freqs: np.ndarray, z: np.ndarray) -> float:
    return float(freqs[int(np.argmax(np.real(z)))])


def test_rear_volume_increase_lowers_low_frequency_impedance_peak():
    """
    Panzer-style qualitative check:
    increasing rear volume (more compliance) lowers the low-frequency resonance.
    """
    medium = AcousticMedium(c=343.0, rho=1.225)
    diameter_m = 0.025
    length_m = 0.02

    base_driver = DriverElectroMechConfig(
        diaphragm_diameter_m=diameter_m,
        mms_kg=0.01,
        cms_m_per_n=5.0e-5,
        rms_ns_per_m=1.0,
        bl_tm=10.0,
        re_ohm=6.0,
        le_h=0.0001,
    )

    cfg_small = CompressionDriverConfig(
        name="rear_small",
        driver=base_driver,
        rear_volume=RearVolumeConfig(volume_m3=50e-6),
        front_duct=FrontDuctConfig(diameter_m=diameter_m, length_m=length_m),
        phase_plug=None,
        exit_cone=None,
    )
    cfg_large = CompressionDriverConfig(
        name="rear_large",
        driver=base_driver,
        rear_volume=RearVolumeConfig(volume_m3=400e-6),
        front_duct=FrontDuctConfig(diameter_m=diameter_m, length_m=length_m),
        phase_plug=None,
        exit_cone=None,
    )

    net_small = CompressionDriverNetwork(cfg_small, medium=medium)
    net_large = CompressionDriverNetwork(cfg_large, medium=medium)
    z_load = plane_wave_tube_load_impedance(medium=medium, area_m2=float(net_small.exit_area_m2 or net_small.diaphragm_area_m2))
    ex = CompressionDriverExcitation(voltage_rms=2.83)

    freqs = np.linspace(100.0, 1200.0, 600)
    z_small = np.array([net_small.solve_with_metrics(float(f), excitation=ex, z_external=complex(z_load))["electrical_impedance"] for f in freqs])
    z_large = np.array([net_large.solve_with_metrics(float(f), excitation=ex, z_external=complex(z_load))["electrical_impedance"] for f in freqs])

    f_small = _peak_freq(freqs, z_small)
    f_large = _peak_freq(freqs, z_large)
    assert f_large < f_small


def test_increasing_slit_resistance_increases_resonance_peak_height():
    """
    Panzer-style qualitative check:
    increasing V0↔V1 slit resistance reduces shunt coupling and increases the motional peak.
    """
    medium = AcousticMedium(c=343.0, rho=1.225)
    diameter_m = 0.025
    length_m = 0.02

    base_driver = DriverElectroMechConfig(
        diaphragm_diameter_m=diameter_m,
        mms_kg=0.01,
        cms_m_per_n=5.0e-5,
        rms_ns_per_m=1.0,
        bl_tm=10.0,
        re_ohm=6.0,
        le_h=0.0001,
    )

    common = dict(
        driver=base_driver,
        rear_volume=RearVolumeConfig(volume_m3=200e-6),
        front_duct=FrontDuctConfig(diameter_m=diameter_m, length_m=length_m),
        suspension_volume_m3=50e-6,
        suspension_diameter_m=0.03,
        voice_coil_slit_mass_pa_s2_per_m3=0.0,
        phase_plug=None,
        exit_cone=None,
    )

    cfg_lo = CompressionDriverConfig(
        name="slit_low_r",
        voice_coil_slit_resistance_pa_s_per_m3=1e5,
        **common,
    )
    cfg_hi = CompressionDriverConfig(
        name="slit_high_r",
        voice_coil_slit_resistance_pa_s_per_m3=5e6,
        **common,
    )

    net_lo = CompressionDriverNetwork(cfg_lo, medium=medium)
    net_hi = CompressionDriverNetwork(cfg_hi, medium=medium)
    z_load = plane_wave_tube_load_impedance(medium=medium, area_m2=float(net_lo.exit_area_m2 or net_lo.diaphragm_area_m2))
    ex = CompressionDriverExcitation(voltage_rms=2.83)

    freqs = np.linspace(100.0, 1200.0, 600)
    z_lo = np.array([net_lo.solve_with_metrics(float(f), excitation=ex, z_external=complex(z_load))["electrical_impedance"] for f in freqs])
    z_hi = np.array([net_hi.solve_with_metrics(float(f), excitation=ex, z_external=complex(z_load))["electrical_impedance"] for f in freqs])

    peak_lo = float(np.max(np.real(z_lo)))
    peak_hi = float(np.max(np.real(z_hi)))
    assert peak_hi > peak_lo
