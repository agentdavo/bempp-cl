from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.driver.compression_config import (
    CompressionDriverConfig,
    DriverElectroMechConfig,
    RearVolumeConfig,
    FrontDuctConfig,
)
from bempp_audio.driver.network import (
    AcousticMedium,
    CompressionDriverNetwork,
    CompressionDriverExcitation,
)


def test_network_returns_throat_volume_velocity_for_matched_tube_load():
    """
    For a uniform tube terminated in its characteristic impedance, the relation is:
        U_out = U_in * exp(-j*k*L)
    where U_in is diaphragm-side volume velocity and U_out is the load-side (throat) volume velocity.
    """
    medium = AcousticMedium(c=343.0, rho=1.225)
    diameter_m = 0.025
    length_m = 0.02
    area = np.pi * (0.5 * diameter_m) ** 2
    zc = medium.rho * medium.c / area

    cfg = CompressionDriverConfig(
        name="test",
        driver=DriverElectroMechConfig(
            diaphragm_diameter_m=diameter_m,
            mms_kg=0.01,
            cms_m_per_n=2.0e-4,
            rms_ns_per_m=1.0,
            bl_tm=10.0,
            re_ohm=6.0,
            le_h=0.0001,
        ),
        rear_volume=RearVolumeConfig(volume_m3=200e-6),
        front_duct=FrontDuctConfig(diameter_m=diameter_m, length_m=length_m),
        phase_plug=None,
        exit_cone=None,
    )

    net = CompressionDriverNetwork(cfg, medium=medium)
    f_hz = 1000.0
    excitation = CompressionDriverExcitation(voltage_rms=2.83)
    metrics = net.solve_with_metrics(f_hz, excitation=excitation, z_external=complex(zc))

    u_diaphragm = metrics["volume_velocity_diaphragm"]
    u_throat = metrics["volume_velocity"]

    assert abs(u_diaphragm) > 0

    k = 2 * np.pi * f_hz / medium.c
    expected_ratio = np.exp(-1j * k * length_m)
    assert (u_throat / u_diaphragm) == pytest.approx(expected_ratio, rel=1e-9, abs=1e-9)

    assert net.solve_volume_velocity(f_hz, excitation=excitation, z_external=complex(zc)) == pytest.approx(
        u_throat, rel=1e-12, abs=1e-12
    )


def test_front_input_impedance_for_open_tube_load_matches_cotangent_formula():
    """
    For a uniform lossless tube terminated by an open load (U2 = 0),
    the input impedance is:
        Z_in = -j * Zc * cot(kL)
    where Zc = rho*c/S.
    """
    medium = AcousticMedium(c=343.0, rho=1.225)
    diameter_m = 0.025
    length_m = 0.02
    area = np.pi * (0.5 * diameter_m) ** 2
    zc = medium.rho * medium.c / area

    cfg = CompressionDriverConfig(
        name="open_load",
        driver=DriverElectroMechConfig(
            diaphragm_diameter_m=diameter_m,
            mms_kg=0.01,
            cms_m_per_n=2.0e-4,
            rms_ns_per_m=1.0,
            bl_tm=10.0,
            re_ohm=6.0,
            le_h=0.0001,
        ),
        rear_volume=None,
        front_duct=FrontDuctConfig(diameter_m=diameter_m, length_m=length_m),
        phase_plug=None,
        exit_cone=None,
    )

    net = CompressionDriverNetwork(cfg, medium=medium)
    f_hz = 1234.0
    k = 2 * np.pi * f_hz / medium.c
    expected = -1j * complex(zc) / np.tan(k * length_m)

    z_in = net.front_input_impedance(f_hz, z_external=np.inf + 0j)
    assert z_in == pytest.approx(expected, rel=1e-10, abs=1e-10)

    metrics = net.solve_with_metrics(f_hz, excitation=CompressionDriverExcitation(voltage_rms=2.83), z_external=np.inf + 0j)
    assert metrics["volume_velocity"] == 0.0 + 0.0j
    assert np.isfinite(metrics["electrical_impedance"])


def test_network_accounts_for_v1_shunt_in_throat_volume_velocity():
    """
    If a shunt path exists from V0 to V1, the throat flow is reduced compared to Sd*v.

    Use a uniform tube terminated in its characteristic impedance (so Z_front = Zc and
    U_out/U_front_in = exp(-j*k*L)). Compare against nodal solve result.
    """
    medium = AcousticMedium(c=343.0, rho=1.225)
    diameter_m = 0.025
    length_m = 0.02
    area = np.pi * (0.5 * diameter_m) ** 2
    zc = medium.rho * medium.c / area

    cfg = CompressionDriverConfig(
        name="test_v1",
        driver=DriverElectroMechConfig(
            diaphragm_diameter_m=diameter_m,
            mms_kg=0.01,
            cms_m_per_n=2.0e-4,
            rms_ns_per_m=1.0,
            bl_tm=10.0,
            re_ohm=6.0,
            le_h=0.0001,
        ),
        rear_volume=RearVolumeConfig(volume_m3=200e-6),
        front_duct=FrontDuctConfig(diameter_m=diameter_m, length_m=length_m),
        suspension_volume_m3=50e-6,
        suspension_diameter_m=0.03,  # S1 comparable to Sd
        voice_coil_slit_resistance_pa_s_per_m3=1e6,
        voice_coil_slit_mass_pa_s2_per_m3=0.0,
        phase_plug=None,
        exit_cone=None,
    )

    net = CompressionDriverNetwork(cfg, medium=medium)
    f_hz = 1000.0
    excitation = CompressionDriverExcitation(voltage_rms=2.83)
    metrics = net.solve_with_metrics(f_hz, excitation=excitation, z_external=complex(zc))

    u_diaphragm = metrics["volume_velocity_diaphragm"]
    u_front = metrics["volume_velocity_front"]
    u_throat = metrics["volume_velocity"]

    # Shunt should divert some flow away from the front chain.
    assert abs(u_front) < abs(u_diaphragm)

    k = 2 * np.pi * f_hz / medium.c
    expected_ratio = np.exp(-1j * k * length_m)
    assert (u_throat / u_front) == pytest.approx(expected_ratio, rel=1e-9, abs=1e-9)


def test_v0_compliance_shifts_low_frequency_impedance_peak():
    """
    Adding an explicit front compression chamber compliance (V0) changes the acoustic loading
    and should shift the low-frequency electrical impedance peak.
    """
    medium = AcousticMedium(c=343.0, rho=1.225)
    diameter_m = 0.025
    length_m = 0.02
    area = np.pi * (0.5 * diameter_m) ** 2
    zc = medium.rho * medium.c / area

    base_driver = DriverElectroMechConfig(
        diaphragm_diameter_m=diameter_m,
        mms_kg=0.01,
        cms_m_per_n=5.0e-5,
        rms_ns_per_m=1.0,
        bl_tm=10.0,
        re_ohm=6.0,
        le_h=0.0001,
    )

    cfg_no_v0 = CompressionDriverConfig(
        name="no_v0",
        driver=base_driver,
        rear_volume=None,
        front_duct=FrontDuctConfig(diameter_m=diameter_m, length_m=length_m),
        front_volume_m3=None,
    )
    cfg_with_v0 = CompressionDriverConfig(
        name="with_v0",
        driver=base_driver,
        rear_volume=None,
        front_duct=FrontDuctConfig(diameter_m=diameter_m, length_m=length_m),
        front_volume_m3=20e-6,
    )

    net0 = CompressionDriverNetwork(cfg_no_v0, medium=medium)
    net1 = CompressionDriverNetwork(cfg_with_v0, medium=medium)
    ex = CompressionDriverExcitation(voltage_rms=2.83)

    freqs = np.linspace(100.0, 800.0, 250)
    z0 = np.array([net0.solve_with_metrics(float(f), excitation=ex, z_external=complex(zc))["electrical_impedance"] for f in freqs])
    z1 = np.array([net1.solve_with_metrics(float(f), excitation=ex, z_external=complex(zc))["electrical_impedance"] for f in freqs])

    # Use the motional peak in the real part (more robust than |Z| when reactance dominates at endpoints).
    f0 = float(freqs[int(np.argmax(np.real(z0)))])
    f1 = float(freqs[int(np.argmax(np.real(z1)))])

    assert f1 > f0
