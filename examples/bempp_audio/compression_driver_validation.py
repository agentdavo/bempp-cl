#!/usr/bin/env python3
"""
Compression Driver Validation Workflow (Panzer ICA 2019 approach).

This example demonstrates the systematic validation workflow for compression
drivers using lumped-element network analysis:

1. Vacuum Z_elec measurement -> fit electro-mechanical parameters
2. Free radiation test -> reveals internal resonances clearly
3. Plane-wave tube test -> clean frequency-independent reference load

Reference:
    Panzer J., "Modeling of a Compression Driver using Lumped Elements",
    ICA 2019, Aachen, Germany.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

_ensure_repo_on_path()

from bempp_audio.driver import (
    CompressionDriverConfig,
    DriverElectroMechConfig,
    VoiceCoilImpedanceModel,
    RearVolumeConfig,
    FrontDuctConfig,
    PhasePlugConfig,
    ExitConeConfig,
    AcousticMedium,
    CompressionDriverNetwork,
    CompressionDriverNetworkOptions,
    CompressionDriverExcitation,
    plane_wave_tube_load_impedance,
    vacuum_electrical_impedance,
)


def create_example_compression_driver() -> CompressionDriverConfig:
    """
    Create an example compression driver configuration (typical 1.4" throat).

    Based on typical parameters for a high-frequency compression driver
    with an 86mm dome diameter.
    """
    return CompressionDriverConfig(
        name="Example CD (1.4\" throat)",
        driver=DriverElectroMechConfig(
            diaphragm_diameter_m=0.086,  # 86mm dome
            mms_kg=0.0035,  # 3.5 grams
            cms_m_per_n=1.5e-4,  # Stiff suspension
            rms_ns_per_m=0.8,  # Moderate damping
            bl_tm=12.0,  # Strong motor
            re_ohm=5.5,
            le_h=0.15e-3,
            voice_coil_model=VoiceCoilImpedanceModel(
                kind="EddyLoss",
                r_eddy_max_ohm=3.0,
                f_corner_hz=800.0,
            ),
        ),
        rear_volume=RearVolumeConfig(volume_m3=25e-6),  # 25 cc rear volume
        front_duct=FrontDuctConfig(diameter_m=0.080, length_m=0.003),  # Compression chamber
        front_volume_m3=5e-6,  # 5 cc compression chamber compliance
        phase_plug=PhasePlugConfig(
            throat_area_m2=3e-4,  # ~20mm diameter at entrance
            mouth_diameter_m=0.025,  # 25mm at exit
            length_m=0.015,
            kind="Conical",
        ),
        exit_cone=ExitConeConfig(
            throat_diameter_m=0.025,
            mouth_diameter_m=0.036,  # 1.4" throat (36mm)
            length_m=0.010,
            kind="Conical",
        ),
        # Suspension chamber coupling (Panzer V1 node)
        suspension_volume_m3=8e-6,  # 8 cc suspension chamber
        suspension_diameter_m=0.100,  # 100mm outer suspension diameter
        voice_coil_slit_resistance_pa_s_per_m3=5e5,
        voice_coil_slit_mass_pa_s2_per_m3=50.0,
        # Free radiation for validation
        front_radiation_mode="FreeRadiation",
        rear_radiation_mode="FreeSpaceApprox",
    )


def compute_frequency_sweep(
    network: CompressionDriverNetwork,
    frequencies: np.ndarray,
    z_external: complex | np.ndarray = np.inf + 0j,
    excitation: CompressionDriverExcitation = CompressionDriverExcitation(),
) -> dict:
    """
    Compute full metrics over a frequency sweep.

    Returns arrays of impedance, excursion, volume velocity, and internal pressures.
    """
    z_ext_array = np.full(len(frequencies), z_external, dtype=complex) if np.isscalar(z_external) or isinstance(z_external, complex) else z_external

    z_elec = np.zeros(len(frequencies), dtype=complex)
    x_mm = np.zeros(len(frequencies), dtype=float)
    u_throat = np.zeros(len(frequencies), dtype=complex)
    p_v0 = np.zeros(len(frequencies), dtype=complex)
    p_v1 = np.zeros(len(frequencies), dtype=complex)

    for i, f in enumerate(frequencies):
        z_eff = network.get_effective_external_load(float(f), z_ext_array[i])
        metrics = network.solve_with_metrics(float(f), excitation=excitation, z_external=z_eff)
        z_elec[i] = metrics["electrical_impedance"]
        x_mm[i] = metrics["excursion_peak_mm"]
        u_throat[i] = metrics["volume_velocity"]
        p_v0[i] = metrics["pressure_v0_pa"]
        p_v1[i] = metrics["pressure_v1_pa"]

    return {
        "frequencies": frequencies,
        "z_elec": z_elec,
        "excursion_mm": x_mm,
        "volume_velocity": u_throat,
        "pressure_v0_pa": p_v0,
        "pressure_v1_pa": p_v1,
    }


def compute_vacuum_impedance(
    config: CompressionDriverConfig,
    frequencies: np.ndarray,
) -> np.ndarray:
    """Compute vacuum electrical impedance (no acoustic loading)."""
    d = config.driver
    vc = d.voice_coil_model
    return np.array([
        vacuum_electrical_impedance(
            frequency_hz=float(f),
            re_ohm=d.re_ohm,
            le_h=d.le_h,
            bl_tm=d.bl_tm,
            mms_kg=d.mms_kg,
            cms_m_per_n=d.cms_m_per_n,
            rms_ns_per_m=d.rms_ns_per_m,
            voice_coil_eddy_rmax_ohm=vc.r_eddy_max_ohm if vc else 0.0,
            voice_coil_eddy_fcorner_hz=vc.f_corner_hz if vc else 1000.0,
        )
        for f in frequencies
    ], dtype=complex)


def compute_panzer_datasets(
    config: CompressionDriverConfig,
    *,
    frequencies: np.ndarray,
    excitation: CompressionDriverExcitation = CompressionDriverExcitation(voltage_rms=2.83),
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Compute (frequencies, vacuum_z, free_radiation_metrics, plane_wave_tube_metrics).
    """
    medium = AcousticMedium()
    network = CompressionDriverNetwork(config, medium=medium)

    z_vacuum = compute_vacuum_impedance(config, frequencies)
    result_free = compute_frequency_sweep(network, frequencies, excitation=excitation)

    exit_area = network.exit_area_m2 or 1e-3
    z_pwt = plane_wave_tube_load_impedance(medium=medium, area_m2=exit_area)
    result_pwt = compute_frequency_sweep(network, frequencies, z_external=z_pwt, excitation=excitation)
    return frequencies, z_vacuum, result_free, result_pwt


def plot_validation_workflow(
    config: CompressionDriverConfig,
    save_path: str | None = None,
) -> object:
    """
    Generate the Panzer validation workflow plots.

    Produces a 2x3 grid of plots:
    1. Vacuum Z_elec (magnitude + phase)
    2. Free radiation Z_elec
    3. Plane-wave tube Z_elec
    4. Free radiation vs plane-wave tube comparison
    5. Internal node pressures (V0, V1)
    6. Diaphragm excursion comparison
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    medium = AcousticMedium()
    network = CompressionDriverNetwork(config, medium=medium)
    excitation = CompressionDriverExcitation(voltage_rms=2.83)

    # Frequency range
    frequencies = np.logspace(np.log10(100), np.log10(20000), 200)

    frequencies, z_vacuum, result_free, result_pwt = compute_panzer_datasets(
        config, frequencies=frequencies, excitation=excitation
    )

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Compression Driver Validation: {config.name}", fontsize=14, fontweight="bold")

    # Plot 1: Vacuum Z_elec
    ax = axes[0, 0]
    ax.semilogx(frequencies, np.abs(z_vacuum), "b-", linewidth=2, label="|Z|")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Impedance (Ω)", color="b")
    ax.set_title("Vacuum Electrical Impedance")
    ax.set_xlim([100, 20000])
    ax.grid(True, alpha=0.3)
    ax_phase = ax.twinx()
    ax_phase.semilogx(frequencies, np.angle(z_vacuum, deg=True), "r--", linewidth=1.5, label="Phase")
    ax_phase.set_ylabel("Phase (°)", color="r")
    ax_phase.set_ylim([-90, 90])

    # Plot 2: Free radiation Z_elec
    ax = axes[0, 1]
    ax.semilogx(frequencies, np.abs(result_free["z_elec"]), "b-", linewidth=2, label="|Z|")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Impedance (Ω)", color="b")
    ax.set_title("Free Radiation Z_elec")
    ax.set_xlim([100, 20000])
    ax.grid(True, alpha=0.3)
    ax_phase = ax.twinx()
    ax_phase.semilogx(frequencies, np.angle(result_free["z_elec"], deg=True), "r--", linewidth=1.5)
    ax_phase.set_ylabel("Phase (°)", color="r")
    ax_phase.set_ylim([-90, 90])

    # Plot 3: Plane-wave tube Z_elec
    ax = axes[0, 2]
    ax.semilogx(frequencies, np.abs(result_pwt["z_elec"]), "b-", linewidth=2, label="|Z|")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Impedance (Ω)", color="b")
    ax.set_title("Plane-Wave Tube Z_elec")
    ax.set_xlim([100, 20000])
    ax.grid(True, alpha=0.3)
    ax_phase = ax.twinx()
    ax_phase.semilogx(frequencies, np.angle(result_pwt["z_elec"], deg=True), "r--", linewidth=1.5)
    ax_phase.set_ylabel("Phase (°)", color="r")
    ax_phase.set_ylim([-90, 90])

    # Plot 4: Comparison overlay
    ax = axes[1, 0]
    ax.semilogx(frequencies, np.abs(z_vacuum), "k-", linewidth=2, label="Vacuum")
    ax.semilogx(frequencies, np.abs(result_free["z_elec"]), "b-", linewidth=1.5, label="Free radiation")
    ax.semilogx(frequencies, np.abs(result_pwt["z_elec"]), "r-", linewidth=1.5, label="Plane-wave tube")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Impedance (Ω)")
    ax.set_title("Z_elec Comparison")
    ax.set_xlim([100, 20000])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 5: Internal node pressures
    ax = axes[1, 1]
    ax.semilogx(frequencies, 20 * np.log10(np.abs(result_free["pressure_v0_pa"]) / 20e-6 + 1e-30), "b-", linewidth=2, label="V0 (front)")
    ax.semilogx(frequencies, 20 * np.log10(np.abs(result_free["pressure_v1_pa"]) / 20e-6 + 1e-30), "r-", linewidth=1.5, label="V1 (suspension)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPL (dB re 20µPa)")
    ax.set_title("Internal Node Pressures (Free Radiation)")
    ax.set_xlim([100, 20000])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 6: Excursion comparison
    ax = axes[1, 2]
    ax.semilogx(frequencies, result_free["excursion_mm"], "b-", linewidth=2, label="Free radiation")
    ax.semilogx(frequencies, result_pwt["excursion_mm"], "r-", linewidth=1.5, label="Plane-wave tube")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Peak Excursion (mm)")
    ax.set_title("Diaphragm Excursion @ 2.83V")
    ax.set_xlim([100, 20000])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def main():
    print("Compression Driver Validation Workflow (Panzer approach)")
    print("=" * 60)

    # Create example driver
    config = create_example_compression_driver()
    print(f"Driver: {config.name}")
    print(f"  Diaphragm: {config.driver.diaphragm_diameter_m * 1000:.1f} mm")
    print(f"  Mms: {config.driver.mms_kg * 1000:.2f} g")
    print(f"  Bl: {config.driver.bl_tm:.1f} T·m")
    print(f"  Re: {config.driver.re_ohm:.1f} Ω")
    print()

    # Compute fs in vacuum
    fs = 1 / (2 * np.pi * np.sqrt(config.driver.mms_kg * config.driver.cms_m_per_n))
    print(f"  Vacuum fs: {fs:.1f} Hz")
    print()

    if MATPLOTLIB_AVAILABLE:
        fig = plot_validation_workflow(config, save_path="compression_driver_validation.png")
        plt.show()
    else:
        print("matplotlib not available - skipping plots")

        # Print some numeric results instead
        medium = AcousticMedium()
        network = CompressionDriverNetwork(config, medium=medium)

        print("\nSample impedance values:")
        for f in [500, 1000, 2000, 5000, 10000]:
            z_eff = network.get_effective_external_load(float(f))
            metrics = network.solve_with_metrics(float(f), z_external=z_eff)
            z = metrics["electrical_impedance"]
            print(f"  {f:5d} Hz: |Z| = {np.abs(z):6.2f} Ω, phase = {np.angle(z, deg=True):+6.1f}°")

    # Export interactive dashboard (always best-effort)
    try:
        out_dir = Path(os.environ.get("BEMPPAUDIO_OUT_DIR", "logs")).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_html = out_dir / f"{Path(__file__).stem}_dashboard.html"

        frequencies = np.logspace(np.log10(100), np.log10(20000), 200)
        frequencies, z_vacuum, result_free, result_pwt = compute_panzer_datasets(config, frequencies=frequencies)

        from bempp_audio.viz.plotly_driver_network import save_panzer_validation_dashboard_html, NetworkValidationData

        def _pack(d: dict) -> NetworkValidationData:
            return NetworkValidationData(
                frequencies_hz=np.asarray(d["frequencies"], dtype=float),
                z_elec=np.asarray(d["z_elec"], dtype=complex),
                excursion_mm=np.asarray(d["excursion_mm"], dtype=float),
                volume_velocity=np.asarray(d["volume_velocity"], dtype=complex),
                pressure_v0_pa=np.asarray(d["pressure_v0_pa"], dtype=complex),
                pressure_v1_pa=np.asarray(d["pressure_v1_pa"], dtype=complex),
            )

        save_panzer_validation_dashboard_html(
            filename=str(out_html),
            title=f"Compression Driver Validation — {config.name}",
            vacuum=(np.asarray(frequencies, dtype=float), np.asarray(z_vacuum, dtype=complex)),
            free_radiation=_pack(result_free),
            plane_wave_tube=_pack(result_pwt),
        )
        print(f"Dashboard exported: {out_html}")
    except Exception as e:
        print(f"Dashboard export skipped: {e}")


if __name__ == "__main__":
    main()
