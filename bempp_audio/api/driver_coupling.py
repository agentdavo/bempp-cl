"""
Coupling between BEM radiation and lumped electro-acoustic driver networks.

This module exists to keep the fluent API orchestration (`Loudspeaker.solve`)
from becoming monolithic.
"""

from __future__ import annotations

from typing import Optional, Callable
import numpy as np

from bempp_audio._optional import require_bempp
from bempp_audio.results import FrequencyResponse
from bempp_audio.velocity import VelocityProfile
from bempp_audio.progress import get_logger
from bempp_audio.baffles import InfiniteBaffle


def solve_with_compression_driver_network(
    *,
    solver: "RadiationSolver",
    frequencies: np.ndarray,
    driver_network: "CompressionDriverNetwork",
    excitation: "CompressionDriverExcitation",
    throat_domain: int,
    baffle_mode: Optional[object],
    progress_callback: Optional[Callable[[float, int, int], None]],
) -> FrequencyResponse:
    """
    Coupled sequential sweep:
    - Solve BEM with unit throat velocity (peak=1 m/s)
    - Compute throat acoustic impedance from BEM result
    - Use it as network termination to obtain throat volume velocity
    - Scale BEM pressure coefficients to the resulting throat velocity
    """
    logger = get_logger()
    logger.info("Drive model: compression-driver lumped network (coupled via throat impedance)")

    response = FrequencyResponse()
    response.set_driver_diaphragm_area(driver_network.diaphragm_area_m2)

    unit_velocity = VelocityProfile.by_domain(
        {int(throat_domain): VelocityProfile.piston(amplitude=1.0)},
        default=VelocityProfile.zero(),
    )

    for i, freq in enumerate(frequencies):
        unit_result = solver.solve(float(freq), unit_velocity)

        imp = unit_result.radiation_impedance()
        z_ext = imp.acoustic()
        if isinstance(baffle_mode, InfiniteBaffle):
            z_ext *= float(baffle_mode.pressure_scale)

        # Solve with full metrics (impedance, excursion, etc.)
        metrics = driver_network.solve_with_metrics(
            float(freq),
            excitation=excitation,
            z_external=z_ext,
        )
        u_rms = metrics['volume_velocity']

        # Store driver metrics in response
        z_in = metrics['electrical_impedance']
        response.add_driver_metrics(
            frequency=float(freq),
            impedance_re=float(np.real(z_in)),
            impedance_im=float(np.imag(z_in)),
            excursion_mm=float(metrics['excursion_peak_mm']),
        )

        area_active = imp.active_area()
        if area_active <= 0:
            area_active = driver_network.exit_area_m2 or driver_network.diaphragm_area_m2

        v_peak = (abs(u_rms) / area_active) * np.sqrt(2.0)
        scale = v_peak / 1.0

        # Apply scaling to boundary pressure coefficients
        # (Avoid importing bempp at module import time.)
        bempp = require_bempp()

        scaled_coeffs = unit_result.surface_pressure.coefficients * scale
        if isinstance(baffle_mode, InfiniteBaffle):
            scaled_coeffs *= float(baffle_mode.pressure_scale)
            unit_result.baffle = baffle_mode
            unit_result.baffle_plane_z = float(baffle_mode.plane_z)

        unit_result.surface_pressure = bempp.GridFunction(
            unit_result.surface_pressure.space,
            coefficients=scaled_coeffs,
        )
        unit_result.velocity = VelocityProfile.by_domain(
            {int(throat_domain): VelocityProfile.piston(amplitude=v_peak)},
            default=VelocityProfile.zero(),
        )

        response.add(unit_result)

        if progress_callback:
            progress_callback(float(freq), i + 1, len(frequencies))

    logger.success("Solve complete")
    return response
