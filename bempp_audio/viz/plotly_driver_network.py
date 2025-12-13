"""
Plotly HTML dashboards for lumped-element compression driver network validation.

These dashboards are intended for the Panzer-style validation workflow where we
have complex Z_elec curves and internal node pressures from the LEM, but not a
full exterior BEM FrequencyResponse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class NetworkValidationData:
    frequencies_hz: np.ndarray
    z_elec: np.ndarray
    excursion_mm: np.ndarray
    volume_velocity: np.ndarray
    pressure_v0_pa: np.ndarray
    pressure_v1_pa: np.ndarray


def _as1d(x, *, dtype=float) -> np.ndarray:
    a = np.asarray(x, dtype=dtype).reshape(-1)
    return a


def _z_mag_phase(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=complex)
    mag = np.abs(z)
    phase = np.degrees(np.angle(z))
    return mag, phase


def panzer_validation_dashboard_figure(
    *,
    title: str,
    vacuum: Optional[tuple[np.ndarray, np.ndarray]] = None,
    free_radiation: Optional[NetworkValidationData] = None,
    plane_wave_tube: Optional[NetworkValidationData] = None,
):
    """
    Build an interactive Plotly dashboard for Panzer-style validation.

    Parameters
    ----------
    vacuum:
        Optional (freqs_hz, z_vacuum) for the vacuum impedance curve.
    free_radiation:
        Optional metrics dict from `compute_frequency_sweep` in the example.
    plane_wave_tube:
        Optional metrics dict from `compute_frequency_sweep` in the example.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        raise ImportError("plotly is required for HTML dashboards (`pip install plotly`).") from e

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "xy", "secondary_y": True}, {"type": "xy", "secondary_y": True}, {"type": "xy", "secondary_y": True}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "Vacuum Z_elec",
            "Free Radiation Z_elec",
            "Plane-Wave Tube Z_elec",
            "Z_elec Comparison (|Z|)",
            "Internal Node Pressures (Free Radiation)",
            "Excursion (2.83 V)",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    def _add_z_panel(row: int, col: int, freqs: np.ndarray, z: np.ndarray, name_prefix: str):
        mag, phase = _z_mag_phase(z)
        fig.add_trace(go.Scatter(x=freqs, y=mag, name=f"{name_prefix} |Z|", mode="lines"), row=row, col=col, secondary_y=False)
        fig.add_trace(
            go.Scatter(x=freqs, y=phase, name=f"{name_prefix} phase", mode="lines", line=dict(dash="dash")),
            row=row,
            col=col,
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Ω", row=row, col=col, secondary_y=False)
        fig.update_yaxes(title_text="°", row=row, col=col, secondary_y=True, range=[-90, 90])

    # Vacuum
    if vacuum is not None:
        f_v, z_v = vacuum
        _add_z_panel(1, 1, _as1d(f_v, dtype=float), np.asarray(z_v, dtype=complex), "Vacuum")

    # Free radiation
    if free_radiation is not None:
        f = _as1d(free_radiation.frequencies_hz, dtype=float)
        _add_z_panel(1, 2, f, np.asarray(free_radiation.z_elec, dtype=complex), "Free")

    # Plane-wave tube
    if plane_wave_tube is not None:
        f = _as1d(plane_wave_tube.frequencies_hz, dtype=float)
        _add_z_panel(1, 3, f, np.asarray(plane_wave_tube.z_elec, dtype=complex), "PWT")

    # Comparison |Z|
    if vacuum is not None:
        f_v, z_v = vacuum
        fig.add_trace(go.Scatter(x=_as1d(f_v), y=np.abs(np.asarray(z_v, dtype=complex)), name="Vacuum |Z|", mode="lines"), row=2, col=1)
    if free_radiation is not None:
        fig.add_trace(
            go.Scatter(x=_as1d(free_radiation.frequencies_hz), y=np.abs(np.asarray(free_radiation.z_elec, dtype=complex)), name="Free |Z|", mode="lines"),
            row=2,
            col=1,
        )
    if plane_wave_tube is not None:
        fig.add_trace(
            go.Scatter(x=_as1d(plane_wave_tube.frequencies_hz), y=np.abs(np.asarray(plane_wave_tube.z_elec, dtype=complex)), name="PWT |Z|", mode="lines"),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text="Ω", row=2, col=1)

    # Internal node pressures (free radiation)
    if free_radiation is not None:
        f = _as1d(free_radiation.frequencies_hz, dtype=float)
        p0 = np.asarray(free_radiation.pressure_v0_pa, dtype=complex)
        p1 = np.asarray(free_radiation.pressure_v1_pa, dtype=complex)
        ref = 20e-6
        spl0 = 20.0 * np.log10(np.maximum(np.abs(p0), ref * 1e-10) / ref)
        spl1 = 20.0 * np.log10(np.maximum(np.abs(p1), ref * 1e-10) / ref)
        fig.add_trace(go.Scatter(x=f, y=spl0, name="V0 SPL (dB)", mode="lines"), row=2, col=2)
        fig.add_trace(go.Scatter(x=f, y=spl1, name="V1 SPL (dB)", mode="lines", line=dict(dash="dash")), row=2, col=2)
        fig.update_yaxes(title_text="dB re 20µPa", row=2, col=2)

    # Excursion comparison
    if free_radiation is not None:
        fig.add_trace(
            go.Scatter(x=_as1d(free_radiation.frequencies_hz), y=_as1d(free_radiation.excursion_mm), name="Free excursion (mm)", mode="lines"),
            row=2,
            col=3,
        )
    if plane_wave_tube is not None:
        fig.add_trace(
            go.Scatter(
                x=_as1d(plane_wave_tube.frequencies_hz),
                y=_as1d(plane_wave_tube.excursion_mm),
                name="PWT excursion (mm)",
                mode="lines",
                line=dict(dash="dash"),
            ),
            row=2,
            col=3,
        )
    fig.update_yaxes(title_text="mm (peak)", row=2, col=3)

    fig.update_xaxes(type="log")
    fig.update_layout(
        title=str(title),
        height=750,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left", x=0.0),
        margin=dict(l=40, r=20, t=60, b=80),
    )
    return fig


def save_panzer_validation_dashboard_html(
    *,
    filename: str,
    title: str,
    vacuum: Optional[tuple[np.ndarray, np.ndarray]] = None,
    free_radiation: Optional[NetworkValidationData] = None,
    plane_wave_tube: Optional[NetworkValidationData] = None,
    include_plotlyjs: str | bool = True,
) -> str:
    fig = panzer_validation_dashboard_figure(
        title=title,
        vacuum=vacuum,
        free_radiation=free_radiation,
        plane_wave_tube=plane_wave_tube,
    )
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=include_plotlyjs)
    return str(path)
