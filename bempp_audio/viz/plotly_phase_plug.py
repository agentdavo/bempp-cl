"""
Plotly HTML dashboards for interior phase plug acoustic metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PhasePlugMetricsSeries:
    frequency_hz: np.ndarray
    transmission_efficiency: np.ndarray
    pressure_uniformity: np.ndarray
    phase_spread_deg: np.ndarray
    throat_impedance_mag: np.ndarray
    throat_impedance_phase_deg: np.ndarray
    impedance_transform_mag: np.ndarray


def _series_from_metrics(metrics: Sequence[object]) -> PhasePlugMetricsSeries:
    def _get(m, k: str):
        if hasattr(m, k):
            return getattr(m, k)
        return m[k]

    f = np.array([float(_get(m, "frequency_hz")) for m in metrics], dtype=float)
    eff = np.array([float(_get(m, "transmission_efficiency")) for m in metrics], dtype=float)
    uni = np.array([float(_get(m, "pressure_uniformity")) for m in metrics], dtype=float)
    phs = np.array([float(_get(m, "phase_spread_deg")) for m in metrics], dtype=float)
    zt = np.array([complex(_get(m, "throat_impedance")) for m in metrics], dtype=complex)
    zd = np.array([complex(_get(m, "impedance_transformation")) for m in metrics], dtype=complex)

    return PhasePlugMetricsSeries(
        frequency_hz=f,
        transmission_efficiency=eff,
        pressure_uniformity=uni,
        phase_spread_deg=phs,
        throat_impedance_mag=np.abs(zt),
        throat_impedance_phase_deg=np.degrees(np.angle(zt)),
        impedance_transform_mag=np.abs(zd),
    )


def phase_plug_dashboard_figure(
    metrics: Sequence[object],
    *,
    title: str = "Phase Plug Metrics Dashboard",
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        raise ImportError("plotly is required for HTML dashboards (`pip install plotly`).") from e

    s = _series_from_metrics(metrics)

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "secondary_y": True}, {"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "Transmission Efficiency (power)",
            "Throat Pressure Uniformity",
            "Throat Phase Spread",
            "Throat Impedance |Z| + phase",
            "Impedance Transform |Z_dome/Z_throat|",
            "Summary",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Scatter(x=s.frequency_hz, y=s.transmission_efficiency, name="η", mode="lines"), row=1, col=1)
    fig.update_yaxes(title_text="η", row=1, col=1, range=[0, 1.05])

    fig.add_trace(go.Scatter(x=s.frequency_hz, y=s.pressure_uniformity, name="Uniformity", mode="lines"), row=1, col=2)
    fig.update_yaxes(title_text="0..1", row=1, col=2, range=[0, 1.05])

    fig.add_trace(go.Scatter(x=s.frequency_hz, y=s.phase_spread_deg, name="Phase spread (deg)", mode="lines"), row=1, col=3)
    fig.update_yaxes(title_text="deg", row=1, col=3)

    fig.add_trace(go.Scatter(x=s.frequency_hz, y=s.throat_impedance_mag, name="|Z_throat|", mode="lines"), row=2, col=1, secondary_y=False)
    fig.add_trace(
        go.Scatter(x=s.frequency_hz, y=s.throat_impedance_phase_deg, name="∠Z_throat", mode="lines", line=dict(dash="dash")),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Pa·s/m³", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="deg", row=2, col=1, secondary_y=True, range=[-180, 180])

    fig.add_trace(go.Scatter(x=s.frequency_hz, y=s.impedance_transform_mag, name="|Z_dome/Z_throat|", mode="lines"), row=2, col=2)
    fig.update_yaxes(title_text="ratio", row=2, col=2)

    # Summary text box
    eff_min = float(np.nanmin(s.transmission_efficiency)) if s.transmission_efficiency.size else float("nan")
    eff_max = float(np.nanmax(s.transmission_efficiency)) if s.transmission_efficiency.size else float("nan")
    uni_min = float(np.nanmin(s.pressure_uniformity)) if s.pressure_uniformity.size else float("nan")
    phs_max = float(np.nanmax(s.phase_spread_deg)) if s.phase_spread_deg.size else float("nan")
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="text",
            text=[
                "Summary<br>"
                f"η: {eff_min:.2f}–{eff_max:.2f}<br>"
                f"Uniformity min: {uni_min:.2f}<br>"
                f"Phase spread max: {phs_max:.1f}°"
            ],
            showlegend=False,
        ),
        row=2,
        col=3,
    )
    fig.update_xaxes(showticklabels=False, row=2, col=3)
    fig.update_yaxes(showticklabels=False, row=2, col=3)

    fig.update_xaxes(type="log")
    fig.update_layout(
        title=str(title),
        height=750,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left", x=0.0),
        margin=dict(l=40, r=20, t=60, b=80),
    )
    return fig


def save_phase_plug_dashboard_html(
    metrics: Sequence[object],
    filename: str,
    *,
    title: str = "Phase Plug Metrics Dashboard",
    include_plotlyjs: str | bool = True,
) -> str:
    fig = phase_plug_dashboard_figure(metrics, title=title)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=include_plotlyjs)
    return str(path)
