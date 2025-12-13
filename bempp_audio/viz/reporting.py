"""
Reporting helpers for acoustic simulation results.

This module centralizes plot/export workflows so that `bempp_audio.results`
containers remain focused on data + computation.
"""

from __future__ import annotations

from typing import Optional, List


def save_summary(
    response: "FrequencyResponse",
    filename: str = "acoustic_summary.png",
    title: Optional[str] = None,
    distance: float = 1.0,
    angles: Optional[List[float]] = None,
    max_angle: float = 90.0,
    normalize_angle: float = 0.0,
    include_driver: Optional[bool] = None,
    include_waveguide: Optional[bool] = None,
) -> None:
    from bempp_audio.viz import ReportBuilder

    if angles is None:
        angles = [0, 15, 30, 45, 60, 90] if max_angle >= 90 else [0, 15, 30, 45, 60]

    builder = (
        ReportBuilder(response)
        .title(title or "Acoustic Radiation Summary")
        .distance(float(distance))
        .polar_options(max_angle_deg=float(max_angle), normalize_angle_deg=float(normalize_angle))
        .preset_waveguide_summary(include_driver=include_driver, include_waveguide=include_waveguide, angles_deg=angles)
    )
    fig = builder.render()
    fig.savefig(str(filename), bbox_inches="tight")

    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


def save_cea2034(
    response: "FrequencyResponse",
    filename_prefix: str = "cea2034",
    measurement_distance: float = 1.0,
    normalize_to_angle: float = 0.0,
    reference_frequency: float = 1000.0,
    charts: str = "all",
    title: Optional[str] = None,
) -> None:
    from bempp_audio.viz import CEA2034Chart

    cea = CEA2034Chart(
        response,
        measurement_distance=measurement_distance,
        reference_frequency=reference_frequency,
        normalize_to_angle=normalize_to_angle,
    )

    if charts in ("spinorama", "all"):
        cea.plot_spinorama(
            save=f"{filename_prefix}_spinorama.png",
            title=title,
            show=False,
        )

    if charts in ("standard", "all"):
        cea.plot_standard(
            save=f"{filename_prefix}_standard.png",
            title=title,
            show=False,
        )

    if charts in ("full", "all"):
        cea.plot_full_report(
            save=f"{filename_prefix}_full.png",
            title=title,
            show=False,
        )


def save_all_plots(
    response: "FrequencyResponse",
    prefix: str = "acoustic",
    title: Optional[str] = None,
    distance: float = 1.0,
    max_angle: float = 90.0,
    normalize_angle: float = 0.0,
) -> None:
    save_summary(
        response,
        filename=f"{prefix}_summary.png",
        title=title,
        distance=distance,
        max_angle=max_angle,
        normalize_angle=normalize_angle,
    )

    # CEA-2034 charts require multiple frequencies for meaningful analysis
    if len(response.frequencies) >= 2:
        save_cea2034(
            response,
            filename_prefix=prefix,
            measurement_distance=distance,
            normalize_to_angle=normalize_angle,
            title=title,
            charts="all",
        )


def save_driver_dashboard_html(
    response: "FrequencyResponse",
    filename: str = "driver_dashboard.html",
    *,
    title: Optional[str] = None,
    distance: float = 1.0,
    max_angle: float = 90.0,
    normalize_angle: float = 10.0,
    structural_mode_hz: Optional[list[float]] = None,
    phase_plug_metrics: Optional[list[object]] = None,
    show_progress: bool = True,
    include_plotlyjs: str | bool = True,
) -> str:
    """
    Save an interactive Plotly HTML dashboard for design-stage inspection.

    This is a convenience wrapper around `bempp_audio.viz.plotly_dashboard`.
    """
    if getattr(response, "frequencies", None) is None or len(response.frequencies) == 0:
        raise ValueError("Cannot build dashboard from an empty FrequencyResponse (no frequencies).")

    from bempp_audio.viz.plotly_dashboard import DashboardOptions, save_driver_dashboard_html as _save

    opts = DashboardOptions(
        distance_m=float(distance),
        max_angle_deg=float(max_angle),
        normalize_angle_deg=float(normalize_angle),
        structural_mode_hz=tuple(float(x) for x in (structural_mode_hz or []) if float(x) > 0),
    )
    return _save(
        response,
        filename,
        title=title or "Driver + Waveguide Dashboard",
        options=opts,
        phase_plug_metrics=phase_plug_metrics,
        show_progress=bool(show_progress),
        include_plotlyjs=include_plotlyjs,
    )
