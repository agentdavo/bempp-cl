"""Visualization tools for acoustic results (optional dependencies)."""

from bempp_audio._optional import UnavailableDependency

from bempp_audio.viz.cea2034 import CEA2034Chart, CEA2034Calculator, plot_cea2034
from bempp_audio.viz.reporting import save_summary, save_cea2034, save_all_plots, save_driver_dashboard_html
from bempp_audio.viz.report import ReportBuilder, Panel, PanelConfig, PanelType
from bempp_audio.viz.style import (
    setup_style,
    setup_freq_axis,
    DEFAULT_FREQ_MIN,
    DEFAULT_FREQ_MAX,
)

try:
    from bempp_audio.viz.plotly_3d import (
        mesh_3d_html,
        gmsh_msh_boundary_html,
        pressure_field_3d_html,
        directivity_balloon_html,
        frequency_animation_html,
    )
except ImportError as e:  # pragma: no cover
    mesh_3d_html = UnavailableDependency(
        name="plotly",
        install_hint="Install with `pip install plotly` or `pip install bempp-cl[audio]`.",
        original_error=e,
    )
    gmsh_msh_boundary_html = mesh_3d_html
    pressure_field_3d_html = mesh_3d_html
    directivity_balloon_html = mesh_3d_html
    frequency_animation_html = mesh_3d_html

try:
    from bempp_audio.viz.mesh_quality import (
        save_edge_length_histogram_html,
        save_element_quality_colormap_html,
    )
except ImportError as e:  # pragma: no cover
    save_edge_length_histogram_html = UnavailableDependency(
        name="plotly",
        install_hint="Install with `pip install plotly` or `pip install bempp-cl[audio]`.",
        original_error=e,
    )
    save_element_quality_colormap_html = save_edge_length_histogram_html

from bempp_audio.viz.scorecard import compute_design_scorecard, save_scorecard_json

__all__ = [
    "CEA2034Chart",
    "CEA2034Calculator",
    "plot_cea2034",
    "ReportBuilder",
    "Panel",
    "PanelConfig",
    "PanelType",
    "save_summary",
    "save_cea2034",
    "save_all_plots",
    "save_driver_dashboard_html",
    "setup_style",
    "setup_freq_axis",
    "DEFAULT_FREQ_MIN",
    "DEFAULT_FREQ_MAX",
    "mesh_3d_html",
    "gmsh_msh_boundary_html",
    "pressure_field_3d_html",
    "directivity_balloon_html",
    "frequency_animation_html",
    "save_edge_length_histogram_html",
    "save_element_quality_colormap_html",
    "compute_design_scorecard",
    "save_scorecard_json",
]
