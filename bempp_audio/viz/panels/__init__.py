"""
Panel renderers for matplotlib summary/report layouts.

Renderers are intentionally small and focused. They are primarily used by
`bempp_audio.viz.report.ReportBuilder` (and `FrequencyResponse.save_summary()`).
"""

from bempp_audio.viz.report import PanelType
from bempp_audio.viz.panels.polar_panel import render_polar_panel
from bempp_audio.viz.panels.spl_panel import render_spl_panel
from bempp_audio.viz.panels.impedance_panel import render_impedance_panel
from bempp_audio.viz.panels.directivity_panel import render_di_panel
from bempp_audio.viz.panels.driver_panel import (
    render_driver_panel,
    render_driver_impedance_panel,
    render_driver_excursion_panel,
)
from bempp_audio.viz.panels.isobar_panel import render_isobars
from bempp_audio.viz.panels.coverage_panel import render_coverage_uniformity, render_pattern_control
from bempp_audio.viz.panels.offaxis_panel import render_normalized_offaxis_family
from bempp_audio.viz.panels.profile_compare_panel import render_profile_compare
from bempp_audio.viz.panels.waveguide_panel import render_xsection_panel, render_throat_impedance

PANEL_RENDERERS = {
    PanelType.POLAR_MAP: render_polar_panel,
    PanelType.SPL_CURVES: render_spl_panel,
    PanelType.RADIATION_IMPEDANCE: render_impedance_panel,
    PanelType.DIRECTIVITY_INDEX: render_di_panel,
    PanelType.DRIVER_IMPEDANCE: render_driver_impedance_panel,
    PanelType.DRIVER_EXCURSION: render_driver_excursion_panel,
    PanelType.WAVEGUIDE_XSECTION: render_xsection_panel,
    PanelType.THROAT_IMPEDANCE: render_throat_impedance,
    PanelType.ISOBARS: render_isobars,
    PanelType.COVERAGE_UNIFORMITY: render_coverage_uniformity,
    PanelType.PATTERN_CONTROL: render_pattern_control,
    PanelType.OFFAXIS_FAMILY: render_normalized_offaxis_family,
    PanelType.PROFILE_COMPARE: render_profile_compare,
}

__all__ = [
    "PANEL_RENDERERS",
    "PanelType",
    "render_polar_panel",
    "render_spl_panel",
    "render_impedance_panel",
    "render_di_panel",
    "render_driver_panel",
    "render_driver_impedance_panel",
    "render_driver_excursion_panel",
    "render_xsection_panel",
    "render_throat_impedance",
    "render_isobars",
    "render_coverage_uniformity",
    "render_pattern_control",
    "render_normalized_offaxis_family",
    "render_profile_compare",
]
