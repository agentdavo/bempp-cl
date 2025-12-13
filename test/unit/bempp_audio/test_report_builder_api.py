from __future__ import annotations

import numpy as np

from bempp_audio.viz.report import Panel, PanelConfig, PanelType, ReportBuilder


class _StubResponse:
    def __init__(self, *, has_driver_metrics: bool, has_waveguide: bool):
        self.frequencies = np.array([100.0, 200.0])
        self.results = []
        self.has_driver_metrics = has_driver_metrics
        self.has_waveguide = has_waveguide


def test_panel_factories_return_panel_configs():
    assert isinstance(Panel.polar_map(), PanelConfig)
    assert Panel.polar_map().panel_type == PanelType.POLAR_MAP
    assert Panel.spl_curves().panel_type == PanelType.SPL_CURVES
    assert Panel.radiation_impedance().panel_type == PanelType.RADIATION_IMPEDANCE
    assert Panel.directivity_index().panel_type == PanelType.DIRECTIVITY_INDEX
    assert Panel.isobars().panel_type == PanelType.ISOBARS
    assert Panel.coverage_uniformity().panel_type == PanelType.COVERAGE_UNIFORMITY
    assert Panel.pattern_control().panel_type == PanelType.PATTERN_CONTROL
    assert Panel.throat_impedance().panel_type == PanelType.THROAT_IMPEDANCE
    assert Panel.offaxis_family().panel_type == PanelType.OFFAXIS_FAMILY
    assert Panel.profile_compare().panel_type == PanelType.PROFILE_COMPARE


def test_report_builder_presets_populate_panels_adaptively():
    rb = ReportBuilder(_StubResponse(has_driver_metrics=True, has_waveguide=True)).preset_waveguide_summary()
    types = [p.panel_type for p in rb._panels]
    assert PanelType.POLAR_MAP in types
    assert PanelType.SPL_CURVES in types
    assert PanelType.RADIATION_IMPEDANCE in types
    assert PanelType.DIRECTIVITY_INDEX in types
    assert PanelType.DRIVER_IMPEDANCE in types
    assert PanelType.DRIVER_EXCURSION in types
    assert PanelType.WAVEGUIDE_XSECTION in types

    rb2 = ReportBuilder(_StubResponse(has_driver_metrics=False, has_waveguide=False)).preset_waveguide_summary()
    types2 = [p.panel_type for p in rb2._panels]
    assert PanelType.DRIVER_IMPEDANCE not in types2
    assert PanelType.WAVEGUIDE_XSECTION not in types2
