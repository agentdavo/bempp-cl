"""
Composable report builder for visualization panels.

This module provides a small "report builder" abstraction that assembles
matplotlib panels defined in `bempp_audio/viz/panels/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

import numpy as np

from bempp_audio.viz.data.polar_data import PolarMapData
from bempp_audio.viz.style import DEFAULT_PLOT_CONFIG, setup_style as _setup_style

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    plt = None
    GridSpec = None

if TYPE_CHECKING:
    from bempp_audio.results import FrequencyResponse


class PanelType(str, Enum):
    POLAR_MAP = "polar_map"
    SPL_CURVES = "spl_curves"
    DIRECTIVITY_INDEX = "directivity_index"
    BEAMWIDTH = "beamwidth"
    RADIATION_IMPEDANCE = "radiation_impedance"
    DRIVER_IMPEDANCE = "driver_impedance"
    DRIVER_EXCURSION = "driver_excursion"
    WAVEGUIDE_XSECTION = "waveguide_xsection"
    THROAT_IMPEDANCE = "throat_impedance"
    ISOBARS = "isobars"
    COVERAGE_UNIFORMITY = "coverage_uniformity"
    PATTERN_CONTROL = "pattern_control"
    POLAR_SINGLE = "polar_single"
    OFFAXIS_FAMILY = "offaxis_family"
    PROFILE_COMPARE = "profile_compare"
    CUSTOM = "custom"


@dataclass(frozen=True)
class PanelConfig:
    panel_type: PanelType
    kwargs: dict[str, Any] = field(default_factory=dict)
    title: Optional[str] = None


class Panel:
    """Factory for commonly-used panels."""

    @staticmethod
    def polar_map(*, cmap: str = "jet") -> PanelConfig:
        return PanelConfig(PanelType.POLAR_MAP, kwargs={"cmap": cmap})

    @staticmethod
    def spl_curves(*, angles_deg: Sequence[float] = (0, 15, 30, 45, 60, 90)) -> PanelConfig:
        return PanelConfig(PanelType.SPL_CURVES, kwargs={"angles_deg": [float(a) for a in angles_deg]})

    @staticmethod
    def radiation_impedance() -> PanelConfig:
        return PanelConfig(PanelType.RADIATION_IMPEDANCE)

    @staticmethod
    def directivity_index(*, di_angles_deg: Sequence[float] = (0, 10, 20), level_db: float = -6.0, plane: str = "xz") -> PanelConfig:
        return PanelConfig(
            PanelType.DIRECTIVITY_INDEX,
            kwargs={"di_angles_deg": [float(a) for a in di_angles_deg], "level_db": float(level_db), "plane": str(plane)},
        )

    @staticmethod
    def driver_impedance() -> PanelConfig:
        return PanelConfig(PanelType.DRIVER_IMPEDANCE)

    @staticmethod
    def driver_excursion() -> PanelConfig:
        return PanelConfig(PanelType.DRIVER_EXCURSION)

    @staticmethod
    def waveguide_xsection() -> PanelConfig:
        return PanelConfig(PanelType.WAVEGUIDE_XSECTION)

    @staticmethod
    def throat_impedance() -> PanelConfig:
        return PanelConfig(PanelType.THROAT_IMPEDANCE)

    @staticmethod
    def isobars(*, levels_db: Sequence[float] = (-3, -6, -10, -20)) -> PanelConfig:
        return PanelConfig(PanelType.ISOBARS, kwargs={"levels_db": [float(x) for x in levels_db]})

    @staticmethod
    def coverage_uniformity(*, coverage_angle_deg: float = 60.0, metric: str = "std") -> PanelConfig:
        return PanelConfig(
            PanelType.COVERAGE_UNIFORMITY,
            kwargs={"coverage_angle_deg": float(coverage_angle_deg), "metric": str(metric)},
        )

    @staticmethod
    def pattern_control(
        *,
        target_beamwidth_deg: float = 90.0,
        tolerance_deg: float = 10.0,
        level_db: float = -6.0,
        plane: str = "xz",
    ) -> PanelConfig:
        return PanelConfig(
            PanelType.PATTERN_CONTROL,
            kwargs={
                "target_beamwidth_deg": float(target_beamwidth_deg),
                "tolerance_deg": float(tolerance_deg),
                "level_db": float(level_db),
                "plane": str(plane),
            },
        )

    @staticmethod
    def offaxis_family(
        *,
        angles_deg: Sequence[float] = (0, 10, 20, 30, 45, 60, 90),
        offset_db: float = 3.0,
    ) -> PanelConfig:
        return PanelConfig(
            PanelType.OFFAXIS_FAMILY,
            kwargs={"angles_deg": [float(a) for a in angles_deg], "offset_db": float(offset_db)},
        )

    @staticmethod
    def profile_compare(
        *,
        profiles: Optional[Sequence[object]] = None,
        labels: Optional[Sequence[str]] = None,
        include_current: bool = True,
    ) -> PanelConfig:
        return PanelConfig(
            PanelType.PROFILE_COMPARE,
            kwargs={
                "profiles": list(profiles) if profiles is not None else [],
                "labels": list(labels) if labels is not None else None,
                "include_current": bool(include_current),
            },
        )

    @staticmethod
    def custom(render_func: Callable[..., Any], **kwargs: Any) -> PanelConfig:
        return PanelConfig(PanelType.CUSTOM, kwargs={"render_func": render_func, **kwargs})


@dataclass
class ReportBuilder:
    """
    Compose and render a multi-panel report from a `FrequencyResponse`.

    This is intentionally minimal; presets cover the common layouts while the
    public `add()` method allows manual customization.
    """

    response: "FrequencyResponse"
    _title: str = "Acoustic Radiation Summary"
    _panels: list[PanelConfig] = field(default_factory=list)

    # Common shared options
    _distance_m: float = 1.0
    _max_angle_deg: float = 180.0
    _normalize_angle_deg: float = 0.0
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def title(self, title: str) -> "ReportBuilder":
        self._title = str(title)
        return self

    def distance(self, distance_m: float) -> "ReportBuilder":
        self._distance_m = float(distance_m)
        return self

    def polar_options(self, *, max_angle_deg: float = 180.0, normalize_angle_deg: float = 0.0) -> "ReportBuilder":
        self._max_angle_deg = float(max_angle_deg)
        self._normalize_angle_deg = float(normalize_angle_deg)
        return self

    def clear(self) -> "ReportBuilder":
        self._panels.clear()
        return self

    def add(self, panel: PanelConfig) -> "ReportBuilder":
        self._panels.append(panel)
        return self

    # ---------------------------------------------------------------------
    # Presets
    # ---------------------------------------------------------------------

    def preset_waveguide_summary(
        self,
        *,
        include_driver: Optional[bool] = None,
        include_waveguide: Optional[bool] = None,
        angles_deg: Sequence[float] = (0, 15, 30, 45, 60, 90),
    ) -> "ReportBuilder":
        """
        Similar to `mpl.summary_overview()`, but composed from panels.
        """
        has_driver = bool(self.response.has_driver_metrics) if include_driver is None else bool(include_driver)
        has_waveguide = bool(self.response.has_waveguide) if include_waveguide is None else bool(include_waveguide)

        self.clear()
        self.add(Panel.polar_map())
        self.add(Panel.spl_curves(angles_deg=angles_deg))
        self.add(Panel.radiation_impedance())
        self.add(Panel.directivity_index())
        if has_driver:
            self.add(Panel.driver_impedance())
            self.add(Panel.driver_excursion())
        if has_waveguide:
            self.add(Panel.waveguide_xsection())
        return self

    def preset_waveguide_designer(
        self,
        *,
        angles_deg: Sequence[float] = (0, 10, 20, 30, 45, 60, 90),
        include_waveguide: Optional[bool] = None,
    ) -> "ReportBuilder":
        """
        Waveguide designer preset: coverage + driver matching + geometry.

        Notes
        -----
        If the builder has not been configured, this preset defaults to normalizing
        to 10° off-axis (common horn measurement convention).
        """
        has_waveguide = bool(self.response.has_waveguide) if include_waveguide is None else bool(include_waveguide)

        if float(self._normalize_angle_deg) == 0.0:
            self._normalize_angle_deg = 10.0

        self.clear()
        self.add(Panel.polar_map())
        self.add(Panel.isobars())
        self.add(Panel.offaxis_family(angles_deg=angles_deg, offset_db=3.0))
        self.add(Panel.spl_curves(angles_deg=angles_deg))
        self.add(Panel.directivity_index())
        self.add(Panel.coverage_uniformity(coverage_angle_deg=60.0, metric="std"))
        self.add(Panel.pattern_control(target_beamwidth_deg=90.0, tolerance_deg=10.0, level_db=-6.0, plane="xz"))
        if has_waveguide:
            self.add(Panel.throat_impedance())
            self.add(Panel.waveguide_xsection())
        return self

    def preset_compression_driver(self) -> "ReportBuilder":
        """
        Driver-focused preset: impedance + excursion plus the usual radiation/DI context.
        """
        self.clear()
        self.add(Panel.driver_impedance())
        self.add(Panel.driver_excursion())
        self.add(Panel.radiation_impedance())
        self.add(Panel.directivity_index())
        return self

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------

    def _needs_polar(self) -> bool:
        return any(
            p.panel_type
            in {
                PanelType.POLAR_MAP,
                PanelType.SPL_CURVES,
                PanelType.DRIVER_EXCURSION,
                PanelType.ISOBARS,
                PanelType.COVERAGE_UNIFORMITY,
                PanelType.OFFAXIS_FAMILY,
            }
            for p in self._panels
        )

    def render(
        self,
        *,
        figsize: Optional[tuple[float, float]] = None,
        dpi: Optional[int] = None,
        show_progress: bool = True,
    ):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for report rendering")

        if dpi is None:
            dpi = DEFAULT_PLOT_CONFIG.dpi

        _setup_style()

        panels = list(self._panels) if self._panels else [Panel.polar_map(), Panel.spl_curves(), Panel.radiation_impedance(), Panel.directivity_index()]

        polar: Optional[PolarMapData] = None
        norm = None
        theta_symmetric = None
        spl_symmetric = None
        spl_ref = None

        cache_key = (
            f"polar:{float(self._distance_m)}:{float(self._max_angle_deg)}:5.0:horizontal:20e-6:"
            f"{float(self._normalize_angle_deg)}"
        )
        if self._needs_polar():
            cached = self._cache.get(cache_key)
            if isinstance(cached, dict):
                polar = cached.get("polar")
                norm = cached.get("norm")
                theta_symmetric = cached.get("theta_symmetric")
                spl_symmetric = cached.get("spl_symmetric")
                spl_ref = cached.get("spl_ref")

            if polar is None or norm is None or theta_symmetric is None or spl_symmetric is None or spl_ref is None:
                polar = PolarMapData.from_response(
                    self.response,
                    distance_m=float(self._distance_m),
                    max_angle_deg=float(self._max_angle_deg),
                    step_deg=5.0,
                    plane="horizontal",
                    ref_pa=20e-6,
                    show_progress=bool(show_progress),
                )
                norm, theta_symmetric, spl_symmetric = polar.normalized_symmetric(
                    normalize_angle_deg=float(self._normalize_angle_deg)
                )
                spl_ref = norm.spl_ref_db
                self._cache[cache_key] = {
                    "polar": polar,
                    "norm": norm,
                    "theta_symmetric": theta_symmetric,
                    "spl_symmetric": spl_symmetric,
                    "spl_ref": spl_ref,
                }

        # Simple packing layout: 2 columns, with full-width panels spanning both.
        full_width = {PanelType.WAVEGUIDE_XSECTION}

        layout: list[tuple[PanelConfig, int, int, int]] = []  # (panel, row, col, colspan)
        row = 0
        col = 0
        for panel in panels:
            if panel.panel_type in full_width:
                if col != 0:
                    row += 1
                    col = 0
                layout.append((panel, row, 0, 2))
                row += 1
                col = 0
            else:
                layout.append((panel, row, col, 1))
                col += 1
                if col >= 2:
                    row += 1
                    col = 0
        n_rows = row + (1 if col != 0 else 0)
        n_rows = max(n_rows, 1)

        if figsize is None:
            figsize = (DEFAULT_PLOT_CONFIG.figsize[0], DEFAULT_PLOT_CONFIG.figsize[1] * 0.5 * n_rows)

        fig = plt.figure(figsize=figsize, dpi=int(dpi))
        gs = GridSpec(n_rows, 2, figure=fig, hspace=0.35, wspace=0.25)
        fig.suptitle(self._title, fontsize=DEFAULT_PLOT_CONFIG.font_size_title + 2, fontweight="bold")

        # Lazy import to avoid import cycles (panels import PanelType from this module).
        from bempp_audio.viz.panels import PANEL_RENDERERS

        freqs = np.asarray(self.response.frequencies, dtype=float)

        for panel, r, c, colspan in layout:
            ax = fig.add_subplot(gs[r, c] if colspan == 1 else gs[r, :])

            if panel.title:
                ax.set_title(str(panel.title))

            if panel.panel_type == PanelType.POLAR_MAP:
                if polar is None or norm is None:
                    raise RuntimeError("POLAR_MAP requires polar data")
                renderer = PANEL_RENDERERS[PanelType.POLAR_MAP]
                renderer(
                    ax,
                    frequencies_hz=freqs,
                    angles_symmetric_deg=theta_symmetric,
                    spl_symmetric_db=spl_symmetric,
                    max_angle_deg=float(self._max_angle_deg),
                    norm_angle_used_deg=float(norm.norm_angle_used_deg),
                    **panel.kwargs,
                )
                continue

            if panel.panel_type == PanelType.SPL_CURVES:
                if polar is None or spl_ref is None or norm is None:
                    raise RuntimeError("SPL_CURVES requires polar data")
                renderer = PANEL_RENDERERS[PanelType.SPL_CURVES]
                angles_deg = panel.kwargs.get("angles_deg", [0, 15, 30, 45, 60, 90])
                renderer(
                    ax,
                    self.response,
                    polar=polar,
                    spl_ref_db=spl_ref,
                    norm_angle_used_deg=float(norm.norm_angle_used_deg),
                    angles_deg=list(angles_deg),
                    distance_m=float(self._distance_m),
                    ref_pa=20e-6,
                    show_progress=bool(show_progress),
                )
                continue

            if panel.panel_type == PanelType.RADIATION_IMPEDANCE:
                renderer = PANEL_RENDERERS[PanelType.RADIATION_IMPEDANCE]
                renderer(ax, self.response, show_progress=bool(show_progress))
                continue

            if panel.panel_type == PanelType.DIRECTIVITY_INDEX:
                renderer = PANEL_RENDERERS[PanelType.DIRECTIVITY_INDEX]
                renderer(ax, self.response, **panel.kwargs)
                continue

            if panel.panel_type == PanelType.DRIVER_IMPEDANCE:
                renderer = PANEL_RENDERERS[PanelType.DRIVER_IMPEDANCE]
                renderer(ax, self.response)
                continue

            if panel.panel_type == PanelType.DRIVER_EXCURSION:
                if polar is None:
                    raise RuntimeError("DRIVER_EXCURSION requires polar data for on-axis SPL")
                renderer = PANEL_RENDERERS[PanelType.DRIVER_EXCURSION]
                spl_on_axis = polar.spl_map_db[0, :] if polar.spl_map_db.size else np.zeros_like(freqs, dtype=float)
                renderer(
                    ax,
                    self.response,
                    distance_m=float(self._distance_m),
                    spl_on_axis_db=spl_on_axis,
                )
                continue

            if panel.panel_type == PanelType.WAVEGUIDE_XSECTION:
                renderer = PANEL_RENDERERS[PanelType.WAVEGUIDE_XSECTION]
                renderer(ax, self.response)
                continue

            if panel.panel_type == PanelType.THROAT_IMPEDANCE:
                renderer = PANEL_RENDERERS[PanelType.THROAT_IMPEDANCE]
                renderer(ax, self.response)
                continue

            if panel.panel_type == PanelType.ISOBARS:
                if polar is None or norm is None:
                    raise RuntimeError("ISOBARS requires polar data")
                renderer = PANEL_RENDERERS[PanelType.ISOBARS]
                renderer(
                    ax,
                    frequencies_hz=freqs,
                    angles_symmetric_deg=theta_symmetric,
                    spl_symmetric_db=spl_symmetric,
                    **panel.kwargs,
                )
                continue

            if panel.panel_type == PanelType.COVERAGE_UNIFORMITY:
                if polar is None or norm is None:
                    raise RuntimeError("COVERAGE_UNIFORMITY requires polar data")
                renderer = PANEL_RENDERERS[PanelType.COVERAGE_UNIFORMITY]
                renderer(
                    ax,
                    frequencies_hz=freqs,
                    angles_deg=theta_symmetric,
                    spl_map_db=spl_symmetric,
                    **panel.kwargs,
                )
                continue

            if panel.panel_type == PanelType.PATTERN_CONTROL:
                renderer = PANEL_RENDERERS[PanelType.PATTERN_CONTROL]
                renderer(ax, self.response, **panel.kwargs)
                continue

            if panel.panel_type == PanelType.OFFAXIS_FAMILY:
                if polar is None or norm is None or spl_ref is None:
                    raise RuntimeError("OFFAXIS_FAMILY requires polar data")
                renderer = PANEL_RENDERERS[PanelType.OFFAXIS_FAMILY]
                angles_deg = panel.kwargs.get("angles_deg", [0, 10, 20, 30, 45, 60, 90])
                renderer(
                    ax,
                    self.response,
                    polar=polar,
                    spl_ref_db=spl_ref,
                    norm_angle_used_deg=float(norm.norm_angle_used_deg),
                    angles_deg=list(angles_deg),
                    offset_db=float(panel.kwargs.get("offset_db", 3.0)),
                    show_progress=bool(show_progress),
                )
                continue

            if panel.panel_type == PanelType.PROFILE_COMPARE:
                renderer = PANEL_RENDERERS[PanelType.PROFILE_COMPARE]
                renderer(ax, self.response, **panel.kwargs)
                continue

            if panel.panel_type == PanelType.CUSTOM:
                render_func = panel.kwargs.get("render_func")
                if not callable(render_func):
                    raise ValueError("CUSTOM panel requires render_func")
                render_func(ax=ax, response=self.response, **{k: v for k, v in panel.kwargs.items() if k != "render_func"})
                continue

            raise NotImplementedError(f"Panel type not wired: {panel.panel_type}")

        with np.errstate(all="ignore"):
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def save(self, filename: str, **render_kwargs: Any) -> str:
        fig = self.render(**render_kwargs)
        fig.savefig(str(filename), dpi=render_kwargs.get("dpi", DEFAULT_PLOT_CONFIG.dpi), bbox_inches="tight")
        return str(filename)
