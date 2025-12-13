"""
Mesh quality visualizations (Plotly HTML exports).

These are diagnostics to help tune mesh presets and sizing parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np

from bempp_audio.mesh.validation import ElementQualityMetrics

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    PLOTLY_AVAILABLE = False
    go = None


UnitsLike = Literal["m", "mm", "um"]


def _unit_scale(units: UnitsLike) -> float:
    if units == "m":
        return 1.0
    if units == "mm":
        return 1e3
    if units == "um":
        return 1e6
    raise ValueError("units must be one of: 'm', 'mm', 'um'")


def _edge_segments(vertices: np.ndarray, elements: np.ndarray):
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for tri in elements:
        for i in range(3):
            p1 = vertices[tri[i]]
            p2 = vertices[tri[(i + 1) % 3]]
            edge_x.extend([float(p1[0]), float(p2[0]), None])
            edge_y.extend([float(p1[1]), float(p2[1]), None])
            edge_z.extend([float(p1[2]), float(p2[2]), None])
    return edge_x, edge_y, edge_z


def save_edge_length_histogram_html(
    mesh: "LoudspeakerMesh",
    filename: str | Path,
    *,
    bins: int = 60,
    units: UnitsLike = "mm",
    title: str = "Edge Length Histogram",
) -> str:
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly required for HTML export. Install with: pip install plotly")

    grid = mesh.grid
    vertices = grid.vertices.T  # (n_vertices, 3)
    elements = grid.elements.T  # (n_elements, 3)

    # All edges (with duplicates across triangles; good enough for diagnostics).
    tri = elements
    p0 = vertices[tri[:, 0]]
    p1 = vertices[tri[:, 1]]
    p2 = vertices[tri[:, 2]]
    e01 = np.linalg.norm(p1 - p0, axis=1)
    e12 = np.linalg.norm(p2 - p1, axis=1)
    e20 = np.linalg.norm(p0 - p2, axis=1)
    edges = np.concatenate([e01, e12, e20])

    scale = _unit_scale(units)
    edges_u = edges * scale

    fig = go.Figure(
        data=[
            go.Histogram(
                x=edges_u,
                nbinsx=int(bins),
                name="Edge lengths",
                marker=dict(color="#4477ff"),
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title=f"Edge length ({units})",
        yaxis_title="Count",
        bargap=0.05,
    )

    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    return str(out_path)


def save_element_quality_colormap_html(
    mesh: "LoudspeakerMesh",
    filename: str | Path,
    *,
    metric: Literal["aspect_ratio", "area"] = "aspect_ratio",
    show_edges: bool = True,
    opacity: float = 0.95,
    title: Optional[str] = None,
) -> str:
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly required for HTML export. Install with: pip install plotly")

    grid = mesh.grid
    vertices = grid.vertices.T  # (n_vertices, 3)
    elements = grid.elements.T  # (n_elements, 3)

    if metric == "aspect_ratio":
        values = ElementQualityMetrics.compute_triangle_aspect_ratios(grid.vertices, grid.elements)
        colorbar_title = "Aspect ratio"
        default_title = "Mesh Quality (Aspect Ratio)"
    elif metric == "area":
        values = ElementQualityMetrics.compute_triangle_areas(grid.vertices, grid.elements)
        colorbar_title = "Area (m²)"
        default_title = "Mesh Quality (Triangle Area)"
    else:
        raise ValueError("metric must be one of: 'aspect_ratio', 'area'")

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=elements[:, 0],
            j=elements[:, 1],
            k=elements[:, 2],
            intensity=np.asarray(values, dtype=float),
            intensitymode="cell",
            colorscale="Viridis",
            opacity=float(opacity),
            flatshading=True,
            colorbar=dict(title=colorbar_title),
            name=str(metric),
            showscale=True,
        )
    )

    if show_edges:
        edge_x, edge_y, edge_z = _edge_segments(vertices, elements)
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color="rgba(0,0,0,0.35)", width=1),
                name="Edges",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title or default_title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    return str(out_path)

