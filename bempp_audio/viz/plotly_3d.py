"""
Plotly-based 3D/HTML exports for bempp_audio.

Separated from the matplotlib panels/report builder to keep optional plotly usage isolated.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None


def mesh_3d_html(
    mesh: "LoudspeakerMesh",
    filename: str = "mesh_3d.html",
    title: str = "Mesh Geometry",
    color: str = "#4488ff",
    opacity: float = 1.0,
    show_edges: bool = True,
    color_by_domain: bool = True,
    domain_colors: dict = None,
    domain_names: dict = None,
    edge_color: str = "black",
    edge_width: float = 1,
    **kwargs,
) -> str:
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly required for HTML export. Install with: pip install plotly")

    grid = mesh.grid
    vertices = grid.vertices.T  # (n_vertices, 3)
    elements = grid.elements.T  # (n_elements, 3)
    domain_indices = grid.domain_indices

    data = []

    default_domain_colors = {
        0: "#FF4444",
        1: "#44AA44",
        2: "#4444FF",
        3: "#FFAA00",
        4: "#AA44AA",
        5: "#44AAAA",
    }

    if color_by_domain:
        unique_domains = np.unique(domain_indices)

        if domain_colors is None:
            domain_colors = {}
            for i, dom in enumerate(unique_domains):
                domain_colors[dom] = default_domain_colors.get(
                    i, f"hsl({(i * 137) % 360}, 70%, 50%)"
                )

        if domain_names is None:
            domain_names = {dom: f"Domain {dom}" for dom in unique_domains}

        for dom in unique_domains:
            mask = domain_indices == dom
            dom_elements = elements[mask]
            if len(dom_elements) == 0:
                continue

            data.append(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=dom_elements[:, 0],
                    j=dom_elements[:, 1],
                    k=dom_elements[:, 2],
                    color=domain_colors.get(dom, color),
                    opacity=opacity,
                    flatshading=True,
                    name=domain_names.get(dom, f"Domain {dom}"),
                    showlegend=True,
                )
            )
    else:
        data.append(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=elements[:, 0],
                j=elements[:, 1],
                k=elements[:, 2],
                color=color,
                opacity=opacity,
                flatshading=True,
                name="Mesh",
            )
        )

    if show_edges:
        edge_x, edge_y, edge_z = [], [], []
        for tri in elements:
            for i in range(3):
                p1 = vertices[tri[i]]
                p2 = vertices[tri[(i + 1) % 3]]
                edge_x.extend([p1[0], p2[0], None])
                edge_y.extend([p1[1], p2[1], None])
                edge_z.extend([p1[2], p2[2], None])

        data.append(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color=edge_color, width=edge_width),
                name="Mesh edges",
                showlegend=False,
            )
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.write_html(filename, include_plotlyjs=True, full_html=True)
    return filename


def gmsh_msh_boundary_html(
    *,
    msh_file: str,
    filename: str = "gmsh_boundary_3d.html",
    title: str = "Gmsh Boundary Mesh",
    opacity: float = 0.9,
    show_edges: bool = True,
) -> str:
    """
    Render the boundary triangles from a Gmsh `.msh` file as an interactive HTML.

    This is useful for FEM-style 3D meshes (tetrahedra) where you still want
    a surface preview (boundary facets) without loading into bempp.
    Requires `plotly` and `meshio`.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly required for HTML export. Install with: pip install plotly")

    try:
        import meshio  # type: ignore
    except Exception as e:
        raise ImportError("meshio required for reading .msh. Install with: pip install meshio") from e

    mesh = meshio.read(msh_file)
    points = np.asarray(mesh.points, dtype=float)
    if points.shape[1] == 2:
        pts = np.zeros((points.shape[0], 3), dtype=float)
        pts[:, :2] = points
        points = pts

    tri = None
    tri_tags = None
    for block_idx, cell_block in enumerate(mesh.cells):
        if cell_block.type == "triangle":
            tri = np.asarray(cell_block.data, dtype=np.int64)
            if mesh.cell_data and "gmsh:physical" in mesh.cell_data:
                tri_tags = np.asarray(mesh.cell_data["gmsh:physical"][block_idx], dtype=int)
            break

    if tri is None:
        raise ValueError(f"No triangle cells found in {msh_file!r}")

    data = []
    if tri_tags is None:
        data.append(
            go.Mesh3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                i=tri[:, 0],
                j=tri[:, 1],
                k=tri[:, 2],
                color="#4488ff",
                opacity=opacity,
                flatshading=True,
                name="Boundary",
                showlegend=False,
            )
        )
    else:
        unique = np.unique(tri_tags)
        palette = [
            "#FF4444",
            "#44AA44",
            "#4444FF",
            "#FFAA00",
            "#AA44AA",
            "#44AAAA",
        ]
        for idx, tag in enumerate(unique):
            mask = tri_tags == tag
            tris = tri[mask]
            if tris.size == 0:
                continue
            data.append(
                go.Mesh3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    i=tris[:, 0],
                    j=tris[:, 1],
                    k=tris[:, 2],
                    color=palette[idx % len(palette)],
                    opacity=opacity,
                    flatshading=True,
                    name=f"Physical {int(tag)}",
                    showlegend=True,
                )
            )

    if show_edges:
        edge_x, edge_y, edge_z = [], [], []
        for t in tri:
            for i0, i1 in ((0, 1), (1, 2), (2, 0)):
                p1 = points[t[i0]]
                p2 = points[t[i1]]
                edge_x.extend([p1[0], p2[0], None])
                edge_y.extend([p1[1], p2[1], None])
                edge_z.extend([p1[2], p2[2], None])
        data.append(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color="black", width=1),
                name="Edges",
                showlegend=False,
            )
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.write_html(filename, include_plotlyjs=True, full_html=True)
    return filename


def pressure_field_3d_html(
    result: "RadiationResult",
    filename: str = "pressure_3d.html",
    quantity: str = "spl",
    title: str = None,
    colorscale: str = "Jet",
    **kwargs,
) -> str:
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly required for HTML export")

    grid = result.mesh.grid
    vertices = grid.vertices.T
    elements = grid.elements.T

    # Evaluate the grid function on vertices for visualization (handles DP0/P1 consistently).
    vertex_pressure = result.surface_pressure.evaluate_on_vertices()[0, :]

    if quantity == "spl":
        values = 20 * np.log10(np.maximum(np.abs(vertex_pressure) / np.sqrt(2), 1e-20) / 20e-6)
        colorbar_title = "SPL (dB)"
    elif quantity == "magnitude":
        values = np.abs(vertex_pressure)
        colorbar_title = "|p| (Pa)"
    elif quantity == "phase":
        values = np.degrees(np.angle(vertex_pressure))
        colorbar_title = "Phase (°)"
    elif quantity == "real":
        values = np.real(vertex_pressure)
        colorbar_title = "Re(p) (Pa)"
    elif quantity == "imag":
        values = np.imag(vertex_pressure)
        colorbar_title = "Im(p) (Pa)"
    else:
        raise ValueError(f"Unknown quantity: {quantity}")

    if title is None:
        title = f"Surface Pressure at {result.frequency:.0f} Hz"

    vertex_values = np.asarray(values, dtype=float)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=elements[:, 0],
                j=elements[:, 1],
                k=elements[:, 2],
                intensity=vertex_values,
                colorscale=colorscale,
                colorbar=dict(title=colorbar_title),
                flatshading=False,
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(filename, include_plotlyjs=True, full_html=True)
    return filename


def directivity_balloon_html(
    result: "RadiationResult",
    filename: str = "directivity_3d.html",
    n_theta: int = 37,
    n_phi: int = 73,
    title: str = None,
    colorscale: str = "RdBu_r",
    **kwargs,
) -> str:
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly required for HTML export")

    dp = result.directivity()
    balloon = dp.balloon_3d(n_theta=n_theta, n_phi=n_phi)

    X, Y, Z = balloon.to_cartesian()
    mag_db = balloon.magnitude_db(normalize=True)

    if title is None:
        title = f"Directivity Balloon at {result.frequency:.0f} Hz"

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=mag_db,
                colorscale=colorscale,
                cmin=-30,
                cmax=0,
                colorbar=dict(title="dB (normalized)"),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(filename, include_plotlyjs=True, full_html=True)
    return filename


def frequency_animation_html(
    response: "FrequencyResponse",
    filename: str = "frequency_sweep.html",
    quantity: str = "spl",
    title: str = "Frequency Sweep Animation",
    colorscale: str = "Jet",
    **kwargs,
) -> str:
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly required for HTML export")

    grid = response.results[0].mesh.grid
    vertices = grid.vertices.T
    elements = grid.elements.T

    frames = []
    slider_steps = []

    for i, result in enumerate(response.results):
        vertex_pressure = result.surface_pressure.evaluate_on_vertices()[0, :]

        if quantity == "spl":
            values = 20 * np.log10(np.maximum(np.abs(vertex_pressure) / np.sqrt(2), 1e-20) / 20e-6)
        elif quantity == "magnitude":
            values = np.abs(vertex_pressure)
        elif quantity == "phase":
            values = np.degrees(np.angle(vertex_pressure))
        else:
            values = np.abs(vertex_pressure)

        vertex_values = np.asarray(values, dtype=float)

        frames.append(
            go.Frame(
                data=[
                    go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=elements[:, 0],
                        j=elements[:, 1],
                        k=elements[:, 2],
                        intensity=vertex_values,
                        colorscale=colorscale,
                        flatshading=False,
                    )
                ],
                name=str(i),
            )
        )

        slider_steps.append(
            dict(
                args=[[str(i)], dict(mode="immediate", frame=dict(redraw=True))],
                label=f"{result.frequency:.0f} Hz",
                method="animate",
            )
        )

    first_result = response.results[0]
    vertex_pressure = first_result.surface_pressure.evaluate_on_vertices()[0, :]
    if quantity == "spl":
        values = 20 * np.log10(np.maximum(np.abs(vertex_pressure) / np.sqrt(2), 1e-20) / 20e-6)
    else:
        values = np.abs(vertex_pressure)

    vertex_values = np.asarray(values, dtype=float)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=elements[:, 0],
                j=elements[:, 1],
                k=elements[:, 2],
                intensity=vertex_values,
                colorscale=colorscale,
                colorbar=dict(title=quantity.upper()),
                flatshading=False,
            )
        ],
        frames=frames,
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                currentvalue=dict(prefix="Frequency: "),
                pad=dict(t=50),
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.1,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=500), fromcurrent=True)],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0), mode="immediate")],
                    ),
                ],
            )
        ],
        margin=dict(l=0, r=0, t=60, b=0),
    )

    fig.write_html(filename, include_plotlyjs=True, full_html=True)
    return filename
