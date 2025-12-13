"""
Mesh topology checks that can run without BEM solves.

These utilities are intentionally lightweight and operate on the BEMPP grid
data (vertices/elements/domain indices).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Tuple
from collections import Counter
import numpy as np


@dataclass(frozen=True)
class MeshTopologyReport:
    n_vertices: int
    n_elements: int
    n_edges: int
    n_boundary_edges: int
    n_nonmanifold_edges: int
    n_duplicate_triangles: int

    @property
    def ok(self) -> bool:
        return self.n_duplicate_triangles == 0 and self.n_nonmanifold_edges == 0


def _edge_counter(elements: np.ndarray) -> Counter:
    edges = Counter()
    for tri in elements.T:
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
        edges[tuple(sorted((v0, v1)))] += 1
        edges[tuple(sorted((v1, v2)))] += 1
        edges[tuple(sorted((v2, v0)))] += 1
    return edges


def _duplicate_triangle_count(elements: np.ndarray) -> int:
    seen = Counter()
    for tri in elements.T:
        key = tuple(sorted((int(tri[0]), int(tri[1]), int(tri[2]))))
        seen[key] += 1
    return sum(1 for _, c in seen.items() if c > 1)


def topology_report_arrays(vertices: np.ndarray, elements: np.ndarray) -> MeshTopologyReport:
    """
    Compute basic topology stats from raw grid arrays.

    Parameters
    ----------
    vertices:
        Vertex array shaped (3, n_vertices).
    elements:
        Element array shaped (3, n_elements) containing vertex indices.
    """
    verts = np.asarray(vertices)
    elems = np.asarray(elements)
    if verts.ndim != 2 or verts.shape[0] != 3:
        raise ValueError("vertices must have shape (3, n_vertices)")
    if elems.ndim != 2 or elems.shape[0] != 3:
        raise ValueError("elements must have shape (3, n_elements)")

    edges = _edge_counter(elems)
    boundary_edges = sum(1 for _, c in edges.items() if c == 1)
    nonmanifold_edges = sum(1 for _, c in edges.items() if c > 2)
    dup_tris = _duplicate_triangle_count(elems)

    return MeshTopologyReport(
        n_vertices=int(verts.shape[1]),
        n_elements=int(elems.shape[1]),
        n_edges=int(len(edges)),
        n_boundary_edges=int(boundary_edges),
        n_nonmanifold_edges=int(nonmanifold_edges),
        n_duplicate_triangles=int(dup_tris),
    )


def topology_report(mesh) -> MeshTopologyReport:
    """
    Compute basic topology stats for a `LoudspeakerMesh`.
    """
    return topology_report_arrays(mesh.grid.vertices, mesh.grid.elements)


def require_domains(mesh, required: Iterable[int]) -> None:
    """Raise if any required domain IDs are missing."""
    required_set = set(int(x) for x in required)
    present = set(int(x) for x in np.unique(mesh.grid.domain_indices))
    missing = sorted(required_set - present)
    if missing:
        raise ValueError(f"Missing required domain IDs: {missing} (present: {sorted(present)})")


def require_manifold(mesh, allow_boundary: bool = True) -> None:
    """
    Raise if mesh contains duplicate triangles or non-manifold edges.

    Parameters
    ----------
    allow_boundary : bool
        If False, also reject boundary edges (requires a closed surface).
    """
    report = topology_report(mesh)
    if report.n_duplicate_triangles:
        raise ValueError(f"Mesh has {report.n_duplicate_triangles} duplicate triangles")
    if report.n_nonmanifold_edges:
        raise ValueError(f"Mesh has {report.n_nonmanifold_edges} non-manifold edges (edge count > 2)")
    if not allow_boundary and report.n_boundary_edges:
        raise ValueError(f"Mesh is not closed: {report.n_boundary_edges} boundary edges found")
