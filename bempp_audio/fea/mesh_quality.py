"""
Mesh quality validation for thin shell FEM analysis.

Provides quality metrics critical for accurate shell FEM:
- Aspect ratio (critical for thin shells to avoid shear locking)
- Jacobian/shape quality (important for curved elements)
- Element size vs wavelength (acoustic and bending wavelength resolution)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from bempp_audio.fea.materials import ShellMaterial, bending_wavelength


@dataclass
class MeshQualityReport:
    """
    Comprehensive mesh quality report.

    Attributes
    ----------
    num_elements : int
        Total number of triangular elements
    num_vertices : int
        Total number of vertices
    aspect_ratio_min : float
        Minimum aspect ratio (ideal = 1.0 for equilateral)
    aspect_ratio_max : float
        Maximum aspect ratio (>4 may cause numerical issues)
    aspect_ratio_mean : float
        Mean aspect ratio
    aspect_ratios : np.ndarray
        Per-element aspect ratios
    jacobian_min : float
        Minimum scaled Jacobian (range: -1 to 1, ideal > 0.5)
    jacobian_max : float
        Maximum scaled Jacobian
    jacobian_mean : float
        Mean scaled Jacobian
    jacobians : np.ndarray
        Per-element scaled Jacobians
    element_size_min : float
        Minimum element characteristic size [m]
    element_size_max : float
        Maximum element characteristic size [m]
    element_size_mean : float
        Mean element characteristic size [m]
    element_sizes : np.ndarray
        Per-element characteristic sizes [m]
    elements_per_acoustic_wavelength : Optional[float]
        Elements per acoustic wavelength at max frequency
    elements_per_bending_wavelength : Optional[float]
        Elements per bending wavelength at max frequency
    warnings : list[str]
        Quality warnings
    """

    num_elements: int
    num_vertices: int
    aspect_ratio_min: float
    aspect_ratio_max: float
    aspect_ratio_mean: float
    aspect_ratios: np.ndarray
    jacobian_min: float
    jacobian_max: float
    jacobian_mean: float
    jacobians: np.ndarray
    element_size_min: float
    element_size_max: float
    element_size_mean: float
    element_sizes: np.ndarray
    elements_per_acoustic_wavelength: Optional[float] = None
    elements_per_bending_wavelength: Optional[float] = None
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    @property
    def is_valid(self) -> bool:
        """Check if mesh passes all quality criteria."""
        return len(self.warnings) == 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Mesh Quality Report",
            "=" * 40,
            f"Elements: {self.num_elements}",
            f"Vertices: {self.num_vertices}",
            "",
            "Aspect Ratio (ideal=1.0, max recommended=4.0):",
            f"  Min: {self.aspect_ratio_min:.3f}",
            f"  Max: {self.aspect_ratio_max:.3f}",
            f"  Mean: {self.aspect_ratio_mean:.3f}",
            "",
            "Scaled Jacobian (ideal=1.0, min recommended=0.3):",
            f"  Min: {self.jacobian_min:.3f}",
            f"  Max: {self.jacobian_max:.3f}",
            f"  Mean: {self.jacobian_mean:.3f}",
            "",
            "Element Size [mm]:",
            f"  Min: {self.element_size_min * 1000:.3f}",
            f"  Max: {self.element_size_max * 1000:.3f}",
            f"  Mean: {self.element_size_mean * 1000:.3f}",
        ]

        if self.elements_per_acoustic_wavelength is not None:
            lines.append("")
            lines.append("Wavelength Resolution (recommended >= 6):")
            lines.append(
                f"  Elements per acoustic wavelength: {self.elements_per_acoustic_wavelength:.1f}"
            )
        if self.elements_per_bending_wavelength is not None:
            lines.append(
                f"  Elements per bending wavelength: {self.elements_per_bending_wavelength:.1f}"
            )

        if self.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        else:
            lines.append("")
            lines.append("✓ All quality checks passed")

        return "\n".join(lines)


def compute_aspect_ratios(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """
    Compute aspect ratios for triangular elements.

    The aspect ratio is defined as the ratio of the longest edge to the
    shortest altitude. For an equilateral triangle, this equals 1.0.
    Higher values indicate more elongated/degenerate elements.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) vertex coordinates
    triangles : np.ndarray
        (M, 3) triangle vertex indices

    Returns
    -------
    np.ndarray
        (M,) aspect ratios per element
    """
    aspect_ratios = np.zeros(len(triangles))

    for i, tri in enumerate(triangles):
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

        # Edge vectors and lengths
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        l0 = np.linalg.norm(e0)
        l1 = np.linalg.norm(e1)
        l2 = np.linalg.norm(e2)

        # Longest edge
        l_max = max(l0, l1, l2)

        # Area via cross product
        area = 0.5 * np.linalg.norm(np.cross(e0, -e2))

        if area > 1e-20:
            # Shortest altitude = 2 * area / longest_edge
            h_min = 2 * area / l_max
            # Aspect ratio = longest_edge / shortest_altitude
            # For equilateral: AR = 1 / (sqrt(3)/2) ≈ 1.155
            # Normalize so equilateral = 1.0
            aspect_ratios[i] = (l_max / h_min) / (2 / np.sqrt(3))
        else:
            # Degenerate element
            aspect_ratios[i] = np.inf

    return aspect_ratios


def compute_scaled_jacobians(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """
    Compute scaled Jacobian quality metric for triangular elements.

    The scaled Jacobian measures element shape quality:
    - 1.0 = perfect equilateral triangle
    - 0.0 = degenerate (zero area)
    - negative = inverted (wrong orientation)

    For flat triangles, this is equivalent to 2 * area / (sqrt(3) * l_max^2)
    where l_max is the longest edge. This is sometimes called the
    "shape quality" or "radius ratio" metric.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) vertex coordinates
    triangles : np.ndarray
        (M, 3) triangle vertex indices

    Returns
    -------
    np.ndarray
        (M,) scaled Jacobians per element (range: typically 0 to 1)
    """
    jacobians = np.zeros(len(triangles))

    for i, tri in enumerate(triangles):
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

        e0 = v1 - v0
        e1 = v2 - v0

        # Signed area via cross product (z-component for 2D, magnitude for 3D)
        cross = np.cross(e0, e1)
        area_signed = np.linalg.norm(cross) / 2

        # Edge lengths
        l0 = np.linalg.norm(e0)
        l1 = np.linalg.norm(e1)
        l2 = np.linalg.norm(v2 - v1)

        l_max = max(l0, l1, l2)

        if l_max > 1e-20:
            # Ideal equilateral area for edge l_max: sqrt(3)/4 * l_max^2
            ideal_area = np.sqrt(3) / 4 * l_max**2
            jacobians[i] = area_signed / ideal_area
        else:
            jacobians[i] = 0.0

    return jacobians


def compute_element_sizes(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """
    Compute characteristic element sizes for triangular elements.

    The characteristic size is defined as the diameter of the inscribed
    circle (incircle), which represents the "effective resolution" of
    the element.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) vertex coordinates
    triangles : np.ndarray
        (M, 3) triangle vertex indices

    Returns
    -------
    np.ndarray
        (M,) characteristic sizes per element [m]
    """
    sizes = np.zeros(len(triangles))

    for i, tri in enumerate(triangles):
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

        # Edge lengths
        a = np.linalg.norm(v1 - v2)
        b = np.linalg.norm(v0 - v2)
        c = np.linalg.norm(v0 - v1)

        # Semi-perimeter
        s = (a + b + c) / 2

        # Area via Heron's formula
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq > 0:
            area = np.sqrt(area_sq)
            # Incircle radius: r = area / s
            # Incircle diameter: d = 2 * area / s
            sizes[i] = 2 * area / s
        else:
            sizes[i] = 0.0

    return sizes


def validate_mesh_quality(
    vertices: np.ndarray,
    triangles: np.ndarray,
    material: Optional[ShellMaterial] = None,
    max_frequency_hz: Optional[float] = None,
    max_aspect_ratio: float = 4.0,
    min_jacobian: float = 0.3,
    min_elements_per_wavelength: int = 6,
    speed_of_sound: float = 343.0,
) -> MeshQualityReport:
    """
    Comprehensive mesh quality validation.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) vertex coordinates [m]
    triangles : np.ndarray
        (M, 3) triangle vertex indices
    material : ShellMaterial, optional
        Material for bending wavelength calculation
    max_frequency_hz : float, optional
        Maximum analysis frequency for wavelength checks [Hz]
    max_aspect_ratio : float
        Maximum acceptable aspect ratio (default: 4.0)
    min_jacobian : float
        Minimum acceptable scaled Jacobian (default: 0.3)
    min_elements_per_wavelength : int
        Minimum elements per wavelength (default: 6)
    speed_of_sound : float
        Speed of sound in air [m/s] (default: 343.0)

    Returns
    -------
    MeshQualityReport
        Comprehensive quality report with metrics and warnings
    """
    warnings = []

    # Compute metrics
    aspect_ratios = compute_aspect_ratios(vertices, triangles)
    jacobians = compute_scaled_jacobians(vertices, triangles)
    element_sizes = compute_element_sizes(vertices, triangles)

    # Aspect ratio checks
    ar_min = float(np.min(aspect_ratios[np.isfinite(aspect_ratios)]))
    ar_max = float(np.max(aspect_ratios[np.isfinite(aspect_ratios)]))
    ar_mean = float(np.mean(aspect_ratios[np.isfinite(aspect_ratios)]))

    num_bad_ar = np.sum(aspect_ratios > max_aspect_ratio)
    if num_bad_ar > 0:
        pct = 100 * num_bad_ar / len(triangles)
        warnings.append(
            f"Aspect ratio > {max_aspect_ratio} in {num_bad_ar} elements ({pct:.1f}%)"
        )

    # Jacobian checks
    j_min = float(np.min(jacobians))
    j_max = float(np.max(jacobians))
    j_mean = float(np.mean(jacobians))

    num_bad_j = np.sum(jacobians < min_jacobian)
    if num_bad_j > 0:
        pct = 100 * num_bad_j / len(triangles)
        warnings.append(
            f"Scaled Jacobian < {min_jacobian} in {num_bad_j} elements ({pct:.1f}%)"
        )

    num_inverted = np.sum(jacobians < 0)
    if num_inverted > 0:
        warnings.append(f"Inverted elements detected: {num_inverted}")

    # Element size stats
    size_min = float(np.min(element_sizes[element_sizes > 0]))
    size_max = float(np.max(element_sizes))
    size_mean = float(np.mean(element_sizes))

    # Wavelength validation
    elems_per_acoustic = None
    elems_per_bending = None

    if max_frequency_hz is not None and max_frequency_hz > 0:
        # Acoustic wavelength
        lambda_acoustic = speed_of_sound / max_frequency_hz
        elems_per_acoustic = lambda_acoustic / size_max

        if elems_per_acoustic < min_elements_per_wavelength:
            warnings.append(
                f"Only {elems_per_acoustic:.1f} elements per acoustic wavelength "
                f"at {max_frequency_hz/1000:.1f} kHz (need >= {min_elements_per_wavelength})"
            )

        # Bending wavelength (if material provided)
        if material is not None:
            lambda_bending = bending_wavelength(max_frequency_hz, material)
            elems_per_bending = lambda_bending / size_max

            if elems_per_bending < min_elements_per_wavelength:
                warnings.append(
                    f"Only {elems_per_bending:.1f} elements per bending wavelength "
                    f"at {max_frequency_hz/1000:.1f} kHz (need >= {min_elements_per_wavelength})"
                )

    return MeshQualityReport(
        num_elements=len(triangles),
        num_vertices=len(vertices),
        aspect_ratio_min=ar_min,
        aspect_ratio_max=ar_max,
        aspect_ratio_mean=ar_mean,
        aspect_ratios=aspect_ratios,
        jacobian_min=j_min,
        jacobian_max=j_max,
        jacobian_mean=j_mean,
        jacobians=jacobians,
        element_size_min=size_min,
        element_size_max=size_max,
        element_size_mean=size_mean,
        element_sizes=element_sizes,
        elements_per_acoustic_wavelength=elems_per_acoustic,
        elements_per_bending_wavelength=elems_per_bending,
        warnings=warnings,
    )


def find_poor_quality_elements(
    vertices: np.ndarray,
    triangles: np.ndarray,
    max_aspect_ratio: float = 4.0,
    min_jacobian: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find indices of elements with poor quality.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) vertex coordinates
    triangles : np.ndarray
        (M, 3) triangle vertex indices
    max_aspect_ratio : float
        Maximum acceptable aspect ratio
    min_jacobian : float
        Minimum acceptable scaled Jacobian

    Returns
    -------
    high_ar_indices : np.ndarray
        Indices of elements with aspect ratio > max_aspect_ratio
    low_j_indices : np.ndarray
        Indices of elements with Jacobian < min_jacobian
    """
    aspect_ratios = compute_aspect_ratios(vertices, triangles)
    jacobians = compute_scaled_jacobians(vertices, triangles)

    high_ar = np.where(aspect_ratios > max_aspect_ratio)[0]
    low_j = np.where(jacobians < min_jacobian)[0]

    return high_ar, low_j
