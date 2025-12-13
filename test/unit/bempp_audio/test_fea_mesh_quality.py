"""Tests for mesh quality validation in bempp_audio.fea."""

from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.fea.mesh_quality import (
    MeshQualityReport,
    compute_aspect_ratios,
    compute_scaled_jacobians,
    compute_element_sizes,
    validate_mesh_quality,
    find_poor_quality_elements,
)
from bempp_audio.fea.materials import ShellMaterial


class TestComputeAspectRatios:
    """Tests for compute_aspect_ratios function."""

    def test_equilateral_triangle_has_aspect_ratio_one(self):
        """Equilateral triangle should have aspect ratio ~1.0."""
        # Equilateral triangle with side length 1
        h = np.sqrt(3) / 2
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, h, 0.0],
        ])
        triangles = np.array([[0, 1, 2]])

        ar = compute_aspect_ratios(vertices, triangles)
        assert len(ar) == 1
        assert np.isclose(ar[0], 1.0, atol=0.01)

    def test_right_triangle_has_higher_aspect_ratio(self):
        """Right triangle (3-4-5) should have AR > 1."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ])
        triangles = np.array([[0, 1, 2]])

        ar = compute_aspect_ratios(vertices, triangles)
        # 3-4-5 right triangle has AR = 5 / h_min where h_min is shortest altitude
        # For 3-4-5: altitudes are 4, 3, 2.4, so AR = 5/2.4 / (2/sqrt(3)) ≈ 1.8
        assert ar[0] > 1.0
        assert ar[0] < 3.0

    def test_elongated_triangle_has_high_aspect_ratio(self):
        """Very elongated triangle should have high AR."""
        # Long thin triangle
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 0.1, 0.0],  # Very narrow
        ])
        triangles = np.array([[0, 1, 2]])

        ar = compute_aspect_ratios(vertices, triangles)
        assert ar[0] > 10.0  # Very high AR

    def test_degenerate_triangle_has_infinite_aspect_ratio(self):
        """Degenerate (zero-area) triangle should have inf AR."""
        # Collinear points
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        triangles = np.array([[0, 1, 2]])

        ar = compute_aspect_ratios(vertices, triangles)
        assert np.isinf(ar[0])


class TestComputeScaledJacobians:
    """Tests for compute_scaled_jacobians function."""

    def test_equilateral_triangle_has_jacobian_one(self):
        """Equilateral triangle should have scaled Jacobian = 1.0."""
        h = np.sqrt(3) / 2
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, h, 0.0],
        ])
        triangles = np.array([[0, 1, 2]])

        j = compute_scaled_jacobians(vertices, triangles)
        assert len(j) == 1
        assert np.isclose(j[0], 1.0, atol=0.01)

    def test_poor_quality_triangle_has_low_jacobian(self):
        """Elongated triangle should have low Jacobian."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 0.1, 0.0],
        ])
        triangles = np.array([[0, 1, 2]])

        j = compute_scaled_jacobians(vertices, triangles)
        assert j[0] < 0.1  # Very poor quality

    def test_jacobian_values_in_valid_range(self):
        """Jacobians should be between 0 and 1 for valid triangles."""
        # Random valid triangles
        np.random.seed(42)
        vertices = np.random.randn(10, 3)
        triangles = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ])

        j = compute_scaled_jacobians(vertices, triangles)
        # All should be positive (properly oriented)
        # Values typically between 0 and 1
        assert np.all(j >= -0.01)  # Allow small numerical error
        assert np.all(j <= 1.01)


class TestComputeElementSizes:
    """Tests for compute_element_sizes function."""

    def test_equilateral_triangle_incircle(self):
        """Test incircle diameter for equilateral triangle."""
        # Equilateral with side s: incircle radius = s / (2 * sqrt(3))
        # Incircle diameter = s / sqrt(3)
        s = 1.0
        h = np.sqrt(3) / 2
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [s, 0.0, 0.0],
            [s / 2, h, 0.0],
        ])
        triangles = np.array([[0, 1, 2]])

        sizes = compute_element_sizes(vertices, triangles)
        expected = s / np.sqrt(3)
        assert np.isclose(sizes[0], expected, rtol=0.01)

    def test_element_sizes_scale_with_geometry(self):
        """Test that element sizes scale correctly."""
        # Small and large triangles
        h = np.sqrt(3) / 2
        small_vertices = np.array([
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.0005, h * 0.001, 0.0],
        ])
        large_vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, h, 0.0],
        ])
        triangles = np.array([[0, 1, 2]])

        small_size = compute_element_sizes(small_vertices, triangles)[0]
        large_size = compute_element_sizes(large_vertices, triangles)[0]

        # Large should be 1000x bigger
        assert np.isclose(large_size / small_size, 1000, rtol=0.01)


class TestValidateMeshQuality:
    """Tests for validate_mesh_quality function."""

    def test_good_quality_mesh_passes(self):
        """Mesh of equilateral triangles should pass all checks."""
        # 4 equilateral triangles forming a larger triangle
        h = np.sqrt(3) / 2
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.5, h, 0.0],
            [1.5, h, 0.0],
            [1.0, 2 * h, 0.0],
        ])
        triangles = np.array([
            [0, 1, 3],
            [1, 2, 4],
            [1, 4, 3],
            [3, 4, 5],
        ])

        report = validate_mesh_quality(vertices, triangles)

        assert report.num_elements == 4
        assert report.num_vertices == 6
        assert report.aspect_ratio_max < 2.0
        assert report.jacobian_min > 0.8
        assert len(report.warnings) == 0
        assert report.is_valid

    def test_poor_quality_mesh_generates_warnings(self):
        """Mesh with bad elements should generate warnings."""
        # Mix of good and bad triangles
        h = np.sqrt(3) / 2
        vertices = np.array([
            # Good equilateral
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, h, 0.0],
            # Bad elongated
            [2.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [7.0, 0.05, 0.0],
        ])
        triangles = np.array([
            [0, 1, 2],  # Good
            [3, 4, 5],  # Bad
        ])

        report = validate_mesh_quality(
            vertices, triangles,
            max_aspect_ratio=4.0,
            min_jacobian=0.3,
        )

        assert len(report.warnings) > 0
        assert not report.is_valid

    def test_wavelength_validation_with_material(self):
        """Test wavelength validation with material properties."""
        # Create mesh with ~1mm element size
        h = np.sqrt(3) / 2
        scale = 0.001  # 1mm
        vertices = scale * np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, h, 0.0],
        ])
        triangles = np.array([[0, 1, 2]])

        titanium = ShellMaterial.titanium(thickness_m=50e-6)

        # At 20 kHz, acoustic wavelength = 343/20000 = 17.15mm
        # 1mm element gives ~17 elements per wavelength - should pass
        report = validate_mesh_quality(
            vertices, triangles,
            material=titanium,
            max_frequency_hz=20000,
            min_elements_per_wavelength=6,
        )

        assert report.elements_per_acoustic_wavelength is not None
        assert report.elements_per_acoustic_wavelength > 10
        assert report.elements_per_bending_wavelength is not None

    def test_coarse_mesh_fails_wavelength_check(self):
        """Coarse mesh should fail wavelength validation at high frequency."""
        # Create mesh with ~10mm element size
        h = np.sqrt(3) / 2
        scale = 0.01  # 10mm
        vertices = scale * np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, h, 0.0],
        ])
        triangles = np.array([[0, 1, 2]])

        titanium = ShellMaterial.titanium(thickness_m=50e-6)

        # At 20 kHz, acoustic wavelength = 17.15mm
        # 10mm element gives only ~1.7 elements per wavelength - should fail
        report = validate_mesh_quality(
            vertices, triangles,
            material=titanium,
            max_frequency_hz=20000,
            min_elements_per_wavelength=6,
        )

        assert any("acoustic wavelength" in w for w in report.warnings)


class TestFindPoorQualityElements:
    """Tests for find_poor_quality_elements function."""

    def test_identifies_high_aspect_ratio_elements(self):
        """Should correctly identify elements with high AR."""
        h = np.sqrt(3) / 2
        vertices = np.array([
            # Good
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, h, 0.0],
            # Bad (elongated)
            [2.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [7.0, 0.1, 0.0],
        ])
        triangles = np.array([
            [0, 1, 2],
            [3, 4, 5],
        ])

        high_ar, low_j = find_poor_quality_elements(
            vertices, triangles,
            max_aspect_ratio=4.0,
            min_jacobian=0.3,
        )

        assert 1 in high_ar  # Second element has high AR
        assert 0 not in high_ar  # First element is good

    def test_identifies_low_jacobian_elements(self):
        """Should correctly identify elements with low Jacobian."""
        vertices = np.array([
            # Good equilateral
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
            # Bad (very flat)
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.5, 0.01, 0.0],
        ])
        triangles = np.array([
            [0, 1, 2],
            [3, 4, 5],
        ])

        high_ar, low_j = find_poor_quality_elements(
            vertices, triangles,
            max_aspect_ratio=50.0,  # High threshold
            min_jacobian=0.3,
        )

        assert 1 in low_j  # Second element has low Jacobian


class TestMeshQualityReport:
    """Tests for MeshQualityReport class."""

    def test_summary_includes_all_metrics(self):
        """Summary should include all key metrics."""
        report = MeshQualityReport(
            num_elements=100,
            num_vertices=60,
            aspect_ratio_min=1.0,
            aspect_ratio_max=2.5,
            aspect_ratio_mean=1.3,
            aspect_ratios=np.ones(100),
            jacobian_min=0.7,
            jacobian_max=1.0,
            jacobian_mean=0.9,
            jacobians=np.ones(100),
            element_size_min=0.0005,
            element_size_max=0.0012,
            element_size_mean=0.0008,
            element_sizes=np.ones(100) * 0.001,
            elements_per_acoustic_wavelength=15.0,
            elements_per_bending_wavelength=8.0,
            warnings=[],
        )

        summary = report.summary()

        assert "Elements: 100" in summary
        assert "Vertices: 60" in summary
        assert "Aspect Ratio" in summary
        assert "Scaled Jacobian" in summary
        assert "Element Size" in summary
        assert "Wavelength Resolution" in summary
        assert "All quality checks passed" in summary

    def test_summary_shows_warnings(self):
        """Summary should display warnings when present."""
        report = MeshQualityReport(
            num_elements=10,
            num_vertices=8,
            aspect_ratio_min=1.0,
            aspect_ratio_max=8.0,
            aspect_ratio_mean=3.0,
            aspect_ratios=np.ones(10),
            jacobian_min=0.2,
            jacobian_max=1.0,
            jacobian_mean=0.6,
            jacobians=np.ones(10),
            element_size_min=0.001,
            element_size_max=0.002,
            element_size_mean=0.0015,
            element_sizes=np.ones(10) * 0.001,
            warnings=["Test warning 1", "Test warning 2"],
        )

        summary = report.summary()

        assert "WARNINGS" in summary
        assert "Test warning 1" in summary
        assert "Test warning 2" in summary
        assert "All quality checks passed" not in summary

    def test_is_valid_property(self):
        """is_valid should be True only when no warnings."""
        valid_report = MeshQualityReport(
            num_elements=1, num_vertices=3,
            aspect_ratio_min=1.0, aspect_ratio_max=1.0, aspect_ratio_mean=1.0,
            aspect_ratios=np.array([1.0]),
            jacobian_min=1.0, jacobian_max=1.0, jacobian_mean=1.0,
            jacobians=np.array([1.0]),
            element_size_min=0.001, element_size_max=0.001, element_size_mean=0.001,
            element_sizes=np.array([0.001]),
            warnings=[],
        )
        invalid_report = MeshQualityReport(
            num_elements=1, num_vertices=3,
            aspect_ratio_min=1.0, aspect_ratio_max=10.0, aspect_ratio_mean=5.0,
            aspect_ratios=np.array([10.0]),
            jacobian_min=0.1, jacobian_max=0.1, jacobian_mean=0.1,
            jacobians=np.array([0.1]),
            element_size_min=0.01, element_size_max=0.01, element_size_mean=0.01,
            element_sizes=np.array([0.01]),
            warnings=["High aspect ratio"],
        )

        assert valid_report.is_valid is True
        assert invalid_report.is_valid is False
