"""Tests for element quality metrics in bempp_audio.mesh.validation."""

from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.mesh.validation import ElementQualityMetrics


class TestComputeTriangleAspectRatios:
    """Tests for compute_triangle_aspect_ratios."""

    def test_equilateral_triangle_aspect_ratio_one(self):
        """An equilateral triangle should have aspect ratio of 1."""
        # Equilateral triangle with side length 1
        vertices = np.array([
            [0, 1, 0.5],
            [0, 0, np.sqrt(3) / 2],
            [0, 0, 0],
        ])
        elements = np.array([[0], [1], [2]])

        ratios = ElementQualityMetrics.compute_triangle_aspect_ratios(vertices, elements)

        assert len(ratios) == 1
        assert np.isclose(ratios[0], 1.0, atol=0.01)

    def test_elongated_triangle_high_aspect_ratio(self):
        """An elongated triangle should have high aspect ratio."""
        # Very elongated triangle
        vertices = np.array([
            [0, 1, 0.01],  # x
            [0, 0, 0],     # y
            [0, 0, 0],     # z
        ])
        elements = np.array([[0], [1], [2]])

        ratios = ElementQualityMetrics.compute_triangle_aspect_ratios(vertices, elements)

        # Longest edge ~1, shortest ~0.01, ratio ~100
        assert ratios[0] > 50

    def test_multiple_triangles(self):
        """Should handle multiple triangles."""
        vertices = np.array([
            [0, 1, 0.5, 2],  # x
            [0, 0, np.sqrt(3) / 2, 0],  # y
            [0, 0, 0, 0],  # z
        ])
        elements = np.array([[0, 1], [1, 2], [2, 3]])

        ratios = ElementQualityMetrics.compute_triangle_aspect_ratios(vertices, elements)

        assert len(ratios) == 2


class TestComputeTriangleAreas:
    """Tests for compute_triangle_areas."""

    def test_unit_right_triangle(self):
        """A right triangle with legs 1,1 should have area 0.5."""
        vertices = np.array([
            [0, 1, 0],  # x
            [0, 0, 1],  # y
            [0, 0, 0],  # z
        ])
        elements = np.array([[0], [1], [2]])

        areas = ElementQualityMetrics.compute_triangle_areas(vertices, elements)

        assert len(areas) == 1
        assert np.isclose(areas[0], 0.5)

    def test_equilateral_triangle_area(self):
        """An equilateral triangle with side 1 should have area sqrt(3)/4."""
        vertices = np.array([
            [0, 1, 0.5],
            [0, 0, np.sqrt(3) / 2],
            [0, 0, 0],
        ])
        elements = np.array([[0], [1], [2]])

        areas = ElementQualityMetrics.compute_triangle_areas(vertices, elements)

        expected_area = np.sqrt(3) / 4
        assert np.isclose(areas[0], expected_area, atol=1e-10)


class TestQualityReport:
    """Tests for quality_report."""

    def test_report_contains_expected_keys(self):
        """Report should contain all expected metrics."""
        vertices = np.array([
            [0, 1, 0.5],
            [0, 0, np.sqrt(3) / 2],
            [0, 0, 0],
        ])
        elements = np.array([[0], [1], [2]])

        report = ElementQualityMetrics.quality_report(vertices, elements, verbose=False)

        expected_keys = [
            "n_elements",
            "aspect_ratio_min",
            "aspect_ratio_max",
            "aspect_ratio_mean",
            "aspect_ratio_std",
            "area_min_mm2",
            "area_max_mm2",
            "area_mean_mm2",
            "area_std_mm2",
            "n_poor_quality",
            "n_very_poor",
        ]
        for key in expected_keys:
            assert key in report

    def test_single_equilateral_no_poor_quality(self):
        """An equilateral triangle should have no poor quality elements."""
        vertices = np.array([
            [0, 1, 0.5],
            [0, 0, np.sqrt(3) / 2],
            [0, 0, 0],
        ])
        elements = np.array([[0], [1], [2]])

        report = ElementQualityMetrics.quality_report(vertices, elements, verbose=False)

        assert report["n_elements"] == 1
        assert report["n_poor_quality"] == 0
        assert report["n_very_poor"] == 0


class TestValidateMeshQuality:
    """Tests for validate_mesh_quality."""

    def test_good_mesh_passes(self):
        """A mesh with good aspect ratios should pass validation."""
        vertices = np.array([
            [0, 1, 0.5],
            [0, 0, np.sqrt(3) / 2],
            [0, 0, 0],
        ])
        elements = np.array([[0], [1], [2]])

        is_valid, report = ElementQualityMetrics.validate_mesh_quality(
            vertices, elements, max_aspect_ratio=10.0, verbose=False
        )

        assert is_valid is True

    def test_elongated_mesh_fails(self):
        """A mesh with elongated elements should fail validation."""
        vertices = np.array([
            [0, 1, 0.001],  # Very elongated
            [0, 0, 0],
            [0, 0, 0],
        ])
        elements = np.array([[0], [1], [2]])

        is_valid, report = ElementQualityMetrics.validate_mesh_quality(
            vertices, elements, max_aspect_ratio=10.0, verbose=False
        )

        assert is_valid is False
        assert report["aspect_ratio_max"] > 10.0
