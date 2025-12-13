"""Tests for phase plug geometry in bempp_audio.fea."""

from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.fea.phase_plug_geometry import (
    PhasePlugGeometry,
    AnnularChannel,
    RadialSlot,
)


class TestAnnularChannel:
    """Tests for AnnularChannel dataclass."""

    def test_width_calculation(self):
        """Width should be outer - inner radius."""
        ch = AnnularChannel(
            inner_radius_m=0.005,
            outer_radius_m=0.008,
            depth_m=0.005,
        )
        assert np.isclose(ch.width_m, 0.003)

    def test_area_calculation(self):
        """Area should be pi * (r_outer^2 - r_inner^2)."""
        ch = AnnularChannel(
            inner_radius_m=0.005,
            outer_radius_m=0.010,
            depth_m=0.005,
        )
        expected = np.pi * (0.010**2 - 0.005**2)
        assert np.isclose(ch.area_m2, expected)

    def test_mean_radius(self):
        """Mean radius should be average of inner and outer."""
        ch = AnnularChannel(
            inner_radius_m=0.004,
            outer_radius_m=0.008,
            depth_m=0.005,
        )
        assert np.isclose(ch.mean_radius_m, 0.006)

    def test_exit_z(self):
        """Exit z should be entry + depth."""
        ch = AnnularChannel(
            inner_radius_m=0.005,
            outer_radius_m=0.008,
            depth_m=0.005,
            entry_z_m=0.001,
        )
        assert np.isclose(ch.exit_z_m, 0.006)


class TestRadialSlot:
    """Tests for RadialSlot dataclass."""

    def test_angular_width(self):
        """Angular width should be end - start angle."""
        slot = RadialSlot(
            start_angle_rad=0.0,
            end_angle_rad=np.pi / 4,
            inner_radius_m=0.005,
            outer_radius_m=0.015,
            depth_m=0.005,
        )
        assert np.isclose(slot.angular_width_rad, np.pi / 4)
        assert np.isclose(slot.angular_width_deg, 45.0)

    def test_mean_angle(self):
        """Mean angle should be midpoint."""
        slot = RadialSlot(
            start_angle_rad=0.0,
            end_angle_rad=np.pi / 2,
            inner_radius_m=0.005,
            outer_radius_m=0.015,
            depth_m=0.005,
        )
        assert np.isclose(slot.mean_angle_rad, np.pi / 4)

    def test_radial_length(self):
        """Radial length should be outer - inner radius."""
        slot = RadialSlot(
            start_angle_rad=0.0,
            end_angle_rad=np.pi / 4,
            inner_radius_m=0.005,
            outer_radius_m=0.015,
            depth_m=0.005,
        )
        assert np.isclose(slot.radial_length_m, 0.010)

    def test_area_calculation(self):
        """Area should be sector area formula."""
        slot = RadialSlot(
            start_angle_rad=0.0,
            end_angle_rad=np.pi / 4,
            inner_radius_m=0.005,
            outer_radius_m=0.015,
            depth_m=0.005,
        )
        # A = 0.5 * dθ * (r_outer^2 - r_inner^2)
        expected = 0.5 * (np.pi / 4) * (0.015**2 - 0.005**2)
        assert np.isclose(slot.area_m2, expected)

    def test_exit_z(self):
        """Exit z should be entry + depth."""
        slot = RadialSlot(
            start_angle_rad=0.0,
            end_angle_rad=np.pi / 4,
            inner_radius_m=0.005,
            outer_radius_m=0.015,
            depth_m=0.005,
            entry_z_m=0.001,
        )
        assert np.isclose(slot.exit_z_m, 0.006)


class TestPhasePlugGeometry:
    """Tests for PhasePlugGeometry class."""

    def test_compression_ratio(self):
        """Compression ratio should be dome_area / throat_area."""
        geo = PhasePlugGeometry(
            dome_diameter_m=0.035,
            gap_height_m=0.5e-3,
            throat_diameter_m=0.025,
            throat_z_m=0.006,
        )
        expected = (0.035 / 2) ** 2 / (0.025 / 2) ** 2
        assert np.isclose(geo.compression_ratio, expected)

    def test_is_axisymmetric_without_radial_slots(self):
        """Should be axisymmetric when no radial slots."""
        geo = PhasePlugGeometry.single_annular(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        assert geo.is_axisymmetric is True

    def test_is_not_axisymmetric_with_radial_slots(self):
        """Should not be axisymmetric when radial slots present."""
        geo = PhasePlugGeometry.radial(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        assert geo.is_axisymmetric is False


class TestPhasePlugStyles:
    """Tests for phase plug style factory methods."""

    def test_single_annular(self):
        """Single annular should have one channel."""
        geo = PhasePlugGeometry.single_annular(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        assert geo.num_channels == 1
        assert geo.num_radial_slots == 0

    def test_dual_annular(self):
        """Dual annular should have two channels."""
        geo = PhasePlugGeometry.dual_annular(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        assert geo.num_channels == 2
        assert geo.num_radial_slots == 0

    def test_triple_annular(self):
        """Triple annular should have three channels."""
        geo = PhasePlugGeometry.triple_annular(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        assert geo.num_channels == 3
        assert geo.num_radial_slots == 0
        assert geo.style == "triple_annular"

    def test_quad_annular(self):
        """Quad annular should have four channels."""
        geo = PhasePlugGeometry.quad_annular(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        assert geo.num_channels == 4
        assert geo.num_radial_slots == 0
        assert geo.style == "quad_annular"

    def test_radial(self):
        """Radial style should have radial slots, no annular channels."""
        geo = PhasePlugGeometry.radial(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
            num_slots=8,
        )
        assert geo.num_channels == 0
        assert geo.num_radial_slots == 8
        assert geo.style == "radial"
        assert geo.is_axisymmetric is False

    def test_radial_slot_distribution(self):
        """Radial slots should be evenly distributed."""
        geo = PhasePlugGeometry.radial(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
            num_slots=4,
        )
        angles = [slot.mean_angle_rad for slot in geo.radial_slots]
        expected = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        for actual, exp in zip(angles, expected):
            assert np.isclose(actual, exp, atol=0.01)

    def test_tangerine(self):
        """Tangerine style should have radial slots and central channel."""
        geo = PhasePlugGeometry.tangerine(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
            num_segments=8,
        )
        assert geo.num_channels == 1  # Central plenum
        assert geo.num_radial_slots == 8
        assert geo.style == "tangerine"

    def test_tangerine_central_channel(self):
        """Tangerine central channel should start at r=0."""
        geo = PhasePlugGeometry.tangerine(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        central = geo.channels[0]
        assert np.isclose(central.inner_radius_m, 0.0)

    def test_exponential_annular(self):
        """Exponential annular should have multiple channels."""
        geo = PhasePlugGeometry.exponential_annular(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
            num_channels=4,
        )
        assert geo.num_channels == 4
        assert geo.num_radial_slots == 0
        assert geo.style == "exponential_annular"

    def test_exponential_annular_channel_spacing(self):
        """Channels should have exponential spacing (denser toward outer edge)."""
        geo = PhasePlugGeometry.exponential_annular(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
            num_channels=4,
            taper_ratio=0.6,
        )
        # Channel centers
        centers = [ch.mean_radius_m for ch in geo.channels]
        # With taper_ratio < 1, channels concentrate toward outer edge
        # So inner spacings are larger than outer spacings
        spacings = np.diff(centers)
        assert spacings[0] > spacings[-1]


class TestFromStyle:
    """Tests for from_style factory method."""

    @pytest.mark.parametrize("style", [
        "single_annular",
        "dual_annular",
        "triple_annular",
        "quad_annular",
        "radial",
        "tangerine",
        "exponential_annular",
    ])
    def test_from_style_creates_geometry(self, style):
        """from_style should create valid geometry for all styles."""
        geo = PhasePlugGeometry.from_style(
            style=style,
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        assert isinstance(geo, PhasePlugGeometry)
        assert geo.dome_diameter_m == 0.035
        assert geo.throat_diameter_m == 0.025

    def test_from_style_invalid(self):
        """from_style should raise for unknown style."""
        with pytest.raises(ValueError, match="Unknown style"):
            PhasePlugGeometry.from_style(
                style="invalid",
                dome_diameter_m=0.035,
                throat_diameter_m=0.025,
            )


class TestSerialization:
    """Tests for to_dict/from_dict serialization."""

    def test_annular_roundtrip(self):
        """Annular style should survive serialization."""
        geo = PhasePlugGeometry.triple_annular(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        data = geo.to_dict()
        restored = PhasePlugGeometry.from_dict(data)

        assert restored.dome_diameter_m == geo.dome_diameter_m
        assert restored.throat_diameter_m == geo.throat_diameter_m
        assert restored.num_channels == geo.num_channels
        assert restored.style == geo.style

    def test_radial_roundtrip(self):
        """Radial style should survive serialization."""
        geo = PhasePlugGeometry.radial(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
            num_slots=6,
        )
        data = geo.to_dict()
        restored = PhasePlugGeometry.from_dict(data)

        assert restored.num_radial_slots == geo.num_radial_slots
        assert restored.style == geo.style

        # Check slot properties preserved
        for orig, rest in zip(geo.radial_slots, restored.radial_slots):
            assert np.isclose(orig.start_angle_rad, rest.start_angle_rad)
            assert np.isclose(orig.end_angle_rad, rest.end_angle_rad)
            assert np.isclose(orig.inner_radius_m, rest.inner_radius_m)
            assert np.isclose(orig.outer_radius_m, rest.outer_radius_m)
            assert np.isclose(orig.depth_m, rest.depth_m)

    def test_tangerine_roundtrip(self):
        """Tangerine style should survive serialization."""
        geo = PhasePlugGeometry.tangerine(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
            num_segments=8,
        )
        data = geo.to_dict()
        restored = PhasePlugGeometry.from_dict(data)

        assert restored.num_channels == geo.num_channels
        assert restored.num_radial_slots == geo.num_radial_slots
        assert restored.style == geo.style

    def test_to_dict_includes_radial_slots(self):
        """to_dict should include radial_slots key."""
        geo = PhasePlugGeometry.radial(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        data = geo.to_dict()
        assert "radial_slots" in data
        assert len(data["radial_slots"]) == geo.num_radial_slots

    def test_to_dict_includes_style(self):
        """to_dict should include style key."""
        geo = PhasePlugGeometry.tangerine(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        data = geo.to_dict()
        assert data["style"] == "tangerine"


class TestSummary:
    """Tests for summary method."""

    def test_summary_includes_style(self):
        """Summary should include style when set."""
        geo = PhasePlugGeometry.quad_annular(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        summary = geo.summary()
        assert "quad_annular" in summary

    def test_summary_shows_axisymmetric(self):
        """Summary should show axisymmetric status."""
        geo = PhasePlugGeometry.single_annular(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        summary = geo.summary()
        assert "Axisymmetric:" in summary
        assert "True" in summary

    def test_summary_shows_radial_slots(self):
        """Summary should show radial slot details."""
        geo = PhasePlugGeometry.radial(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
            num_slots=4,
        )
        summary = geo.summary()
        assert "Radial slots: 4" in summary
        assert "Slot 1:" in summary


class TestValidation:
    """Tests for geometry validation."""

    def test_validate_catches_zero_gap(self):
        """Validation should reject zero gap height."""
        geo = PhasePlugGeometry(
            dome_diameter_m=0.035,
            gap_height_m=0.0,
            throat_diameter_m=0.025,
            throat_z_m=0.006,
        )
        with pytest.raises(ValueError, match="gap_height_m must be > 0"):
            geo.validate()

    def test_validate_catches_throat_before_gap(self):
        """Validation should reject throat z <= gap height."""
        geo = PhasePlugGeometry(
            dome_diameter_m=0.035,
            gap_height_m=0.005,
            throat_diameter_m=0.025,
            throat_z_m=0.003,
        )
        with pytest.raises(ValueError, match="throat_z_m must be > gap_height_m"):
            geo.validate()

    def test_valid_geometry_passes(self):
        """Valid geometry should pass validation."""
        geo = PhasePlugGeometry.single_annular(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        geo.validate()  # Should not raise


class TestTotalChannelArea:
    """Tests for total_channel_area_m2 property."""

    def test_includes_annular_channels(self):
        """Total area should include annular channels."""
        geo = PhasePlugGeometry.dual_annular(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        expected = sum(ch.area_m2 for ch in geo.channels)
        assert np.isclose(geo.total_channel_area_m2, expected)

    def test_includes_radial_slots(self):
        """Total area should include radial slots."""
        geo = PhasePlugGeometry.radial(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        expected = sum(slot.area_m2 for slot in geo.radial_slots)
        assert np.isclose(geo.total_channel_area_m2, expected)

    def test_includes_both_when_present(self):
        """Total area should sum both annular and radial."""
        geo = PhasePlugGeometry.tangerine(
            dome_diameter_m=0.035,
            throat_diameter_m=0.025,
        )
        annular = sum(ch.area_m2 for ch in geo.channels)
        radial = sum(slot.area_m2 for slot in geo.radial_slots)
        assert np.isclose(geo.total_channel_area_m2, annular + radial)
