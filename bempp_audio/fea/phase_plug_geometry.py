"""
Parametric phase plug geometry for compression drivers.

Phase plugs transform the dome's velocity distribution into a more uniform
wavefront at the throat. They consist of:
- A solid body facing the dome
- Annular channels (slots) for sound transmission
- A throat exit leading to the waveguide

This module provides parametric geometry definitions; meshing is handled
by phase_plug_meshing.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Optional
import numpy as np


PhasePlugStyle = Literal[
    "single_annular",
    "dual_annular",
    "triple_annular",
    "quad_annular",
    "radial",
    "tangerine",
    "exponential_annular",
]


@dataclass(frozen=True)
class RadialSlot:
    """
    Specification for a radial (pie-shaped) slot.

    Radial slots are non-axisymmetric and create pie-wedge shaped channels
    extending from near the center toward the dome edge. They break up
    circumferential resonances and can provide more uniform path lengths.

    Geometry:
    - Angular extent: from start_angle to end_angle (radians)
    - Radial extent: from inner_radius to outer_radius
    - Depth in z-direction
    """

    start_angle_rad: float  # Starting angle [radians, 0 = +x axis]
    end_angle_rad: float  # Ending angle [radians]
    inner_radius_m: float  # Inner radial extent
    outer_radius_m: float  # Outer radial extent
    depth_m: float  # Axial depth of the slot
    entry_z_m: float = 0.0  # Z-coordinate of slot entry

    @property
    def angular_width_rad(self) -> float:
        """Angular width of the slot [radians]."""
        return self.end_angle_rad - self.start_angle_rad

    @property
    def angular_width_deg(self) -> float:
        """Angular width of the slot [degrees]."""
        return np.degrees(self.angular_width_rad)

    @property
    def mean_angle_rad(self) -> float:
        """Mean angle of the slot [radians]."""
        return (self.start_angle_rad + self.end_angle_rad) / 2

    @property
    def radial_length_m(self) -> float:
        """Radial length of the slot [m]."""
        return self.outer_radius_m - self.inner_radius_m

    @property
    def area_m2(self) -> float:
        """
        Approximate cross-sectional area at mean radius [m²].

        For a pie-shaped slot: A ≈ (r_outer² - r_inner²) * Δθ / 2
        """
        return 0.5 * self.angular_width_rad * (
            self.outer_radius_m**2 - self.inner_radius_m**2
        )

    @property
    def exit_z_m(self) -> float:
        """Z-coordinate of slot exit."""
        return self.entry_z_m + self.depth_m


@dataclass(frozen=True)
class AnnularChannel:
    """
    Specification for a single annular (ring-shaped) channel.

    The channel connects the dome-facing surface to the throat through
    an axisymmetric slot.

    Coordinates:
    - z=0 at dome interface (where dome meets phase plug gap)
    - z increases toward throat exit
    - r is radial distance from axis
    """

    inner_radius_m: float  # Inner edge of the annular slot
    outer_radius_m: float  # Outer edge of the annular slot
    depth_m: float  # Axial length of the channel (z-direction)
    entry_z_m: float = 0.0  # Z-coordinate of channel entry (dome side)

    @property
    def width_m(self) -> float:
        """Radial width of the channel [m]."""
        return self.outer_radius_m - self.inner_radius_m

    @property
    def area_m2(self) -> float:
        """Cross-sectional area of the channel [m²]."""
        return np.pi * (self.outer_radius_m**2 - self.inner_radius_m**2)

    @property
    def mean_radius_m(self) -> float:
        """Mean radius of the channel [m]."""
        return (self.inner_radius_m + self.outer_radius_m) / 2

    @property
    def exit_z_m(self) -> float:
        """Z-coordinate of channel exit (throat side)."""
        return self.entry_z_m + self.depth_m


@dataclass
class PhasePlugGeometry:
    """
    Parametric phase plug geometry for compression drivers.

    Defines the acoustic domain between the dome and throat:
    - Dome interface: where the dome diaphragm faces the phase plug
    - Channels: annular slots connecting dome to throat
    - Throat exit: circular opening to the waveguide

    Coordinate system:
    - Origin at center of dome interface
    - Z-axis points from dome toward throat (positive = toward listener)
    - Axisymmetric about z-axis

    The acoustic domain includes:
    1. The gap between dome and phase plug body
    2. The channel volumes
    3. Any plenum/collection volume at the throat
    """

    # Dome interface
    dome_diameter_m: float  # Diameter of dome at interface
    gap_height_m: float  # Air gap between dome apex and phase plug body

    # Channels (annular - axisymmetric)
    channels: List[AnnularChannel] = field(default_factory=list)

    # Radial slots (pie-shaped - non-axisymmetric)
    radial_slots: List[RadialSlot] = field(default_factory=list)

    # Throat
    throat_diameter_m: float = 0.025  # Throat exit diameter
    throat_z_m: float = 0.010  # Z-position of throat exit

    # Phase plug body parameters (for solid geometry, not acoustic domain)
    body_outer_diameter_m: Optional[float] = None  # Outer diameter of plug body

    # Style identifier for reference
    style: Optional[PhasePlugStyle] = None

    @property
    def dome_radius_m(self) -> float:
        """Dome interface radius [m]."""
        return self.dome_diameter_m / 2

    @property
    def throat_radius_m(self) -> float:
        """Throat exit radius [m]."""
        return self.throat_diameter_m / 2

    @property
    def throat_area_m2(self) -> float:
        """Throat exit area [m²]."""
        return np.pi * self.throat_radius_m**2

    @property
    def dome_area_m2(self) -> float:
        """Dome interface area [m²]."""
        return np.pi * self.dome_radius_m**2

    @property
    def compression_ratio(self) -> float:
        """
        Compression ratio = dome area / throat area.

        Typical values: 4:1 to 10:1 for compression drivers.
        """
        return self.dome_area_m2 / self.throat_area_m2

    @property
    def total_channel_area_m2(self) -> float:
        """Total cross-sectional area of all channels (annular + radial) [m²]."""
        annular_area = sum(ch.area_m2 for ch in self.channels)
        radial_area = sum(slot.area_m2 for slot in self.radial_slots)
        return annular_area + radial_area

    @property
    def num_channels(self) -> int:
        """Number of annular channels."""
        return len(self.channels)

    @property
    def num_radial_slots(self) -> int:
        """Number of radial slots."""
        return len(self.radial_slots)

    @property
    def is_axisymmetric(self) -> bool:
        """True if geometry is axisymmetric (no radial slots)."""
        return len(self.radial_slots) == 0

    def channel_velocity_ratio(self, dome_velocity: float = 1.0) -> float:
        """
        Estimate mean channel velocity from continuity.

        v_channel = v_dome * A_dome / A_channels

        Parameters
        ----------
        dome_velocity : float
            Mean dome surface velocity [m/s]

        Returns
        -------
        float
            Estimated channel velocity [m/s]
        """
        if self.total_channel_area_m2 == 0:
            return float('inf')
        return dome_velocity * self.dome_area_m2 / self.total_channel_area_m2

    def validate(self) -> None:
        """
        Basic geometric sanity checks.

        Raises ValueError if parameters are inconsistent enough to likely produce
        disconnected or degenerate meshes.
        """
        if self.gap_height_m <= 0:
            raise ValueError("gap_height_m must be > 0")
        if self.throat_diameter_m <= 0:
            raise ValueError("throat_diameter_m must be > 0")
        if self.throat_z_m <= self.gap_height_m:
            raise ValueError("throat_z_m must be > gap_height_m")
        if self.dome_diameter_m <= self.throat_diameter_m:
            raise ValueError("dome_diameter_m should exceed throat_diameter_m for a phase plug")
        for ch in self.channels:
            if ch.inner_radius_m < 0 or ch.outer_radius_m <= 0:
                raise ValueError("channel radii must be positive")
            if ch.outer_radius_m <= ch.inner_radius_m:
                raise ValueError("channel outer_radius_m must be > inner_radius_m")
            if ch.entry_z_m < 0:
                raise ValueError("channel entry_z_m must be >= 0")
            if ch.depth_m <= 0:
                raise ValueError("channel depth_m must be > 0")

    @classmethod
    def single_annular(
        cls,
        dome_diameter_m: float,
        throat_diameter_m: float,
        gap_height_m: float = 0.5e-3,
        channel_width_m: float = 1.0e-3,
        channel_depth_m: float = 5.0e-3,
    ) -> "PhasePlugGeometry":
        """
        Create a single-annular-slot phase plug.

        The simplest phase plug design: one ring-shaped channel
        at the mean radius between dome edge and throat.

        Parameters
        ----------
        dome_diameter_m : float
            Dome diameter at interface [m]
        throat_diameter_m : float
            Throat exit diameter [m]
        gap_height_m : float
            Air gap between dome and phase plug [m]
        channel_width_m : float
            Radial width of the annular slot [m]
        channel_depth_m : float
            Axial depth of the channel [m]

        Returns
        -------
        PhasePlugGeometry
        """
        dome_r = dome_diameter_m / 2
        throat_r = throat_diameter_m / 2

        # Place channel at mean radius
        mean_r = (dome_r + throat_r) / 2
        inner_r = mean_r - channel_width_m / 2
        outer_r = mean_r + channel_width_m / 2

        channel = AnnularChannel(
            inner_radius_m=inner_r,
            outer_radius_m=outer_r,
            depth_m=channel_depth_m,
            entry_z_m=gap_height_m,
        )

        return cls(
            dome_diameter_m=dome_diameter_m,
            gap_height_m=gap_height_m,
            channels=[channel],
            throat_diameter_m=throat_diameter_m,
            throat_z_m=gap_height_m + channel_depth_m,
        )

    @classmethod
    def dual_annular(
        cls,
        dome_diameter_m: float,
        throat_diameter_m: float,
        gap_height_m: float = 0.5e-3,
        channel_width_m: float = 0.8e-3,
        channel_depth_m: float = 5.0e-3,
        radial_spacing_m: float = 2.0e-3,
    ) -> "PhasePlugGeometry":
        """
        Create a dual-annular-slot phase plug.

        Two concentric ring-shaped channels provide better HF response
        by reducing the path length variation across the dome.

        Parameters
        ----------
        dome_diameter_m : float
            Dome diameter at interface [m]
        throat_diameter_m : float
            Throat exit diameter [m]
        gap_height_m : float
            Air gap between dome and phase plug [m]
        channel_width_m : float
            Radial width of each slot [m]
        channel_depth_m : float
            Axial depth of channels [m]
        radial_spacing_m : float
            Radial separation between channel centers [m]

        Returns
        -------
        PhasePlugGeometry
        """
        dome_r = dome_diameter_m / 2
        throat_r = throat_diameter_m / 2

        # Place channels symmetrically about mean radius
        mean_r = (dome_r + throat_r) / 2

        # Inner channel
        inner_center = mean_r - radial_spacing_m / 2
        inner_ch = AnnularChannel(
            inner_radius_m=inner_center - channel_width_m / 2,
            outer_radius_m=inner_center + channel_width_m / 2,
            depth_m=channel_depth_m,
            entry_z_m=gap_height_m,
        )

        # Outer channel
        outer_center = mean_r + radial_spacing_m / 2
        outer_ch = AnnularChannel(
            inner_radius_m=outer_center - channel_width_m / 2,
            outer_radius_m=outer_center + channel_width_m / 2,
            depth_m=channel_depth_m,
            entry_z_m=gap_height_m,
        )

        return cls(
            dome_diameter_m=dome_diameter_m,
            gap_height_m=gap_height_m,
            channels=[inner_ch, outer_ch],
            throat_diameter_m=throat_diameter_m,
            throat_z_m=gap_height_m + channel_depth_m,
        )

    @classmethod
    def triple_annular(
        cls,
        dome_diameter_m: float,
        throat_diameter_m: float,
        gap_height_m: float = 0.5e-3,
        channel_width_m: float = 0.6e-3,
        channel_depth_m: float = 5.0e-3,
    ) -> "PhasePlugGeometry":
        """
        Create a triple-annular-slot phase plug.

        Three channels distribute flow more evenly across the dome radius.

        Parameters
        ----------
        dome_diameter_m : float
            Dome diameter at interface [m]
        throat_diameter_m : float
            Throat exit diameter [m]
        gap_height_m : float
            Air gap [m]
        channel_width_m : float
            Radial width of each slot [m]
        channel_depth_m : float
            Axial depth [m]

        Returns
        -------
        PhasePlugGeometry
        """
        dome_r = dome_diameter_m / 2
        throat_r = throat_diameter_m / 2

        # Distribute channels evenly from throat to dome radius
        # Leave some margin at edges
        margin = channel_width_m
        available_range = dome_r - throat_r - 2 * margin

        channels = []
        for i in range(3):
            # Position: 0.25, 0.5, 0.75 of available range
            frac = (i + 1) / 4
            center_r = throat_r + margin + frac * available_range
            ch = AnnularChannel(
                inner_radius_m=center_r - channel_width_m / 2,
                outer_radius_m=center_r + channel_width_m / 2,
                depth_m=channel_depth_m,
                entry_z_m=gap_height_m,
            )
            channels.append(ch)

        return cls(
            dome_diameter_m=dome_diameter_m,
            gap_height_m=gap_height_m,
            channels=channels,
            throat_diameter_m=throat_diameter_m,
            throat_z_m=gap_height_m + channel_depth_m,
            style="triple_annular",
        )

    @classmethod
    def quad_annular(
        cls,
        dome_diameter_m: float,
        throat_diameter_m: float,
        gap_height_m: float = 0.5e-3,
        channel_width_m: float = 0.5e-3,
        channel_depth_m: float = 5.0e-3,
    ) -> "PhasePlugGeometry":
        """
        Create a quad-annular-slot phase plug.

        Four concentric channels provide excellent wavefront coherence
        and extended high-frequency response. Common in high-end 2" drivers.

        Parameters
        ----------
        dome_diameter_m : float
            Dome diameter at interface [m]
        throat_diameter_m : float
            Throat exit diameter [m]
        gap_height_m : float
            Air gap [m]
        channel_width_m : float
            Radial width of each slot [m]
        channel_depth_m : float
            Axial depth [m]

        Returns
        -------
        PhasePlugGeometry
        """
        dome_r = dome_diameter_m / 2
        throat_r = throat_diameter_m / 2

        # Distribute 4 channels evenly from throat to dome radius
        margin = channel_width_m
        available_range = dome_r - throat_r - 2 * margin

        channels = []
        for i in range(4):
            # Position: 0.2, 0.4, 0.6, 0.8 of available range
            frac = (i + 1) / 5
            center_r = throat_r + margin + frac * available_range
            ch = AnnularChannel(
                inner_radius_m=center_r - channel_width_m / 2,
                outer_radius_m=center_r + channel_width_m / 2,
                depth_m=channel_depth_m,
                entry_z_m=gap_height_m,
            )
            channels.append(ch)

        return cls(
            dome_diameter_m=dome_diameter_m,
            gap_height_m=gap_height_m,
            channels=channels,
            throat_diameter_m=throat_diameter_m,
            throat_z_m=gap_height_m + channel_depth_m,
            style="quad_annular",
        )

    @classmethod
    def radial(
        cls,
        dome_diameter_m: float,
        throat_diameter_m: float,
        num_slots: int = 8,
        gap_height_m: float = 0.5e-3,
        slot_angular_width_deg: float = 20.0,
        slot_depth_m: float = 5.0e-3,
        inner_radius_fraction: float = 0.3,
    ) -> "PhasePlugGeometry":
        """
        Create a radial (pie-shaped) slot phase plug.

        Radial slots extend outward from near the center like spokes of a wheel.
        This design:
        - Breaks up circumferential resonances
        - Provides more uniform path lengths across dome radius
        - Common in modern high-frequency compression drivers

        The slots are evenly distributed around the circumference.

        Parameters
        ----------
        dome_diameter_m : float
            Dome diameter at interface [m]
        throat_diameter_m : float
            Throat exit diameter [m]
        num_slots : int
            Number of radial slots (typically 4, 6, 8, or 12)
        gap_height_m : float
            Air gap between dome and phase plug [m]
        slot_angular_width_deg : float
            Angular width of each slot [degrees]
        slot_depth_m : float
            Axial depth of the slots [m]
        inner_radius_fraction : float
            Inner radius as fraction of throat radius (0-1)

        Returns
        -------
        PhasePlugGeometry

        Notes
        -----
        The phase plug body between slots acts as the solid "fingers" that
        define the channels. The number of slots affects:
        - HF response: more slots = smoother HF
        - LF loading: more slots = less compression effect
        - Manufacturing: more slots = more complex tooling
        """
        dome_r = dome_diameter_m / 2
        throat_r = throat_diameter_m / 2

        # Slot geometry
        slot_width_rad = np.radians(slot_angular_width_deg)
        angular_spacing = 2 * np.pi / num_slots

        # Inner and outer radius of slots
        inner_r = throat_r * inner_radius_fraction
        outer_r = dome_r * 0.95  # Leave small margin at dome edge

        radial_slots = []
        for i in range(num_slots):
            center_angle = i * angular_spacing
            slot = RadialSlot(
                start_angle_rad=center_angle - slot_width_rad / 2,
                end_angle_rad=center_angle + slot_width_rad / 2,
                inner_radius_m=inner_r,
                outer_radius_m=outer_r,
                depth_m=slot_depth_m,
                entry_z_m=gap_height_m,
            )
            radial_slots.append(slot)

        return cls(
            dome_diameter_m=dome_diameter_m,
            gap_height_m=gap_height_m,
            channels=[],  # No annular channels
            radial_slots=radial_slots,
            throat_diameter_m=throat_diameter_m,
            throat_z_m=gap_height_m + slot_depth_m,
            style="radial",
        )

    @classmethod
    def tangerine(
        cls,
        dome_diameter_m: float,
        throat_diameter_m: float,
        num_segments: int = 8,
        gap_height_m: float = 0.5e-3,
        slot_angular_width_deg: float = 25.0,
        slot_depth_m: float = 5.0e-3,
        center_hole_diameter_m: Optional[float] = None,
    ) -> "PhasePlugGeometry":
        """
        Create a tangerine-style phase plug.

        Named for its resemblance to a sectioned citrus fruit, this design
        combines radial slots with a central collection area. Popular in
        professional compression drivers (e.g., JBL, TAD).

        Features:
        - Radial slots for path length equalization
        - Central collection plenum (like tangerine center)
        - Excellent HF extension and smooth response

        Parameters
        ----------
        dome_diameter_m : float
            Dome diameter at interface [m]
        throat_diameter_m : float
            Throat exit diameter [m]
        num_segments : int
            Number of tangerine segments (radial slots)
        gap_height_m : float
            Air gap between dome and phase plug [m]
        slot_angular_width_deg : float
            Angular width of each slot [degrees]
        slot_depth_m : float
            Axial depth of the slots [m]
        center_hole_diameter_m : float, optional
            Diameter of central hole. If None, uses throat_diameter_m * 0.6

        Returns
        -------
        PhasePlugGeometry

        Notes
        -----
        The tangerine design provides excellent phase coherence because:
        1. Sound from different dome radii travels similar path lengths
        2. The central plenum provides smooth transition to throat
        3. Radial symmetry minimizes directivity artifacts

        Common configurations:
        - 6 segments: Simple, good for 1" drivers
        - 8 segments: Standard for 1.4" drivers
        - 12 segments: High-end 2" drivers, maximum HF extension
        """
        dome_r = dome_diameter_m / 2
        throat_r = throat_diameter_m / 2

        if center_hole_diameter_m is None:
            center_hole_diameter_m = throat_diameter_m * 0.6
        center_r = center_hole_diameter_m / 2

        # Slot geometry
        slot_width_rad = np.radians(slot_angular_width_deg)
        angular_spacing = 2 * np.pi / num_segments

        # Slots extend from center hole to near dome edge
        inner_r = center_r
        outer_r = dome_r * 0.92

        radial_slots = []
        for i in range(num_segments):
            center_angle = i * angular_spacing
            slot = RadialSlot(
                start_angle_rad=center_angle - slot_width_rad / 2,
                end_angle_rad=center_angle + slot_width_rad / 2,
                inner_radius_m=inner_r,
                outer_radius_m=outer_r,
                depth_m=slot_depth_m,
                entry_z_m=gap_height_m,
            )
            radial_slots.append(slot)

        # Add a central annular channel representing the collection plenum
        center_channel = AnnularChannel(
            inner_radius_m=0,
            outer_radius_m=center_r,
            depth_m=slot_depth_m * 0.8,  # Slightly shallower
            entry_z_m=gap_height_m,
        )

        return cls(
            dome_diameter_m=dome_diameter_m,
            gap_height_m=gap_height_m,
            channels=[center_channel],
            radial_slots=radial_slots,
            throat_diameter_m=throat_diameter_m,
            throat_z_m=gap_height_m + slot_depth_m,
            style="tangerine",
        )

    @classmethod
    def exponential_annular(
        cls,
        dome_diameter_m: float,
        throat_diameter_m: float,
        num_channels: int = 4,
        gap_height_m: float = 0.5e-3,
        channel_depth_m: float = 6.0e-3,
        taper_ratio: float = 0.7,
    ) -> "PhasePlugGeometry":
        """
        Create an exponential-taper annular phase plug.

        The channels are positioned with exponentially-spaced radii,
        providing progressively finer sampling toward the dome center
        where velocity gradients are typically steeper.

        This design is optimized for:
        - Smooth transition from piston-like to modal behavior
        - Better handling of first breakup mode
        - Extended HF response

        Parameters
        ----------
        dome_diameter_m : float
            Dome diameter at interface [m]
        throat_diameter_m : float
            Throat exit diameter [m]
        num_channels : int
            Number of annular channels (3-6 typical)
        gap_height_m : float
            Air gap between dome and phase plug [m]
        channel_depth_m : float
            Axial depth of channels [m]
        taper_ratio : float
            Exponential taper ratio (0.5-0.9). Lower = more channels near center.

        Returns
        -------
        PhasePlugGeometry

        Notes
        -----
        The exponential spacing follows:
            r_i = r_throat + (r_dome - r_throat) * (1 - taper_ratio^i) / (1 - taper_ratio^N)

        This places more channels where the dome velocity varies most rapidly,
        improving wavefront coherence at breakup frequencies.
        """
        dome_r = dome_diameter_m / 2
        throat_r = throat_diameter_m / 2

        # Calculate exponentially-spaced radii
        if abs(taper_ratio - 1.0) < 1e-6:
            # Linear spacing fallback
            positions = np.linspace(0, 1, num_channels + 2)[1:-1]
        else:
            # Exponential spacing
            i_vals = np.arange(1, num_channels + 1)
            positions = (1 - taper_ratio**i_vals) / (1 - taper_ratio**(num_channels + 1))

        # Convert to radii
        available_range = dome_r - throat_r
        margin = available_range * 0.1
        usable_range = available_range - 2 * margin

        channels = []
        # Channel width decreases toward center (matches exponential spacing)
        base_width = usable_range / (num_channels * 2)

        for i, pos in enumerate(positions):
            center_r = throat_r + margin + pos * usable_range
            # Width proportional to local spacing
            width = base_width * (1 + 0.5 * pos)  # Wider channels toward edge

            ch = AnnularChannel(
                inner_radius_m=max(throat_r * 0.5, center_r - width / 2),
                outer_radius_m=min(dome_r * 0.98, center_r + width / 2),
                depth_m=channel_depth_m,
                entry_z_m=gap_height_m,
            )
            channels.append(ch)

        return cls(
            dome_diameter_m=dome_diameter_m,
            gap_height_m=gap_height_m,
            channels=channels,
            throat_diameter_m=throat_diameter_m,
            throat_z_m=gap_height_m + channel_depth_m,
            style="exponential_annular",
        )

    @classmethod
    def from_style(
        cls,
        style: PhasePlugStyle,
        dome_diameter_m: float,
        throat_diameter_m: float,
        **kwargs,
    ) -> "PhasePlugGeometry":
        """
        Create a phase plug from a named style.

        Parameters
        ----------
        style : PhasePlugStyle
            One of: "single_annular", "dual_annular", "triple_annular",
            "radial", "tangerine", "exponential_annular"
        dome_diameter_m : float
            Dome diameter [m]
        throat_diameter_m : float
            Throat diameter [m]
        **kwargs
            Additional parameters passed to the specific factory method

        Returns
        -------
        PhasePlugGeometry
        """
        factories = {
            "single_annular": cls.single_annular,
            "dual_annular": cls.dual_annular,
            "triple_annular": cls.triple_annular,
            "quad_annular": cls.quad_annular,
            "radial": cls.radial,
            "tangerine": cls.tangerine,
            "exponential_annular": cls.exponential_annular,
        }

        if style not in factories:
            raise ValueError(
                f"Unknown style '{style}'. Valid options: {list(factories.keys())}"
            )

        return factories[style](dome_diameter_m, throat_diameter_m, **kwargs)

    def get_boundary_markers(self) -> dict:
        """
        Return standard boundary marker assignments.

        Used for applying boundary conditions in HelmholtzFEMSolver.

        Returns
        -------
        dict
            Mapping of boundary names to marker integers
        """
        return {
            "dome_interface": 1,  # Velocity BC from dome motion
            "throat_exit": 2,  # Radiation/impedance BC
            "hard_wall": 3,  # Phase plug body surfaces (∂p/∂n = 0)
            "channel_walls": 4,  # Channel side walls (hard)
        }

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "dome_diameter_m": self.dome_diameter_m,
            "gap_height_m": self.gap_height_m,
            "throat_diameter_m": self.throat_diameter_m,
            "throat_z_m": self.throat_z_m,
            "body_outer_diameter_m": self.body_outer_diameter_m,
            "style": self.style,
            "channels": [
                {
                    "inner_radius_m": ch.inner_radius_m,
                    "outer_radius_m": ch.outer_radius_m,
                    "depth_m": ch.depth_m,
                    "entry_z_m": ch.entry_z_m,
                }
                for ch in self.channels
            ],
            "radial_slots": [
                {
                    "start_angle_rad": slot.start_angle_rad,
                    "end_angle_rad": slot.end_angle_rad,
                    "inner_radius_m": slot.inner_radius_m,
                    "outer_radius_m": slot.outer_radius_m,
                    "depth_m": slot.depth_m,
                    "entry_z_m": slot.entry_z_m,
                }
                for slot in self.radial_slots
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhasePlugGeometry":
        """Deserialize from dictionary."""
        channels = [
            AnnularChannel(**ch_data) for ch_data in data.get("channels", [])
        ]
        radial_slots = [
            RadialSlot(**slot_data) for slot_data in data.get("radial_slots", [])
        ]
        return cls(
            dome_diameter_m=data["dome_diameter_m"],
            gap_height_m=data["gap_height_m"],
            throat_diameter_m=data.get("throat_diameter_m", 0.025),
            throat_z_m=data.get("throat_z_m", 0.010),
            body_outer_diameter_m=data.get("body_outer_diameter_m"),
            channels=channels,
            radial_slots=radial_slots,
            style=data.get("style"),
        )

    def summary(self) -> str:
        """Return a text summary of the phase plug geometry."""
        lines = [
            "Phase Plug Geometry Summary",
            "=" * 40,
        ]

        if self.style:
            lines.append(f"Style:              {self.style}")

        lines.extend([
            f"Dome diameter:      {self.dome_diameter_m*1000:.2f} mm",
            f"Throat diameter:    {self.throat_diameter_m*1000:.2f} mm",
            f"Compression ratio:  {self.compression_ratio:.1f}:1",
            f"Gap height:         {self.gap_height_m*1000:.3f} mm",
            f"Axisymmetric:       {self.is_axisymmetric}",
        ])

        if self.num_channels > 0:
            lines.extend([
                "",
                f"Annular channels: {self.num_channels}",
            ])
            for i, ch in enumerate(self.channels):
                lines.extend([
                    f"  Channel {i+1}:",
                    f"    Radii: {ch.inner_radius_m*1000:.2f} - {ch.outer_radius_m*1000:.2f} mm",
                    f"    Width: {ch.width_m*1000:.3f} mm",
                    f"    Depth: {ch.depth_m*1000:.2f} mm",
                    f"    Area:  {ch.area_m2*1e6:.2f} mm²",
                ])

        if self.num_radial_slots > 0:
            lines.extend([
                "",
                f"Radial slots: {self.num_radial_slots}",
            ])
            for i, slot in enumerate(self.radial_slots):
                lines.extend([
                    f"  Slot {i+1}:",
                    f"    Angle:  {np.degrees(slot.start_angle_rad):.1f}° - {np.degrees(slot.end_angle_rad):.1f}°",
                    f"    Width:  {slot.angular_width_deg:.1f}°",
                    f"    Radii:  {slot.inner_radius_m*1000:.2f} - {slot.outer_radius_m*1000:.2f} mm",
                    f"    Depth:  {slot.depth_m*1000:.2f} mm",
                    f"    Area:   {slot.area_m2*1e6:.2f} mm²",
                ])

        lines.extend([
            "",
            f"Total channel area: {self.total_channel_area_m2*1e6:.2f} mm²",
            f"Throat area:        {self.throat_area_m2*1e6:.2f} mm²",
        ])

        return "\n".join(lines)
