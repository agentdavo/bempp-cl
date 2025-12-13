"""
Parametric dome geometry generation for compression driver diaphragms.

Supports common dome profiles:
- Spherical (constant curvature)
- Elliptical (oblate ellipsoid section)
- Parabolic
- Conical (for comparison/hybrid profiles)

Also supports radial corrugations for stiffness tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, List, Tuple
import numpy as np


DomeProfile = Literal["spherical", "elliptical", "parabolic", "conical"]


@dataclass(frozen=True)
class CorrugationSpec:
    """
    Specification for radial corrugations in a dome diaphragm.

    Corrugations add local stiffness and can push breakup modes higher.
    """

    num_corrugations: int
    depth_m: float
    width_m: float
    inner_radius_m: float  # Starting radius from center
    outer_radius_m: float  # Ending radius


@dataclass
class DomeGeometry:
    """
    Parametric dome geometry for compression driver diaphragms.

    The dome is defined by its base diameter, height (apex to base),
    and profile type. Optionally, corrugations can be added.

    Coordinates:
    - Origin at dome apex (highest point)
    - Z-axis points downward (toward base)
    - Base is at z = dome_height_m
    """

    base_diameter_m: float
    dome_height_m: float
    profile: DomeProfile = "spherical"
    corrugations: Optional[CorrugationSpec] = None

    # Derived parameters (computed on first access)
    _sphere_radius: Optional[float] = field(default=None, repr=False, compare=False)
    _center_z: Optional[float] = field(default=None, repr=False, compare=False)

    @property
    def base_radius_m(self) -> float:
        """Base radius [m]."""
        return self.base_diameter_m / 2

    @property
    def sphere_radius_m(self) -> float:
        """
        Radius of curvature for spherical profile [m].

        For a spherical cap: R = (r² + h²) / (2h)
        where r is base radius and h is dome height.
        """
        if self._sphere_radius is None:
            r = self.base_radius_m
            h = self.dome_height_m
            object.__setattr__(self, "_sphere_radius", (r**2 + h**2) / (2 * h))
        return self._sphere_radius

    @property
    def sphere_center_z_m(self) -> float:
        """
        Z-coordinate of sphere center [m].

        For a dome with apex at z=0 (r=0) and base at z=h (r=r_base),
        the sphere center is at z=R where R is the radius of curvature.

        The dome surface is the portion of the sphere from z=0 to z=h.
        """
        if self._center_z is None:
            # Center is at z=R so that the sphere passes through apex at (r=0, z=0)
            # Verification: at apex, r² + (0-R)² = 0 + R² = R² ✓
            object.__setattr__(self, "_center_z", self.sphere_radius_m)
        return self._center_z

    def radius_at_z(self, z: float) -> float:
        """
        Compute radial distance from axis at given z-coordinate.

        Parameters
        ----------
        z : float
            Axial coordinate (0 at apex, positive toward base)

        Returns
        -------
        float
            Radial distance from axis [m]
        """
        if self.profile == "spherical":
            # Sphere centered at (0, 0, R): r² + (z - R)² = R²
            # At z=0: r² = R² - R² = 0 (apex) ✓
            # At z=h: r² = R² - (h-R)² = 2Rh - h² (base radius²) ✓
            R = self.sphere_radius_m
            r_sq = R**2 - (z - R)**2
            return np.sqrt(max(0.0, r_sq))

        elif self.profile == "elliptical":
            # Ellipsoid: (r/a)² + (z/b)² = 1
            # where a = base_radius, b = dome_height
            a = self.base_radius_m
            b = self.dome_height_m
            # Normalized coordinate: z goes from 0 (apex) to b (base)
            # At apex (z=0): r = 0
            # At base (z=b): r = a
            z_norm = z / b
            if z_norm > 1.0:
                return 0.0
            return a * np.sqrt(max(0.0, 1 - (1 - z_norm) ** 2))

        elif self.profile == "parabolic":
            # Parabola: r² = 4f * z where f = r_base² / (4 * h)
            # At z=h: r = r_base
            f = self.base_radius_m**2 / (4 * self.dome_height_m)
            return np.sqrt(max(0.0, 4 * f * z))

        elif self.profile == "conical":
            # Linear: r = r_base * (z / h)
            return self.base_radius_m * z / self.dome_height_m

        else:
            raise ValueError(f"Unknown profile: {self.profile}")

    def z_at_radius(self, r: float) -> float:
        """
        Compute z-coordinate at given radial distance.

        Inverse of radius_at_z().

        Parameters
        ----------
        r : float
            Radial distance from axis [m]

        Returns
        -------
        float
            Axial coordinate (0 at apex, positive toward base) [m]
        """
        if r <= 0:
            return 0.0
        if r >= self.base_radius_m:
            return self.dome_height_m

        if self.profile == "spherical":
            # z = center_z - sqrt(R² - r²)
            r_sq = self.sphere_radius_m**2 - r**2
            return self.sphere_center_z_m - np.sqrt(max(0.0, r_sq))

        elif self.profile == "elliptical":
            # From (r/a)² + ((b-z)/b)² = 1
            # (b-z)/b = sqrt(1 - (r/a)²)
            # z = b * (1 - sqrt(1 - (r/a)²))
            a = self.base_radius_m
            b = self.dome_height_m
            return b * (1 - np.sqrt(max(0.0, 1 - (r / a) ** 2)))

        elif self.profile == "parabolic":
            f = self.base_radius_m**2 / (4 * self.dome_height_m)
            return r**2 / (4 * f)

        elif self.profile == "conical":
            return self.dome_height_m * r / self.base_radius_m

        else:
            raise ValueError(f"Unknown profile: {self.profile}")

    def generate_profile_points(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate profile curve as (r, z) arrays.

        Parameters
        ----------
        num_points : int
            Number of points along profile

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (r_array, z_array) in meters
        """
        z = np.linspace(0, self.dome_height_m, num_points)
        r = np.array([self.radius_at_z(zi) for zi in z])
        return r, z

    def surface_area_m2(self, num_points: int = 200) -> float:
        """
        Estimate surface area by numerical integration.

        Uses revolution surface formula: A = 2π ∫ r * sqrt(1 + (dr/dz)²) dz
        """
        r, z = self.generate_profile_points(num_points)
        dr_dz = np.gradient(r, z)
        integrand = r * np.sqrt(1 + dr_dz**2)
        area = 2 * np.pi * np.trapz(integrand, z)
        return area

    @classmethod
    def spherical(
        cls,
        base_diameter_m: float,
        dome_height_m: float,
        corrugations: Optional[CorrugationSpec] = None,
    ) -> "DomeGeometry":
        """Create a spherical dome (constant curvature)."""
        return cls(
            base_diameter_m=base_diameter_m,
            dome_height_m=dome_height_m,
            profile="spherical",
            corrugations=corrugations,
        )

    @classmethod
    def elliptical(
        cls,
        base_diameter_m: float,
        dome_height_m: float,
        corrugations: Optional[CorrugationSpec] = None,
    ) -> "DomeGeometry":
        """Create an elliptical dome (oblate ellipsoid section)."""
        return cls(
            base_diameter_m=base_diameter_m,
            dome_height_m=dome_height_m,
            profile="elliptical",
            corrugations=corrugations,
        )

    @classmethod
    def parabolic(
        cls,
        base_diameter_m: float,
        dome_height_m: float,
        corrugations: Optional[CorrugationSpec] = None,
    ) -> "DomeGeometry":
        """Create a parabolic dome."""
        return cls(
            base_diameter_m=base_diameter_m,
            dome_height_m=dome_height_m,
            profile="parabolic",
            corrugations=corrugations,
        )

    @classmethod
    def from_radius_of_curvature(
        cls,
        base_diameter_m: float,
        radius_of_curvature_m: float,
        profile: DomeProfile = "spherical",
    ) -> "DomeGeometry":
        """
        Create dome from radius of curvature instead of height.

        For a spherical cap: h = R - sqrt(R² - r²)
        """
        r = base_diameter_m / 2
        R = radius_of_curvature_m
        if R < r:
            raise ValueError(f"Radius of curvature ({R}m) must be >= base radius ({r}m)")
        h = R - np.sqrt(R**2 - r**2)
        return cls(
            base_diameter_m=base_diameter_m,
            dome_height_m=h,
            profile=profile,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        d = {
            "base_diameter_m": self.base_diameter_m,
            "dome_height_m": self.dome_height_m,
            "profile": self.profile,
        }
        if self.corrugations is not None:
            d["corrugations"] = {
                "num_corrugations": self.corrugations.num_corrugations,
                "depth_m": self.corrugations.depth_m,
                "width_m": self.corrugations.width_m,
                "inner_radius_m": self.corrugations.inner_radius_m,
                "outer_radius_m": self.corrugations.outer_radius_m,
            }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "DomeGeometry":
        """Deserialize from dictionary."""
        corrugations = None
        if "corrugations" in data:
            corrugations = CorrugationSpec(**data["corrugations"])
        return cls(
            base_diameter_m=data["base_diameter_m"],
            dome_height_m=data["dome_height_m"],
            profile=data.get("profile", "spherical"),
            corrugations=corrugations,
        )
