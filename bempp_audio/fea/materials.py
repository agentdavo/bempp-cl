"""
Material property definitions for shell FEM analysis.

Provides standard materials commonly used in compression driver diaphragms:
- Titanium (Grade 1, Grade 5)
- Beryllium
- Aluminum alloys

Also provides utility functions for bending mode estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class ShellMaterial:
    """
    Material properties for thin shell FEM analysis.

    All properties are in SI units.
    """

    name: str
    youngs_modulus_pa: float
    density_kg_m3: float
    poissons_ratio: float
    thickness_m: float
    structural_damping: float = 0.001  # Loss factor η

    @property
    def flexural_rigidity(self) -> float:
        """Flexural rigidity D = Eh³ / (12(1-ν²)) [N·m]."""
        h = self.thickness_m
        e = self.youngs_modulus_pa
        nu = self.poissons_ratio
        return e * h**3 / (12 * (1 - nu**2))

    @property
    def mass_per_area(self) -> float:
        """Mass per unit area ρh [kg/m²]."""
        return self.density_kg_m3 * self.thickness_m

    @classmethod
    def titanium(cls, thickness_m: float = 50e-6, grade: str = "1") -> "ShellMaterial":
        """
        Titanium material (Grade 1 or Grade 5).

        Grade 1: Pure titanium, commonly used for CD domes
        Grade 5: Ti-6Al-4V alloy, higher strength
        """
        if grade == "1":
            return cls(
                name="Titanium Grade 1",
                youngs_modulus_pa=110e9,
                density_kg_m3=4500,
                poissons_ratio=0.34,
                thickness_m=thickness_m,
                structural_damping=0.002,
            )
        elif grade == "5":
            return cls(
                name="Titanium Grade 5 (Ti-6Al-4V)",
                youngs_modulus_pa=114e9,
                density_kg_m3=4430,
                poissons_ratio=0.33,
                thickness_m=thickness_m,
                structural_damping=0.001,
            )
        else:
            raise ValueError(f"Unknown titanium grade: {grade}")

    @classmethod
    def beryllium(cls, thickness_m: float = 25e-6) -> "ShellMaterial":
        """
        Beryllium material.

        Very high specific stiffness (E/ρ), ideal for HF domes.
        Pushes breakup modes to higher frequencies.
        """
        return cls(
            name="Beryllium",
            youngs_modulus_pa=287e9,
            density_kg_m3=1850,
            poissons_ratio=0.032,
            thickness_m=thickness_m,
            structural_damping=0.0005,
        )

    @classmethod
    def aluminum(cls, thickness_m: float = 100e-6, alloy: str = "6061") -> "ShellMaterial":
        """
        Aluminum alloy material.

        Common alloys: 6061-T6, 7075-T6
        """
        if alloy == "6061":
            return cls(
                name="Aluminum 6061-T6",
                youngs_modulus_pa=69e9,
                density_kg_m3=2700,
                poissons_ratio=0.33,
                thickness_m=thickness_m,
                structural_damping=0.002,
            )
        elif alloy == "7075":
            return cls(
                name="Aluminum 7075-T6",
                youngs_modulus_pa=72e9,
                density_kg_m3=2810,
                poissons_ratio=0.33,
                thickness_m=thickness_m,
                structural_damping=0.002,
            )
        else:
            raise ValueError(f"Unknown aluminum alloy: {alloy}")

    @classmethod
    def polyimide(cls, thickness_m: float = 25e-6) -> "ShellMaterial":
        """
        Polyimide (Kapton) material.

        Flexible, high damping, used in soft dome tweeters.
        """
        return cls(
            name="Polyimide (Kapton)",
            youngs_modulus_pa=2.5e9,
            density_kg_m3=1420,
            poissons_ratio=0.34,
            thickness_m=thickness_m,
            structural_damping=0.03,
        )


# Preset material constants (50μm titanium is typical for HF CD domes)
TITANIUM = ShellMaterial.titanium(thickness_m=50e-6)
BERYLLIUM = ShellMaterial.beryllium(thickness_m=25e-6)
ALUMINUM = ShellMaterial.aluminum(thickness_m=100e-6)


def estimate_first_bending_mode(
    *,
    material: Optional[ShellMaterial] = None,
    thickness_m: Optional[float] = None,
    radius_m: float,
    youngs_modulus_pa: Optional[float] = None,
    density_kg_m3: Optional[float] = None,
    poissons_ratio: Optional[float] = None,
    boundary: str = "clamped",
) -> float:
    """
    Estimate first bending mode frequency for a circular plate.

    Uses the classical formula for vibrating plates:
        f₁ = (λ₁² / 2π) * (h / R²) * sqrt(D / (ρh))

    where D = Eh³ / (12(1-ν²)) is the flexural rigidity.

    Parameters
    ----------
    material : ShellMaterial, optional
        Material properties (alternative to specifying individual params)
    thickness_m : float, optional
        Shell thickness [m]
    radius_m : float
        Plate radius [m]
    youngs_modulus_pa : float, optional
        Young's modulus [Pa]
    density_kg_m3 : float, optional
        Density [kg/m³]
    poissons_ratio : float, optional
        Poisson's ratio [-]
    boundary : str
        Boundary condition: "clamped" or "simply_supported"

    Returns
    -------
    float
        First bending mode frequency [Hz]

    Notes
    -----
    For a dome (curved surface), the actual frequency will be higher due
    to membrane stiffening. This estimate is for a flat plate and serves
    as a lower bound.
    """
    if material is not None:
        h = material.thickness_m
        e = material.youngs_modulus_pa
        rho = material.density_kg_m3
        nu = material.poissons_ratio
    else:
        if any(p is None for p in [thickness_m, youngs_modulus_pa, density_kg_m3, poissons_ratio]):
            raise ValueError(
                "Either 'material' or all of (thickness_m, youngs_modulus_pa, "
                "density_kg_m3, poissons_ratio) must be provided"
            )
        h = thickness_m
        e = youngs_modulus_pa
        rho = density_kg_m3
        nu = poissons_ratio

    r = radius_m

    # Flexural rigidity
    d = e * h**3 / (12 * (1 - nu**2))

    # First mode eigenvalue λ₁² (from Leissa, "Vibration of Plates")
    if boundary == "clamped":
        # First axisymmetric mode (0,0) for clamped edge
        lambda_sq = 10.21
    elif boundary == "simply_supported":
        # First axisymmetric mode for simply supported edge
        lambda_sq = 4.977
    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")

    # Frequency formula
    f1 = (lambda_sq / (2 * np.pi)) * (1 / r**2) * np.sqrt(d / (rho * h))

    return f1


def bending_wavelength(
    frequency_hz: float,
    material: ShellMaterial,
) -> float:
    """
    Compute bending wavelength at a given frequency.

    Parameters
    ----------
    frequency_hz : float
        Frequency [Hz]
    material : ShellMaterial
        Material properties

    Returns
    -------
    float
        Bending wavelength [m]
    """
    omega = 2 * np.pi * frequency_hz
    d = material.flexural_rigidity
    rho_h = material.mass_per_area

    # Bending wavelength: λ_b = 2π * (D / (ρh * ω²))^(1/4)
    lambda_b = 2 * np.pi * (d / (rho_h * omega**2)) ** 0.25

    return lambda_b


def recommended_mesh_size(
    max_frequency_hz: float,
    material: ShellMaterial,
    elements_per_wavelength: int = 6,
) -> float:
    """
    Compute recommended mesh element size for accurate FEM analysis.

    The mesh must resolve both the acoustic wavelength and the structural
    bending wavelength. The bending wavelength is typically shorter,
    especially at high frequencies.

    Parameters
    ----------
    max_frequency_hz : float
        Maximum analysis frequency [Hz]
    material : ShellMaterial
        Material properties
    elements_per_wavelength : int
        Number of elements per wavelength (default: 6)

    Returns
    -------
    float
        Recommended element size [m]
    """
    # Acoustic wavelength in air (approximate)
    c_air = 343.0  # m/s
    lambda_acoustic = c_air / max_frequency_hz

    # Bending wavelength in the shell
    lambda_bending = bending_wavelength(max_frequency_hz, material)

    # Use the smaller wavelength
    lambda_min = min(lambda_acoustic, lambda_bending)

    return lambda_min / elements_per_wavelength
