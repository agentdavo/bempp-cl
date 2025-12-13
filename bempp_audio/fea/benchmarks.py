"""
Canonical benchmark configurations for vibroacoustic validation.

These configurations are intentionally explicit and stable so that modal and
impedance validation results are reproducible across solver refactors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Optional
import numpy as np

from bempp_audio.fea.dome_geometry import DomeGeometry
from bempp_audio.fea.materials import ShellMaterial

if TYPE_CHECKING:  # pragma: no cover
    from bempp_audio.fea.dome_meshing import DomeMesh


@dataclass(frozen=True)
class Benchmark3TiConfig_v1:
    """
    Benchmark 3.0-Ti: deep-drawn titanium dome with a finite clamp ring.

    Structural domain is the dome mid-surface out to the clamp outer radius.
    The clamp is modeled as a finite annular band where all DOFs are fixed.

    Final specs (user-confirmed):
    - Rise (h): 14.5 mm
    - Clamp outer radius: 37.25 mm
    - Clamp band width: 0.75 mm (inner = 36.5 mm)
    - Material: Ti (E=110 GPa, ρ=4500 kg/m³, ν=0.34)
    - Thickness: 50 µm
    """

    rise_m: float = 14.5e-3
    clamp_outer_radius_m: float = 37.25e-3
    clamp_band_width_m: float = 0.75e-3
    material: ShellMaterial = ShellMaterial.titanium(thickness_m=50e-6, grade="1")

    @property
    def clamp_inner_radius_m(self) -> float:
        return float(self.clamp_outer_radius_m - self.clamp_band_width_m)

    @property
    def base_diameter_m(self) -> float:
        return float(2.0 * self.clamp_outer_radius_m)

    @property
    def sphere_radius_m(self) -> float:
        a = float(self.clamp_outer_radius_m)
        h = float(self.rise_m)
        return float((a * a + h * h) / (2.0 * h))

    def dome_geometry(self) -> DomeGeometry:
        return DomeGeometry.spherical(base_diameter_m=self.base_diameter_m, dome_height_m=float(self.rise_m))

    def clamp_vertex_mask(self, vertices_xyz: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask selecting vertices inside the clamp annulus.

        Parameters
        ----------
        vertices_xyz : np.ndarray
            (N,3) coordinates in meters.
        """
        v = np.asarray(vertices_xyz, dtype=float)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("vertices_xyz must be shape (N,3).")
        r = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
        r0 = float(self.clamp_inner_radius_m)
        r1 = float(self.clamp_outer_radius_m)
        return (r >= r0) & (r <= r1)

    def clamp_vertex_mask_for_mesh(self, mesh: "DomeMesh") -> np.ndarray:
        return self.clamp_vertex_mask(mesh.vertices)


@dataclass(frozen=True)
class Benchmark3TiFormerConfig_v1:
    """
    Benchmark3Ti + former (lumped ring parameters).

    This config extends `Benchmark3TiConfig_v1` with parameters for a bonded
    Kapton/Mylar former and adhesive annulus. These parameters are currently
    informational/lumped (they are not yet coupled into the shell stiffness
    or boundary conditions), but they are kept here to standardize future
    "real dome" comparisons and sweeps.
    """

    base: Benchmark3TiConfig_v1 = Benchmark3TiConfig_v1()

    # Former geometry (simple cylindrical ring model around the clamp region).
    former_mean_radius_m: float = 36.9e-3  # ~mid of 36.5–37.25mm band
    former_height_m: float = 4.0e-3
    former_thickness_m: float = 50e-6
    former_material: ShellMaterial = ShellMaterial.polyimide(thickness_m=50e-6)

    # Adhesive annulus (lumped mass + loss factor; stiffness coupling TBD).
    adhesive_width_m: float = 0.8e-3
    adhesive_thickness_m: float = 30e-6
    adhesive_density_kg_m3: float = 1200.0
    adhesive_loss_factor: float = 0.10

    @property
    def clamp_inner_radius_m(self) -> float:
        return self.base.clamp_inner_radius_m

    @property
    def clamp_outer_radius_m(self) -> float:
        return self.base.clamp_outer_radius_m

    @property
    def sphere_radius_m(self) -> float:
        return self.base.sphere_radius_m

    def dome_geometry(self) -> DomeGeometry:
        return self.base.dome_geometry()

    def clamp_vertex_mask(self, vertices_xyz: np.ndarray) -> np.ndarray:
        return self.base.clamp_vertex_mask(vertices_xyz)

    def clamp_vertex_mask_for_mesh(self, mesh: "DomeMesh") -> np.ndarray:
        return self.base.clamp_vertex_mask_for_mesh(mesh)

    @property
    def former_mass_kg(self) -> float:
        """
        Approximate former ring mass (cylindrical shell).

        Volume ≈ (circumference at mean radius) * height * thickness.
        """
        r = float(self.former_mean_radius_m)
        vol = (2.0 * np.pi * r) * float(self.former_height_m) * float(self.former_thickness_m)
        return float(self.former_material.density_kg_m3) * float(vol)

    @property
    def adhesive_mass_kg(self) -> float:
        """
        Approximate adhesive annulus mass.

        Area ≈ (circumference at mean radius) * width.
        Volume ≈ area * thickness.
        """
        r = float(self.former_mean_radius_m)
        area = (2.0 * np.pi * r) * float(self.adhesive_width_m)
        vol = area * float(self.adhesive_thickness_m)
        return float(self.adhesive_density_kg_m3) * float(vol)

    @property
    def added_mass_kg(self) -> float:
        """Former + adhesive total lumped added mass."""
        return float(self.former_mass_kg + self.adhesive_mass_kg)


@dataclass(frozen=True)
class Benchmark3TiSurroundConfig_v1:
    """
    Benchmark3Ti + surround (lumped compliance + damping parameters).

    The intent is to capture "dome + compliant surround + clamp" behavior for
    breakup-control studies. Like the former config, these parameters are
    currently informational (not yet baked into the shell FEM formulation).
    """

    base: Benchmark3TiConfig_v1 = Benchmark3TiConfig_v1()

    surround_inner_radius_m: float = 28.0e-3
    surround_outer_radius_m: float = 36.5e-3  # meets the clamp inner radius by default
    surround_material: ShellMaterial = ShellMaterial.polyimide(thickness_m=25e-6)
    surround_loss_factor: float = 0.15

    @property
    def clamp_inner_radius_m(self) -> float:
        return self.base.clamp_inner_radius_m

    @property
    def clamp_outer_radius_m(self) -> float:
        return self.base.clamp_outer_radius_m

    @property
    def sphere_radius_m(self) -> float:
        return self.base.sphere_radius_m

    def dome_geometry(self) -> DomeGeometry:
        return self.base.dome_geometry()

    def clamp_vertex_mask(self, vertices_xyz: np.ndarray) -> np.ndarray:
        return self.base.clamp_vertex_mask(vertices_xyz)

    def clamp_vertex_mask_for_mesh(self, mesh: "DomeMesh") -> np.ndarray:
        return self.base.clamp_vertex_mask_for_mesh(mesh)

    def surround_vertex_mask(self, vertices_xyz: np.ndarray) -> np.ndarray:
        v = np.asarray(vertices_xyz, dtype=float)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("vertices_xyz must be shape (N,3).")
        r = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
        r0 = float(self.surround_inner_radius_m)
        r1 = float(self.surround_outer_radius_m)
        return (r >= r0) & (r <= r1)


def sweep_benchmark3ti_v1(
    *,
    rise_m: Optional[Iterable[float]] = None,
    clamp_band_width_m: Optional[Iterable[float]] = None,
    thickness_m: Optional[Iterable[float]] = None,
    structural_damping: Optional[Iterable[float]] = None,
) -> List[Benchmark3TiConfig_v1]:
    """
    Generate a small grid of `Benchmark3TiConfig_v1` variants for sweeps.

    This is a pure-configuration helper (it does not run FEM/BEM).
    """
    base = Benchmark3TiConfig_v1()
    rise_vals = list(rise_m) if rise_m is not None else [base.rise_m]
    band_vals = list(clamp_band_width_m) if clamp_band_width_m is not None else [base.clamp_band_width_m]
    t_vals = list(thickness_m) if thickness_m is not None else [base.material.thickness_m]
    eta_vals = list(structural_damping) if structural_damping is not None else [base.material.structural_damping]

    out: list[Benchmark3TiConfig_v1] = []
    for h in rise_vals:
        for bw in band_vals:
            for t in t_vals:
                for eta in eta_vals:
                    mat = ShellMaterial.titanium(thickness_m=float(t), grade="1")
                    mat = ShellMaterial(
                        name=mat.name,
                        youngs_modulus_pa=mat.youngs_modulus_pa,
                        density_kg_m3=mat.density_kg_m3,
                        poissons_ratio=mat.poissons_ratio,
                        thickness_m=mat.thickness_m,
                        structural_damping=float(eta),
                    )
                    out.append(
                        Benchmark3TiConfig_v1(
                            rise_m=float(h),
                            clamp_outer_radius_m=base.clamp_outer_radius_m,
                            clamp_band_width_m=float(bw),
                            material=mat,
                        )
                    )
    return out
