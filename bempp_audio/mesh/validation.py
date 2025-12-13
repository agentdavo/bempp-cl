"""
Mesh resolution validation utilities for BEM acoustic simulations.

Provides tools to verify mesh resolution is adequate for target frequencies
using the λ/N rule (N elements per wavelength).
"""

import numpy as np
from typing import Dict, Tuple, Optional

from bempp_audio.progress import get_logger
from bempp_audio.mesh.checks import topology_report_arrays


class MeshResolutionValidator:
    """Validates mesh resolution for BEM accuracy at target frequencies."""
    
    # Speed of sound in air at 20°C (m/s)
    DEFAULT_SOUND_SPEED = 343.0
    
    # Quality thresholds (elements per wavelength)
    QUALITY_THRESHOLDS = {
        'excellent': 10.0,
        'very_good': 8.0,
        'good': 6.0,
        'marginal': 4.0,
        'inadequate': 0.0,
    }
    
    @classmethod
    def validate_element_size(
        cls,
        element_size_m: float,
        freq_max_hz: float,
        c: float = DEFAULT_SOUND_SPEED,
        verbose: bool = True
    ) -> Tuple[bool, str, float]:
        """
        Check if element size is adequate for target frequency.
        
        Parameters
        ----------
        element_size_m : float
            Maximum element size in meters
        freq_max_hz : float
            Maximum frequency to simulate (Hz)
        c : float, optional
            Speed of sound (m/s), default 343
        verbose : bool, optional
            Print detailed analysis
            
        Returns
        -------
        is_adequate : bool
            True if N >= 6 (good or better)
        quality : str
            Quality rating ('excellent', 'very_good', 'good', 'marginal', 'inadequate')
        N : float
            Elements per wavelength at freq_max
            
        Examples
        --------
        >>> from bempp_audio.mesh.validation import MeshResolutionValidator
        >>> is_ok, quality, N = MeshResolutionValidator.validate_element_size(
        ...     element_size_m=0.003,  # 3mm
        ...     freq_max_hz=16000,
        ...     verbose=True
        ... )
        At 16000 Hz with 3.0 mm elements:
          λ = 21.4 mm
          N = 7.1 elements/wavelength
          Quality: GOOD ✓
        """
        wavelength = c / freq_max_hz
        N = wavelength / element_size_m
        
        # Determine quality rating
        if N >= cls.QUALITY_THRESHOLDS['excellent']:
            quality = 'excellent'
        elif N >= cls.QUALITY_THRESHOLDS['very_good']:
            quality = 'very_good'
        elif N >= cls.QUALITY_THRESHOLDS['good']:
            quality = 'good'
        elif N >= cls.QUALITY_THRESHOLDS['marginal']:
            quality = 'marginal'
        else:
            quality = 'inadequate'

        is_adequate = N >= cls.QUALITY_THRESHOLDS['good']

        if verbose:
            logger = get_logger()
            logger.info(
                f"Mesh resolution @ {float(freq_max_hz):.0f} Hz: "
                f"h={element_size_m * 1e3:.1f}mm λ={wavelength * 1e3:.1f}mm "
                f"N={N:.1f} quality={quality}"
            )

        return is_adequate, quality, N
    
    @classmethod
    def validate_mesh_config(
        cls,
        config: Dict[str, float],
        freq_max_hz: float,
        c: float = DEFAULT_SOUND_SPEED,
        verbose: bool = True
    ) -> Dict[str, Tuple[bool, str, float]]:
        """
        Validate multiple mesh regions (throat, mouth, box).
        
        Parameters
        ----------
        config : dict
            Mesh configuration with keys like 'h_throat', 'h_mouth', 'h_box'
        freq_max_hz : float
            Maximum frequency to simulate (Hz)
        c : float, optional
            Speed of sound (m/s)
        verbose : bool, optional
            Print detailed analysis
            
        Returns
        -------
        results : dict
            Dictionary mapping region names to (is_adequate, quality, N) tuples
            
        Examples
        --------
        >>> from bempp_audio.mesh.unified_enclosure import UnifiedMeshConfig
        >>> config = UnifiedMeshConfig(
        ...     throat_diameter=0.025, mouth_diameter=0.15,
        ...     waveguide_length=0.1, box_width=0.3, box_height=0.4, box_depth=0.25
        ... )
        >>> results = MeshResolutionValidator.validate_mesh_config(
        ...     config={'h_throat': config.h_throat, 'h_mouth': config.h_mouth, 'h_box': config.h_box},
        ...     freq_max_hz=16000
        ... )
        """
        results = {}

        if verbose:
            logger = get_logger()
            logger.info(f"Mesh resolution validation: f_max={freq_max_hz:.0f} Hz")

        for region, element_size in config.items():
            if verbose:
                logger = get_logger()
                logger.info(f"{region}:")
            is_adequate, quality, N = cls.validate_element_size(
                element_size_m=element_size,
                freq_max_hz=freq_max_hz,
                c=c,
                verbose=verbose
            )
            results[region] = (is_adequate, quality, N)

        if verbose:
            all_adequate = all(r[0] for r in results.values())
            if all_adequate:
                logger = get_logger()
                logger.info("All regions meet minimum quality threshold (N >= 6)")
            else:
                inadequate = [k for k, v in results.items() if not v[0]]
                logger = get_logger()
                logger.warning(f"Inadequate resolution in: {', '.join(inadequate)}")

        return results
    
    @classmethod
    def recommend_element_size(
        cls,
        freq_max_hz: float,
        target_quality: str = 'good',
        c: float = DEFAULT_SOUND_SPEED,
        verbose: bool = True
    ) -> float:
        """
        Recommend element size for target frequency and quality.
        
        Parameters
        ----------
        freq_max_hz : float
            Maximum frequency to simulate (Hz)
        target_quality : str, optional
            Desired quality: 'excellent' (N≥10), 'very_good' (N≥8), 'good' (N≥6)
        c : float, optional
            Speed of sound (m/s)
        verbose : bool, optional
            Print recommendation
            
        Returns
        -------
        h_max : float
            Maximum recommended element size (meters)
            
        Examples
        --------
        >>> h = MeshResolutionValidator.recommend_element_size(
        ...     freq_max_hz=20000,
        ...     target_quality='excellent'
        ... )
        For f_max = 20000 Hz with 'excellent' quality (N ≥ 10):
          Recommended element size: ≤ 1.7 mm
        """
        if target_quality not in ['excellent', 'very_good', 'good']:
            raise ValueError(f"Invalid quality '{target_quality}'. Use 'excellent', 'very_good', or 'good'.")
        
        wavelength = c / freq_max_hz
        N_target = cls.QUALITY_THRESHOLDS[target_quality]
        h_max = wavelength / N_target
        
        if verbose:
            logger = get_logger()
            logger.info(
                f"Recommended element size: f_max={float(freq_max_hz):.0f} Hz "
                f"quality={target_quality} (N>={float(N_target):.0f}) -> h<={h_max * 1e3:.1f}mm"
            )

        return h_max
    
    @classmethod
    def frequency_range_for_element_size(
        cls,
        element_size_m: float,
        min_quality: str = 'good',
        c: float = DEFAULT_SOUND_SPEED,
        verbose: bool = True
    ) -> float:
        """
        Calculate maximum frequency for given element size and quality.
        
        Parameters
        ----------
        element_size_m : float
            Element size (meters)
        min_quality : str, optional
            Minimum quality: 'excellent', 'very_good', 'good'
        c : float, optional
            Speed of sound (m/s)
        verbose : bool, optional
            Print result
            
        Returns
        -------
        f_max : float
            Maximum frequency (Hz) for specified quality
            
        Examples
        --------
        >>> f = MeshResolutionValidator.frequency_range_for_element_size(
        ...     element_size_m=0.005,  # 5mm
        ...     min_quality='good'
        ... )
        With 5.0 mm elements and 'good' quality (N ≥ 6):
          Accurate up to: 11433 Hz
        """
        if min_quality not in ['excellent', 'very_good', 'good']:
            raise ValueError(f"Invalid quality '{min_quality}'. Use 'excellent', 'very_good', or 'good'.")
        
        N_min = cls.QUALITY_THRESHOLDS[min_quality]
        wavelength = element_size_m * N_min
        f_max = c / wavelength
        
        if verbose:
            logger = get_logger()
            logger.info(
                f"Max frequency for h={element_size_m * 1e3:.1f}mm quality={min_quality} "
                f"(N>={float(N_min):.0f}) -> f_max={float(f_max):.0f} Hz"
            )

        return f_max


class MeshResolutionPresets:
    """Predefined mesh resolution presets for different use cases."""

    @staticmethod
    def ultra_fast() -> Dict[str, float]:
        """
        Ultra-fast computation (very coarse mesh).

        Intended for "does it run?" smoke tests and rapid iteration where you want
        roughly ~1/4 the element count of `super_fast` on typical waveguide meshes.
        High-frequency accuracy will be limited.
        """
        return {
            "h_throat": 0.012,  # 12mm
            "h_mouth": 0.030,  # 30mm
            "h_box": 0.060,
            "h_baffle": 0.030,
            "h_sides": 0.060,
            "h_back": 0.080,
            # Must satisfy WaveguideMeshConfig.validate (>= 10).
            "n_axial_slices": 10,
            # Very coarse perimeter sampling for "does it run?" checks.
            "n_circumferential": 16,
            "corner_resolution": 2,
        }

    @staticmethod
    def super_fast() -> Dict[str, float]:
        """
        Very fast computation (coarse mesh).

        Intended for quick iteration and sanity checks over typical audio
        sweeps (e.g. 200 Hz–20 kHz) where high-frequency accuracy is not the
        priority.
        
        Returns
        -------
        dict
            Element sizes: h_throat, h_mouth, h_box (meters)
        """
        return {
            # Coarse "interactive" preset intended to keep large waveguides in the ~O(1e3) vertex range.
            'h_throat': 0.007,  # 7mm
            'h_mouth': 0.018,   # 18mm
            'h_box': 0.036,     # 36mm
            # Unified enclosure sizing knobs (preferred)
            'h_baffle': 0.018,
            'h_sides': 0.036,
            'h_back': 0.050,
            # Waveguide discretization hints (used by examples / generators that morph cross-sections)
            'n_axial_slices': 10,
            # 1" throat circumference ~80mm -> 24 samples ~3.3mm spacing
            'n_circumferential': 24,
            'corner_resolution': 4,
        }
    
    @staticmethod
    def fast() -> Dict[str, float]:
        """
        Fast computation (coarser mesh).

        Good tradeoff for many interactive runs while still keeping throat
        resolution reasonable.
        
        Returns
        -------
        dict
            Element sizes: h_throat, h_mouth, h_box (meters)
        """
        return {
            'h_throat': 0.003,  # 3mm
            'h_mouth': 0.008,   # 8mm
            'h_box': 0.016,     # 16mm
            'h_baffle': 0.008,
            'h_sides': 0.016,
            'h_back': 0.024,
            'n_axial_slices': 30,
            # Avoid over-refining the throat cap by boundary segmentation.
            # 48 samples gives ~1.7mm spacing on a 1" throat; adequate for 16 kHz design iteration.
            'n_circumferential': 48,
            # Bias more of the fixed vertex budget into the corner-arc regions so a rounded-rectangle
            # mouth keeps the requested fillet radius even on fast/coarse runs.
            'corner_resolution': 6,
        }
    
    @staticmethod
    def standard() -> Dict[str, float]:
        """
        Standard resolution (default).

        Targets "good or better" resolution at the throat for 20 kHz sweeps
        while keeping the mouth/baffle coarser for performance.
        
        Returns
        -------
        dict
            Element sizes: h_throat, h_mouth, h_box (meters)
        """
        return {
            'h_throat': 0.002,  # 2mm
            'h_mouth': 0.006,   # 6mm
            'h_box': 0.012,     # 12mm
            'h_baffle': 0.006,
            'h_sides': 0.012,
            'h_back': 0.020,
            'n_axial_slices': 40,
            # 64 samples gives ~1.25mm spacing on a 1" throat; good margin for 20 kHz.
            'n_circumferential': 64,
            'corner_resolution': 8,
        }

    @staticmethod
    def slow() -> Dict[str, float]:
        """
        Slower computation (finer mesh).

        Use this when you want extra margin for high-frequency accuracy.
        """
        return {
            'h_throat': 0.0015,  # 1.5mm
            'h_mouth': 0.0030,   # 3mm
            'h_box': 0.0060,     # 6mm
            'h_baffle': 0.0030,
            'h_sides': 0.0060,
            'h_back': 0.0100,
            'n_axial_slices': 60,
            'n_circumferential': 96,
            'corner_resolution': 12,
        }
    
    @classmethod
    def get_preset(cls, name: str) -> Dict[str, float]:
        """
        Get preset by name.
        
        Parameters
        ----------
        name : str
            Preset name: 'super_fast', 'fast', 'standard', 'slow' (dashes are accepted,
            e.g. 'super-fast').
            
        Returns
        -------
        dict
            Element sizes
            
        Raises
        ------
        ValueError
            If preset name is invalid
        """
        key = str(name).strip().lower().replace("-", "_")
        presets = {
            'ultra_fast': cls.ultra_fast,
            'super_fast': cls.super_fast,
            'fast': cls.fast,
            'standard': cls.standard,
            'slow': cls.slow,
        }
        
        if key not in presets:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(presets.keys())}")
        
        return presets[key]()
    
    @classmethod
    def list_presets(cls, verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        List all available presets with details.
        
        Parameters
        ----------
        verbose : bool, optional
            Print formatted table
            
        Returns
        -------
        dict
            All presets
        """
        presets = {
            'ultra_fast': cls.ultra_fast(),
            'super_fast': cls.super_fast(),
            'fast': cls.fast(),
            'standard': cls.standard(),
            'slow': cls.slow(),
        }
        
        if verbose:
            logger = get_logger()
            logger.info("Mesh resolution presets:")
            logger.info(
                f"{'Preset':<20} {'h_throat':<10} {'h_mouth':<10} {'h_box':<10} "
                f"{'n_axial':<8} {'n_circ':<8} {'f_max (N>=6)':<15}"
            )

            for name, sizes in presets.items():
                # Calculate max frequency (for throat, most restrictive)
                f_max = MeshResolutionValidator.frequency_range_for_element_size(
                    sizes['h_throat'],
                    min_quality='good',
                    verbose=False
                )
                h_t = sizes["h_throat"] * 1e3
                h_m = sizes["h_mouth"] * 1e3
                h_b = sizes["h_box"] * 1e3
                n_a = int(sizes.get("n_axial_slices", 0))
                n_c = int(sizes.get("n_circumferential", 0))
                f_k = f_max / 1e3
                logger.info(
                    f"{name:<20} {h_t:6.1f} mm   {h_m:6.1f} mm   {h_b:6.1f} mm   "
                    f"{n_a:6d}   {n_c:6d}   {f_k:6.1f} kHz"
                )

        return presets


class MeshTopologyValidator:
    """Validates mesh topology for BEM suitability (closedness, manifoldness)."""

    @classmethod
    def validate_closed(
        cls,
        vertices: np.ndarray,
        elements: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[bool, Dict]:
        """
        Check if a triangular mesh is closed (watertight).

        A closed mesh has every edge shared by exactly 2 triangles.
        Non-manifold edges (shared by >2 triangles) or boundary edges
        (shared by only 1 triangle) indicate holes or topology issues.

        Parameters
        ----------
        vertices : np.ndarray
            Vertex array of shape (3, n_vertices).
        elements : np.ndarray
            Element array of shape (3, n_elements) with vertex indices.
        verbose : bool
            If True, log validation results.

        Returns
        -------
        is_closed : bool
            True if mesh is closed (all edges shared by exactly 2 triangles).
        info : dict
            Detailed topology information:
            - n_edges: Total number of unique edges
            - n_boundary: Number of boundary edges (shared by 1 triangle)
            - n_nonmanifold: Number of non-manifold edges (shared by >2 triangles)
            - boundary_edges: List of (v1, v2) tuples for boundary edges
        """
        logger = get_logger()
        report = topology_report_arrays(vertices, elements)
        is_closed = report.n_boundary_edges == 0 and report.n_nonmanifold_edges == 0 and report.n_duplicate_triangles == 0

        info = {
            "n_edges": report.n_edges,
            "n_boundary": report.n_boundary_edges,
            "n_nonmanifold": report.n_nonmanifold_edges,
            "n_duplicate_triangles": report.n_duplicate_triangles,
        }

        if verbose:
            if is_closed:
                logger.info(f"Mesh topology: closed (manifold), {report.n_edges} edges")
            else:
                if report.n_boundary_edges:
                    logger.warning(f"Mesh topology: {report.n_boundary_edges} boundary edges (mesh has holes)")
                if report.n_nonmanifold_edges:
                    logger.warning(f"Mesh topology: {report.n_nonmanifold_edges} non-manifold edges")
                if report.n_duplicate_triangles:
                    logger.warning(f"Mesh topology: {report.n_duplicate_triangles} duplicate triangles")

        return is_closed, info

    @classmethod
    def validate_mesh(
        cls,
        mesh: "LoudspeakerMesh",
        verbose: bool = True,
    ) -> Tuple[bool, Dict]:
        """
        Validate mesh topology from a LoudspeakerMesh object.

        Parameters
        ----------
        mesh : LoudspeakerMesh
            Mesh to validate.
        verbose : bool
            If True, log validation results.

        Returns
        -------
        is_valid : bool
            True if mesh passes all topology checks.
        info : dict
            Detailed validation results.
        """
        grid = mesh.grid
        vertices = np.asarray(grid.vertices)
        elements = np.asarray(grid.elements)
        return cls.validate_closed(vertices, elements, verbose=verbose)


# =============================================================================
# Element Quality Metrics
# =============================================================================


class ElementQualityMetrics:
    """Compute and report element quality metrics for triangular meshes.

    For BEM, the key quality concerns are:
    - Aspect ratio: Very elongated triangles can cause integration issues
    - Size uniformity: Smooth size transitions prevent numerical artifacts

    Typical thresholds:
    - Aspect ratio < 3: Good quality
    - Aspect ratio < 5: Acceptable
    - Aspect ratio > 10: May cause accuracy issues
    """

    @staticmethod
    def compute_triangle_aspect_ratios(
        vertices: np.ndarray,
        elements: np.ndarray,
    ) -> np.ndarray:
        """Compute aspect ratio for each triangle.

        Aspect ratio = longest_edge / shortest_edge

        Parameters
        ----------
        vertices : np.ndarray
            Vertex coordinates, shape (3, n_vertices).
        elements : np.ndarray
            Triangle connectivity, shape (3, n_elements).

        Returns
        -------
        np.ndarray
            Aspect ratio for each element, shape (n_elements,).
        """
        vertices = np.asarray(vertices)
        elements = np.asarray(elements)

        if vertices.shape[0] != 3:
            vertices = vertices.T
        if elements.shape[0] != 3:
            elements = elements.T

        n_elem = elements.shape[1]
        aspect_ratios = np.zeros(n_elem)

        for i in range(n_elem):
            v0 = vertices[:, elements[0, i]]
            v1 = vertices[:, elements[1, i]]
            v2 = vertices[:, elements[2, i]]

            # Edge lengths
            e0 = np.linalg.norm(v1 - v0)
            e1 = np.linalg.norm(v2 - v1)
            e2 = np.linalg.norm(v0 - v2)

            edges = np.array([e0, e1, e2])
            shortest = edges.min()
            longest = edges.max()

            if shortest > 1e-12:
                aspect_ratios[i] = longest / shortest
            else:
                aspect_ratios[i] = np.inf

        return aspect_ratios

    @staticmethod
    def compute_triangle_areas(
        vertices: np.ndarray,
        elements: np.ndarray,
    ) -> np.ndarray:
        """Compute area of each triangle.

        Parameters
        ----------
        vertices : np.ndarray
            Vertex coordinates, shape (3, n_vertices).
        elements : np.ndarray
            Triangle connectivity, shape (3, n_elements).

        Returns
        -------
        np.ndarray
            Area of each element, shape (n_elements,).
        """
        vertices = np.asarray(vertices)
        elements = np.asarray(elements)

        if vertices.shape[0] != 3:
            vertices = vertices.T
        if elements.shape[0] != 3:
            elements = elements.T

        n_elem = elements.shape[1]
        areas = np.zeros(n_elem)

        for i in range(n_elem):
            v0 = vertices[:, elements[0, i]]
            v1 = vertices[:, elements[1, i]]
            v2 = vertices[:, elements[2, i]]

            # Cross product for area
            cross = np.cross(v1 - v0, v2 - v0)
            areas[i] = 0.5 * np.linalg.norm(cross)

        return areas

    @classmethod
    def quality_report(
        cls,
        vertices: np.ndarray,
        elements: np.ndarray,
        verbose: bool = True,
    ) -> Dict:
        """Generate comprehensive element quality report.

        Parameters
        ----------
        vertices : np.ndarray
            Vertex coordinates.
        elements : np.ndarray
            Triangle connectivity.
        verbose : bool
            If True, log the report.

        Returns
        -------
        dict
            Quality metrics including:
            - n_elements: Number of triangles
            - aspect_ratio_min/max/mean/std: Aspect ratio statistics
            - area_min/max/mean/std: Area statistics in mm²
            - n_poor_quality: Count of elements with aspect ratio > 5
            - n_very_poor: Count of elements with aspect ratio > 10
        """
        logger = get_logger()

        aspect_ratios = cls.compute_triangle_aspect_ratios(vertices, elements)
        areas = cls.compute_triangle_areas(vertices, elements)

        n_elem = len(aspect_ratios)
        n_poor = int(np.sum(aspect_ratios > 5))
        n_very_poor = int(np.sum(aspect_ratios > 10))

        report = {
            "n_elements": n_elem,
            "aspect_ratio_min": float(np.min(aspect_ratios)),
            "aspect_ratio_max": float(np.max(aspect_ratios)),
            "aspect_ratio_mean": float(np.mean(aspect_ratios)),
            "aspect_ratio_std": float(np.std(aspect_ratios)),
            "area_min_mm2": float(np.min(areas)) * 1e6,
            "area_max_mm2": float(np.max(areas)) * 1e6,
            "area_mean_mm2": float(np.mean(areas)) * 1e6,
            "area_std_mm2": float(np.std(areas)) * 1e6,
            "n_poor_quality": n_poor,
            "n_very_poor": n_very_poor,
        }

        if verbose:
            logger.info(
                f"Element quality: {n_elem} triangles, "
                f"aspect ratio {report['aspect_ratio_min']:.1f}-{report['aspect_ratio_max']:.1f} "
                f"(mean {report['aspect_ratio_mean']:.1f})"
            )
            if n_poor > 0:
                logger.warning(
                    f"Element quality: {n_poor} elements with aspect ratio > 5 "
                    f"({100.0 * n_poor / n_elem:.1f}%)"
                )
            if n_very_poor > 0:
                logger.warning(
                    f"Element quality: {n_very_poor} elements with aspect ratio > 10 "
                    f"({100.0 * n_very_poor / n_elem:.1f}%)"
                )

        return report

    @classmethod
    def validate_mesh_quality(
        cls,
        vertices: np.ndarray,
        elements: np.ndarray,
        max_aspect_ratio: float = 10.0,
        verbose: bool = True,
    ) -> Tuple[bool, Dict]:
        """Validate mesh quality against thresholds.

        Parameters
        ----------
        vertices : np.ndarray
            Vertex coordinates.
        elements : np.ndarray
            Triangle connectivity.
        max_aspect_ratio : float
            Maximum allowed aspect ratio (default 10.0).
        verbose : bool
            If True, log validation results.

        Returns
        -------
        is_valid : bool
            True if no element exceeds max_aspect_ratio.
        report : dict
            Quality metrics.
        """
        report = cls.quality_report(vertices, elements, verbose=verbose)
        is_valid = report["aspect_ratio_max"] <= max_aspect_ratio
        return is_valid, report
