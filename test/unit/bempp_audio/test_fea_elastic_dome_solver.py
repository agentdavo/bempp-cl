"""Tests for elastic dome BEM solver in bempp_audio.fea."""

from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.fea.elastic_dome_solver import (
    ModalVelocityProfile,
    ElasticDomeResult,
    ElasticDomeSolverOptions,
)

# Import bempp-dependent classes only if available
try:
    import bempp_cl.api
    BEMPP_AVAILABLE = True
except ImportError:
    BEMPP_AVAILABLE = False


class TestModalVelocityProfile:
    """Tests for ModalVelocityProfile class."""

    def test_from_piston_creates_uniform_velocity(self):
        """Piston profile should have uniform velocity."""
        freqs = np.array([1000, 5000, 10000])
        n_elements = 100
        amplitude = 1.0 + 0.5j

        profile = ModalVelocityProfile.from_piston(freqs, n_elements, amplitude)

        assert len(profile.frequencies_hz) == 3
        assert len(profile.velocity_distributions) == 3

        for freq in freqs:
            v = profile.velocity_distributions[float(freq)]
            assert len(v) == n_elements
            assert np.allclose(v, amplitude)

    def test_from_first_mode_creates_bessel_profile(self):
        """First mode profile should follow J0 Bessel function."""
        freqs = np.array([1000, 5000])
        n_elements = 50
        dome_radius = 0.015  # 15mm radius

        # Create radial positions
        element_radii = np.linspace(0, dome_radius, n_elements)

        profile = ModalVelocityProfile.from_first_mode(
            freqs, element_radii, dome_radius, amplitude=1.0
        )

        # Check that center has maximum (for J0)
        v = profile.velocity_distributions[1000.0]
        assert np.abs(v[0]) > np.abs(v[-1])  # Center > edge

    def test_frequencies_are_stored(self):
        """Profile should store frequency array."""
        freqs = np.array([100, 500, 1000, 5000])
        profile = ModalVelocityProfile.from_piston(freqs, 10)

        assert np.array_equal(profile.frequencies_hz, freqs)


class TestElasticDomeResult:
    """Tests for ElasticDomeResult dataclass."""

    def test_radiation_resistance_extracts_real_part(self):
        """Radiation resistance is real part of impedance."""
        result = ElasticDomeResult(
            frequency_hz=1000,
            wavenumber=18.3,
            surface_pressure=np.zeros(10, dtype=complex),
            radiation_impedance=100 + 50j,
            radiated_power=0.1,
            velocity_distribution=np.ones(10, dtype=complex),
        )

        assert result.radiation_resistance() == 100.0

    def test_radiation_reactance_extracts_imaginary_part(self):
        """Radiation reactance is imaginary part of impedance."""
        result = ElasticDomeResult(
            frequency_hz=1000,
            wavenumber=18.3,
            surface_pressure=np.zeros(10, dtype=complex),
            radiation_impedance=100 + 50j,
            radiated_power=0.1,
            velocity_distribution=np.ones(10, dtype=complex),
        )

        assert result.radiation_reactance() == 50.0


class TestElasticDomeSolverOptions:
    """Tests for ElasticDomeSolverOptions configuration."""

    def test_default_options(self):
        """Default options should have reasonable values."""
        opts = ElasticDomeSolverOptions()

        assert opts.tol == 1e-5
        assert opts.maxiter == 1000
        assert opts.space_type == "DP"
        assert opts.space_order == 0
        assert opts.include_rigid_comparison is True

    def test_custom_options(self):
        """Custom options should be respected."""
        opts = ElasticDomeSolverOptions(
            tol=1e-6,
            maxiter=500,
            space_type="P",
            space_order=1,
            coupling_parameter=10.0,
            include_rigid_comparison=False,
        )

        assert opts.tol == 1e-6
        assert opts.maxiter == 500
        assert opts.space_type == "P"
        assert opts.space_order == 1
        assert opts.coupling_parameter == 10.0
        assert opts.include_rigid_comparison is False


@pytest.mark.skipif(not BEMPP_AVAILABLE, reason="bempp-cl not available")
class TestElasticDomeBEMSolver:
    """Tests that require bempp-cl."""

    @pytest.fixture
    def simple_grid(self):
        """Create a simple triangular mesh (flat disk)."""
        # Create a hexagonal disk mesh
        n_rings = 3
        vertices_list = [[0, 0, 0]]  # center
        elements_list = []

        for ring in range(1, n_rings + 1):
            n_pts = 6 * ring
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            r = 0.01 * ring  # 1cm per ring
            for angle in angles:
                vertices_list.append([r * np.cos(angle), r * np.sin(angle), 0])

        vertices = np.array(vertices_list).T
        n_vertices = vertices.shape[1]

        # Create triangular elements
        # First ring (6 triangles from center)
        for i in range(6):
            elements_list.append([0, i + 1, ((i + 1) % 6) + 1])

        elements = np.array(elements_list, dtype=np.uint32).T

        return bempp_cl.api.Grid(vertices, elements)

    def test_solver_creation(self, simple_grid):
        """Solver should be created with grid."""
        from bempp_audio.fea.elastic_dome_solver import ElasticDomeBEMSolver

        solver = ElasticDomeBEMSolver(simple_grid)

        assert solver.grid is simple_grid
        assert solver.c == 343.0
        assert solver.rho == 1.225

    def test_element_areas_computed(self, simple_grid):
        """Element areas should be computed on init."""
        from bempp_audio.fea.elastic_dome_solver import ElasticDomeBEMSolver

        solver = ElasticDomeBEMSolver(simple_grid)

        assert len(solver.element_areas) == simple_grid.number_of_elements
        assert np.all(solver.element_areas > 0)

    def test_solve_returns_result(self, simple_grid):
        """Solve should return ElasticDomeResult."""
        from bempp_audio.fea.elastic_dome_solver import (
            ElasticDomeBEMSolver,
            ElasticDomeResult,
        )

        solver = ElasticDomeBEMSolver(
            simple_grid,
            options=ElasticDomeSolverOptions(include_rigid_comparison=False),
        )

        velocity = np.ones(simple_grid.number_of_elements, dtype=complex)
        result = solver.solve(1000, velocity)

        assert isinstance(result, ElasticDomeResult)
        assert result.frequency_hz == 1000
        assert len(result.surface_pressure) == simple_grid.number_of_elements

    def test_radiated_power_is_positive(self, simple_grid):
        """Radiated power should be positive for non-zero velocity."""
        from bempp_audio.fea.elastic_dome_solver import ElasticDomeBEMSolver

        solver = ElasticDomeBEMSolver(
            simple_grid,
            options=ElasticDomeSolverOptions(include_rigid_comparison=False),
        )

        velocity = np.ones(simple_grid.number_of_elements, dtype=complex)
        result = solver.solve(1000, velocity)

        assert result.radiated_power > 0

    def test_radiation_impedance_has_positive_real_part(self, simple_grid):
        """Radiation resistance (real part) should be positive."""
        from bempp_audio.fea.elastic_dome_solver import ElasticDomeBEMSolver

        solver = ElasticDomeBEMSolver(
            simple_grid,
            options=ElasticDomeSolverOptions(include_rigid_comparison=False),
        )

        velocity = np.ones(simple_grid.number_of_elements, dtype=complex)
        result = solver.solve(1000, velocity)

        assert result.radiation_resistance() > 0

    def test_solve_frequencies_returns_response(self, simple_grid):
        """Frequency sweep should return response container."""
        from bempp_audio.fea.elastic_dome_solver import (
            ElasticDomeBEMSolver,
            ElasticDomeFrequencyResponse,
            ModalVelocityProfile,
        )

        solver = ElasticDomeBEMSolver(
            simple_grid,
            options=ElasticDomeSolverOptions(include_rigid_comparison=False),
        )

        freqs = np.array([500, 1000, 2000])
        profile = ModalVelocityProfile.from_piston(
            freqs, simple_grid.number_of_elements
        )

        response = solver.solve_frequencies(profile)

        assert isinstance(response, ElasticDomeFrequencyResponse)
        assert len(response.results) == 3
        assert np.array_equal(response.frequencies, freqs)

    def test_rigid_comparison_included_when_enabled(self, simple_grid):
        """Rigid piston comparison should be included when enabled."""
        from bempp_audio.fea.elastic_dome_solver import ElasticDomeBEMSolver

        solver = ElasticDomeBEMSolver(
            simple_grid,
            options=ElasticDomeSolverOptions(include_rigid_comparison=True),
        )

        velocity = np.ones(simple_grid.number_of_elements, dtype=complex)
        result = solver.solve(1000, velocity)

        assert result.rigid_piston_comparison is not None
        assert "z_elastic" in result.rigid_piston_comparison
        assert "z_piston" in result.rigid_piston_comparison
        assert "efficiency_ratio" in result.rigid_piston_comparison

    def test_cache_cleared(self, simple_grid):
        """clear_cache should empty operator cache."""
        from bempp_audio.fea.elastic_dome_solver import ElasticDomeBEMSolver

        solver = ElasticDomeBEMSolver(simple_grid)

        velocity = np.ones(simple_grid.number_of_elements, dtype=complex)
        solver.solve(1000, velocity)  # Populate cache

        assert len(solver._operator_cache) > 0

        solver.clear_cache()

        assert len(solver._operator_cache) == 0


@pytest.mark.skipif(not BEMPP_AVAILABLE, reason="bempp-cl not available")
class TestElasticDomeFrequencyResponse:
    """Tests for frequency response container."""

    @pytest.fixture
    def sample_response(self):
        """Create a sample frequency response."""
        from bempp_audio.fea.elastic_dome_solver import (
            ElasticDomeResult,
            ElasticDomeFrequencyResponse,
            ModalVelocityProfile,
        )

        results = []
        freqs = [500, 1000, 2000]
        for i, freq in enumerate(freqs):
            results.append(
                ElasticDomeResult(
                    frequency_hz=freq,
                    wavenumber=2 * np.pi * freq / 343,
                    surface_pressure=np.ones(10, dtype=complex) * (i + 1),
                    radiation_impedance=100 * (i + 1) + 50j * (i + 1),
                    radiated_power=0.1 * (i + 1),
                    velocity_distribution=np.ones(10, dtype=complex),
                    rigid_piston_comparison={
                        "z_elastic": 100 * (i + 1) + 50j * (i + 1),
                        "z_piston": 100 * (i + 1) + 40j * (i + 1),
                    },
                )
            )

        profile = ModalVelocityProfile.from_piston(np.array(freqs), 10)
        return ElasticDomeFrequencyResponse(results=results, velocity_profile=profile)

    def test_frequencies_property(self, sample_response):
        """Should return frequency array."""
        assert np.array_equal(sample_response.frequencies, [500, 1000, 2000])

    def test_radiation_impedance_property(self, sample_response):
        """Should return impedance array."""
        z = sample_response.radiation_impedance
        assert len(z) == 3
        assert z[0] == 100 + 50j

    def test_radiated_power_property(self, sample_response):
        """Should return power array."""
        p = sample_response.radiated_power
        assert len(p) == 3
        assert p[0] == pytest.approx(0.1)

    def test_spl_db_computed(self, sample_response):
        """SPL should be computed from power."""
        spl = sample_response.spl_db()
        assert len(spl) == 3
        # 0.1 W = 110 dB re 1e-12 W
        assert spl[0] == pytest.approx(110, abs=0.1)

    def test_radiation_resistance_array(self, sample_response):
        """Should return real parts of impedance."""
        r = sample_response.radiation_resistance()
        assert len(r) == 3
        assert r[0] == 100.0
        assert r[1] == 200.0
        assert r[2] == 300.0

    def test_radiation_reactance_array(self, sample_response):
        """Should return imaginary parts of impedance."""
        x = sample_response.radiation_reactance()
        assert len(x) == 3
        assert x[0] == 50.0
        assert x[1] == 100.0
        assert x[2] == 150.0
