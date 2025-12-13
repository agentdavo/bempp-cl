"""Tests for FEM-BEM coupling utilities in bempp_audio.fea."""

from __future__ import annotations

import numpy as np
import pytest

from bempp_audio.fea.fem_bem_coupling import (
    shell_displacement_to_velocity,
    iterative_fem_bem_coupling,
    BEMPP_AVAILABLE,
)


class TestShellDisplacementToVelocity:
    """Tests for shell_displacement_to_velocity function."""

    def test_velocity_is_j_omega_times_displacement(self):
        """v = jω*u for harmonic motion."""
        disp = np.array([1.0 + 0j, 0.5 + 0.5j, 0.0 + 1.0j])
        freq = 1000  # Hz
        omega = 2 * np.pi * freq

        vel = shell_displacement_to_velocity(disp, freq)
        expected = 1j * omega * disp

        assert np.allclose(vel, expected)

    def test_velocity_at_different_frequencies(self):
        """Velocity scales with frequency."""
        disp = np.array([1.0 + 0j])

        vel_1k = shell_displacement_to_velocity(disp, 1000)
        vel_2k = shell_displacement_to_velocity(disp, 2000)

        # At double frequency, velocity should double
        assert np.isclose(abs(vel_2k[0]), 2 * abs(vel_1k[0]))

    def test_phase_shift_is_90_degrees(self):
        """Velocity leads displacement by 90°."""
        disp = np.array([1.0 + 0j])
        freq = 1000

        vel = shell_displacement_to_velocity(disp, freq)

        # Phase of velocity relative to displacement should be π/2
        phase_diff = np.angle(vel[0]) - np.angle(disp[0])
        assert np.isclose(phase_diff, np.pi / 2, atol=1e-10)


class TestIterativeFemBemCoupling:
    """Tests for iterative_fem_bem_coupling function."""

    def test_converges_for_simple_problem(self):
        """Simple fixed-point iteration should converge."""
        # Mock FEM: returns pressure-dependent velocity
        # Mock BEM: returns velocity-dependent pressure
        # Simple linear relationship for testing

        def fem_solve(pressure):
            # Structural response: v = H * (F - p*A)
            # For testing, just return scaled input
            return 0.5 * np.ones_like(pressure) - 0.1 * pressure

        def bem_solve(velocity):
            # Acoustic response: p = Z * v
            return 0.2 * velocity

        initial_v = np.ones(10, dtype=complex)

        result = iterative_fem_bem_coupling(
            fem_solve,
            bem_solve,
            initial_v,
            max_iterations=50,
            tolerance=1e-6,
            relaxation=0.7,
        )

        assert result['converged']
        assert result['iterations'] < 50
        # Check that result is a fixed point (approximately)
        p_check = bem_solve(result['velocity'])
        v_check = fem_solve(p_check)
        assert np.allclose(v_check, result['velocity'], rtol=1e-4)

    def test_returns_residual_history(self):
        """Should return history of residuals."""

        def fem_solve(pressure):
            return 0.5 * np.ones_like(pressure)

        def bem_solve(velocity):
            return 0.2 * velocity

        initial_v = np.ones(5, dtype=complex)

        result = iterative_fem_bem_coupling(
            fem_solve, bem_solve, initial_v,
            max_iterations=5,
        )

        assert 'residual_history' in result
        assert len(result['residual_history']) == result['iterations']

    def test_respects_max_iterations(self):
        """Should stop at max_iterations if not converged."""

        def fem_solve(pressure):
            # Oscillating response that won't converge
            return pressure * np.exp(1j * 0.5)

        def bem_solve(velocity):
            return velocity * np.exp(1j * 0.3)

        initial_v = np.ones(5, dtype=complex)

        result = iterative_fem_bem_coupling(
            fem_solve, bem_solve, initial_v,
            max_iterations=3,
            tolerance=1e-10,  # Very tight tolerance
        )

        assert result['iterations'] == 3
        assert not result['converged']

    def test_under_relaxation_improves_stability(self):
        """Lower relaxation should help convergence for stiff problems."""

        # Problem that's unstable without relaxation
        def fem_solve(pressure):
            return 2.0 * np.ones_like(pressure) - 1.5 * pressure

        def bem_solve(velocity):
            return 0.8 * velocity

        initial_v = np.ones(5, dtype=complex)

        # With high relaxation (may not converge)
        result_high = iterative_fem_bem_coupling(
            fem_solve, bem_solve, initial_v,
            max_iterations=20,
            relaxation=0.9,
            tolerance=1e-4,
        )

        # With low relaxation (should be more stable)
        result_low = iterative_fem_bem_coupling(
            fem_solve, bem_solve, initial_v,
            max_iterations=50,
            relaxation=0.3,
            tolerance=1e-4,
        )

        # Low relaxation should converge or at least have smaller residual
        if not result_high['converged'] and result_low['converged']:
            # Low relaxation helped
            pass
        elif result_high['converged'] and result_low['converged']:
            # Both converged, check iterations
            pass
        else:
            # Check final residuals
            final_residual_high = result_high['residual_history'][-1]
            final_residual_low = result_low['residual_history'][-1]
            # Low relaxation should at least not be worse
            assert final_residual_low <= final_residual_high * 1.5


@pytest.mark.skipif(not BEMPP_AVAILABLE, reason="bempp-cl not available")
class TestBemppDependentFunctions:
    """Tests that require bempp-cl."""

    def test_get_element_areas(self):
        """Test element area computation."""
        from bempp_audio.fea.fem_bem_coupling import get_element_areas
        import bempp_cl.api

        # Create a simple grid (unit square split into 2 triangles)
        vertices = np.array([
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
        ], dtype=float)
        elements = np.array([
            [0, 0],
            [1, 2],
            [2, 3],
        ], dtype=np.uint32)
        grid = bempp_cl.api.Grid(vertices, elements)

        areas = get_element_areas(grid)

        assert len(areas) == 2
        # Each triangle has area 0.5
        assert np.allclose(areas, [0.5, 0.5])

    def test_get_element_centroids(self):
        """Test element centroid computation."""
        from bempp_audio.fea.fem_bem_coupling import get_element_centroids
        import bempp_cl.api

        # Single equilateral triangle
        h = np.sqrt(3) / 2
        vertices = np.array([
            [0, 1, 0.5],
            [0, 0, h],
            [0, 0, 0],
        ], dtype=float)
        elements = np.array([[0], [1], [2]], dtype=np.uint32)
        grid = bempp_cl.api.Grid(vertices, elements)

        centroids = get_element_centroids(grid)

        assert centroids.shape == (1, 3)
        expected = np.array([0.5, h / 3, 0])
        assert np.allclose(centroids[0], expected, atol=1e-10)

    def test_pressure_to_fem_surface_load(self):
        """Test pressure to FEM load conversion."""
        from bempp_audio.fea.fem_bem_coupling import (
            pressure_to_fem_surface_load,
            get_element_centroids,
        )
        import bempp_cl.api

        # Create simple grid
        vertices = np.array([
            [0, 1, 0.5],
            [0, 0, 1],
            [0, 0, 0],
        ], dtype=float)
        elements = np.array([[0], [1], [2]], dtype=np.uint32)
        grid = bempp_cl.api.Grid(vertices, elements)

        pressure = np.array([100.0 + 50j])  # Pa

        load, centroids = pressure_to_fem_surface_load(pressure, grid)

        assert len(load) == 1
        assert np.isclose(load[0], pressure[0])
        assert centroids.shape == (1, 3)

    def test_assess_coupling_strength(self):
        """Test coupling strength assessment."""
        from bempp_audio.fea.fem_bem_coupling import assess_coupling_strength
        import bempp_cl.api

        # Create a small disk-like grid
        # 6 triangles forming a hexagon
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        r = 0.01  # 1 cm radius
        outer_verts = np.array([r * np.cos(angles), r * np.sin(angles), np.zeros(6)])
        center = np.array([[0], [0], [0]])
        vertices = np.hstack([center, outer_verts])

        elements = np.array([
            [0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 1],
        ], dtype=np.uint32)
        grid = bempp_cl.api.Grid(vertices, elements)

        # Typical thin titanium dome parameters
        shell_mass_per_area = 4500 * 50e-6  # kg/m²
        shell_stiffness = 110e9 * (50e-6)**3 / (12 * (1 - 0.34**2)) / r**2

        result = assess_coupling_strength(
            grid,
            frequency_hz=10000,
            shell_mass_per_area=shell_mass_per_area,
            shell_stiffness=shell_stiffness,
        )

        assert 'coupling_ratio' in result
        assert 'ka' in result
        assert 'recommendation' in result
        assert result['recommendation'] in ['one-way', 'two-way']
        assert 'reason' in result
