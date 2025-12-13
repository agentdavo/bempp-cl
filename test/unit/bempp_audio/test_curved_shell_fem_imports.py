from __future__ import annotations


def test_curved_shell_solver_import_does_not_require_dolfinx():
    # Import should not trigger MPI init (dolfinx/petsc4py) in environments where those are problematic.
    import bempp_audio.fea.curved_shell_fem as m

    assert hasattr(m, "SphericalCapMindlinReissnerShellSolver")

