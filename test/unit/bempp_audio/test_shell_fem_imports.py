from __future__ import annotations


def test_mindlin_reissner_plate_solver_symbol_importable_without_dolfinx_import():
    # This should not import/initialize dolfinx (MPI-sensitive) at import time.
    from bempp_audio.fea import MindlinReissnerPlateSolver, RayleighDamping, ModalResult, HarmonicResult

    assert MindlinReissnerPlateSolver is not None
    assert RayleighDamping(alpha=0.0, beta=0.0) is not None
    assert ModalResult is not None
    assert HarmonicResult is not None

