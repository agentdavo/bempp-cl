import pytest


def test_loudspeaker_solver_preset_sets_options():
    from bempp_audio import Loudspeaker

    spk = Loudspeaker().solver_preset("ultra-fast")
    assert spk.state.solver_options.tol == pytest.approx(3e-4)
    assert spk.state.solver_options.maxiter == 400

    spk2 = Loudspeaker().solver_preset("ultra_fast")
    assert spk2.state.solver_options.tol == pytest.approx(3e-4)

    spk3 = Loudspeaker().solver_preset("Ultra Fast")
    assert spk3.state.solver_options.tol == pytest.approx(3e-4)

