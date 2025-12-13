import numpy as np
import pytest


def test_performance_preset_sets_mesh_and_solver_defaults():
    from bempp_audio import Loudspeaker

    spk = Loudspeaker().performance_preset("ultra-fast", mode="horn")
    assert spk.state.mesh_preset == "ultra-fast"
    assert spk.state.solver_options.tol == pytest.approx(3e-4)
    assert spk.state.solver_options.maxiter == 400

    assert spk.state.frequencies is not None
    assert spk.state.frequencies.size == 20
    assert np.isclose(spk.state.frequencies[0], 200.0)
    assert np.isclose(spk.state.frequencies[-1], 20000.0)
    assert spk.state.polar_end == 90.0
    assert spk.state.norm_angle == 10.0


