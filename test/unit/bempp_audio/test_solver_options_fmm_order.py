from __future__ import annotations

from bempp_audio import Loudspeaker


def test_solver_options_sets_fmm_expansion_order():
    speaker = Loudspeaker().solver_options(use_fmm=True, fmm_expansion_order=7)
    assert speaker.state.solver_options.use_fmm is True
    assert speaker.state.solver_options.fmm_expansion_order == 7

