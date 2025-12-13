from __future__ import annotations

import os
import numpy as np
import pytest

from bempp_audio._optional import optional_import


@pytest.mark.skipif(os.environ.get("BEMPPAUDIO_RUN_SLOW") != "1", reason="set BEMPPAUDIO_RUN_SLOW=1 to enable")
def test_piston_one_frequency_pressure_at_point():
    gmsh, has_gmsh = optional_import("gmsh")
    bempp, has_bempp = optional_import("bempp_cl.api")
    if not has_gmsh or not has_bempp:
        pytest.skip("requires gmsh and bempp_cl")

    from bempp_audio import Loudspeaker

    response = (
        Loudspeaker()
        .circular_piston(radius=0.02, element_size=0.01)
        .free_space()
        .single_frequency(1000.0)
        .solve()
    )

    result = response.results[0]
    p = result.pressure_at(np.array([[0.0], [0.0], [1.0]]))
    assert p.shape == (1,)
    assert np.isfinite(p.real).all()
    assert np.isfinite(p.imag).all()

