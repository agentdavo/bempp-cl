from __future__ import annotations

import numpy as np
import pytest

from bempp_audio._optional import optional_import


def _minimal_mesh():
    bempp, ok = optional_import("bempp_cl.api")
    if not ok:
        pytest.skip("requires bempp_cl.api")

    from bempp_audio.mesh.loudspeaker_mesh import LoudspeakerMesh

    vertices = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    elements = np.array([[0], [1], [2]], dtype=int)
    grid = bempp.Grid(vertices, elements, domain_indices=np.array([1], dtype=int))
    return LoudspeakerMesh(grid)


def test_build_returns_request_without_mutating_builder_defaults():
    from bempp_audio import Loudspeaker
    from bempp_audio.api.request import SimulationRequest

    spk = Loudspeaker().from_mesh(_minimal_mesh())
    assert spk.state.velocity is None
    assert spk.state.frequencies is None

    req = spk.build(strict=False)
    assert isinstance(req, SimulationRequest)
    assert req.mesh is spk.state.mesh
    assert req.velocity is not None
    assert req.frequencies is not None
    assert spk.state.velocity is None
    assert spk.state.frequencies is None


def test_build_strict_requires_velocity_and_frequencies():
    from bempp_audio import Loudspeaker

    spk = Loudspeaker().from_mesh(_minimal_mesh())
    with pytest.raises(ValueError, match="Velocity not configured"):
        spk.build(strict=True)


def test_build_carries_execution_config():
    from bempp_audio import Loudspeaker
    from bempp_audio.config import SimulationConfig, ExecutionConfig

    spk = (
        Loudspeaker()
        .from_mesh(_minimal_mesh())
        .apply_config(SimulationConfig(execution=ExecutionConfig(n_workers=3, show_progress=False)))
        .velocity(mode="piston", amplitude=0.01)
    )
    req = spk.build(strict=True)
    assert req.execution_config is not None
    assert req.execution_config.n_workers == 3
    assert req.execution_config.show_progress is False

