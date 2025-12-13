from __future__ import annotations

import numpy as np
import pytest

from bempp_audio._optional import optional_import


def test_validate_reports_missing_fields():
    from bempp_audio import Loudspeaker

    spk = Loudspeaker()
    errors = spk.validate(strict=True)
    assert any("Mesh not configured" in e for e in errors)
    assert any("Velocity not configured" in e for e in errors)
    assert any("Frequencies not configured" in e for e in errors)


def test_snapshot_has_schema_and_version():
    from bempp_audio import Loudspeaker

    snap = Loudspeaker().to_snapshot()
    assert snap["schema"] == "bempp_audio.LoudspeakerSnapshot"
    assert snap["version"] == 1


def _mesh_with_domains(domain_ids: list[int]):
    bempp, ok = optional_import("bempp_cl.api")
    if not ok:
        pytest.skip("requires bempp_cl.api")

    from bempp_audio.mesh.loudspeaker_mesh import LoudspeakerMesh

    vertices = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    # Two triangles covering a square (0,0)-(1,1)
    elements = np.array([[0, 1], [1, 3], [2, 2]], dtype=int)
    grid = bempp.Grid(vertices, elements, domain_indices=np.asarray(domain_ids, dtype=int))
    return LoudspeakerMesh(grid)


def test_validate_domain_coverage_error_policy():
    bempp, ok = optional_import("bempp_cl.api")
    if not ok:
        pytest.skip("requires bempp_cl.api")

    from bempp_audio import Loudspeaker
    from bempp_audio.velocity import VelocityProfile

    mesh = _mesh_with_domains([1, 2])
    spk = Loudspeaker().from_mesh(mesh).default_bc("error").velocity_profile(
        VelocityProfile.by_domain({1: VelocityProfile.piston(0.01)})
    )
    errors = spk.validate(strict=True)
    assert any("requires explicit velocity assignment" in e for e in errors)


def test_snapshot_roundtrip_with_mesh_data():
    bempp, ok = optional_import("bempp_cl.api")
    if not ok:
        pytest.skip("requires bempp_cl.api")

    from bempp_audio import Loudspeaker

    mesh = _mesh_with_domains([1, 1])
    spk = Loudspeaker().from_mesh(mesh).frequency_range(100.0, 200.0, num=3).velocity(mode="piston", amplitude=0.01)
    snap = spk.to_snapshot(include_mesh_data=True)
    spk2 = Loudspeaker.from_snapshot(snap)

    assert spk2.state.frequencies is not None
    assert np.allclose(spk2.state.frequencies, spk.state.frequencies)
    assert spk2.state.velocity is not None
    assert spk2.state.mesh is not None
    assert np.allclose(spk2.state.mesh.to_dict()["vertices"], spk.state.mesh.to_dict()["vertices"])


def test_with_defaults_does_not_mutate_builder():
    from bempp_audio import Loudspeaker

    spk = Loudspeaker()
    spk2 = spk.with_defaults()

    assert spk.state.velocity is None
    assert spk.state.frequencies is None
    assert spk2.state.velocity is not None
    assert spk2.state.frequencies is not None


def test_snapshot_json_roundtrip(tmp_path):
    bempp, ok = optional_import("bempp_cl.api")
    if not ok:
        pytest.skip("requires bempp_cl.api")

    from bempp_audio import Loudspeaker

    mesh = _mesh_with_domains([1, 1])
    spk = Loudspeaker().from_mesh(mesh).frequency_range(100.0, 200.0, num=3).velocity(mode="piston", amplitude=0.01)
    path = tmp_path / "snapshot.json"
    spk.to_snapshot_json(path, include_mesh_data=True)
    spk2 = Loudspeaker.from_snapshot_json(path)

    assert spk2.state.frequencies is not None
    assert np.allclose(spk2.state.frequencies, spk.state.frequencies)
    assert spk2.state.mesh is not None
