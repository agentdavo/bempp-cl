from __future__ import annotations

from bempp_audio.mesh.validation import MeshResolutionPresets


def test_mesh_resolution_presets_have_expected_keys():
    for name in ("ultra_fast", "super_fast", "fast", "standard", "slow"):
        d = MeshResolutionPresets.get_preset(name)
        for k in ("h_throat", "h_mouth", "h_box", "h_baffle", "h_sides", "h_back"):
            assert k in d
            assert float(d[k]) > 0
        for k in ("n_axial_slices", "n_circumferential", "corner_resolution"):
            assert k in d
            assert int(d[k]) >= 0


def test_mesh_resolution_presets_accept_dash_aliases():
    a = MeshResolutionPresets.get_preset("super_fast")
    b = MeshResolutionPresets.get_preset("super-fast")
    # Compare a few representative keys rather than relying on exact dict equality.
    for k in ("h_throat", "h_mouth", "h_box", "n_axial_slices", "n_circumferential", "corner_resolution"):
        assert a[k] == b[k]

    a2 = MeshResolutionPresets.get_preset("ultra_fast")
    b2 = MeshResolutionPresets.get_preset("ultra-fast")
    for k in ("h_throat", "h_mouth", "h_box", "n_axial_slices", "n_circumferential", "corner_resolution"):
        assert a2[k] == b2[k]


def test_mesh_resolution_presets_monotone_fineness():
    # Finer presets should have smaller characteristic sizes.
    uf = MeshResolutionPresets.get_preset("ultra_fast")
    sf = MeshResolutionPresets.get_preset("super_fast")
    f = MeshResolutionPresets.get_preset("fast")
    st = MeshResolutionPresets.get_preset("standard")
    sl = MeshResolutionPresets.get_preset("slow")

    assert uf["h_throat"] >= sf["h_throat"] >= f["h_throat"] >= st["h_throat"] >= sl["h_throat"]
    assert uf["h_mouth"] >= sf["h_mouth"] >= f["h_mouth"] >= st["h_mouth"] >= sl["h_mouth"]

    # Finer presets should generally use at least as many axial/circumferential slices.
    assert sf["n_axial_slices"] <= f["n_axial_slices"] <= st["n_axial_slices"] <= sl["n_axial_slices"]
    assert sf["n_circumferential"] <= f["n_circumferential"] <= st["n_circumferential"] <= sl["n_circumferential"]
