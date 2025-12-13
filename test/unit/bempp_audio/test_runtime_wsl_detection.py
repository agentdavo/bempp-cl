def test_multiprocessing_start_method_can_be_forced(monkeypatch):
    from bempp_audio.runtime import multiprocessing_start_method, is_wsl

    monkeypatch.setenv("BEMPPAUDIO_ASSUME_WSL", "1")
    assert is_wsl() is True
    assert multiprocessing_start_method() == "spawn"

    monkeypatch.setenv("BEMPPAUDIO_MP_START", "forkserver")
    assert multiprocessing_start_method() == "forkserver"

