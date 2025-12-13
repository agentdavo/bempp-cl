"""
Runtime environment detection and conservative workarounds.

This module is intentionally lightweight and safe to import in any environment.
"""

from __future__ import annotations

import os
from pathlib import Path


def is_wsl() -> bool:
    """
    Best-effort WSL/WSL2 detection.

    Can be forced for tests via:
    - `BEMPPAUDIO_ASSUME_WSL=1` / `0`
    """
    forced = os.environ.get("BEMPPAUDIO_ASSUME_WSL")
    if forced is not None:
        return str(forced).strip() not in ("", "0", "false", "False", "no")

    try:
        txt = Path("/proc/sys/kernel/osrelease").read_text(errors="ignore").lower()
        if "microsoft" in txt or "wsl" in txt:
            return True
    except Exception:
        pass

    try:
        txt = Path("/proc/version").read_text(errors="ignore").lower()
        if "microsoft" in txt or "wsl" in txt:
            return True
    except Exception:
        pass

    return False


def multiprocessing_start_method() -> str | None:
    """
    Recommended multiprocessing start method for this runtime, or None.

    WSL environments can be more reliable with `spawn` than `fork` when using
    heavy numerical stacks and JIT backends.
    """
    override = os.environ.get("BEMPPAUDIO_MP_START")
    if override:
        return str(override).strip().lower()
    if is_wsl():
        return "spawn"
    return None

