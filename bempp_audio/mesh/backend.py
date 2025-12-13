"""
Mesh backend utilities (Gmsh session + import/export helpers).

Goal: reduce duplicated Gmsh boilerplate and make mesh generation consistent.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import tempfile
from typing import Iterator, Optional

from bempp_audio._optional import require_bempp, require_gmsh
from bempp_audio.progress import get_logger


@contextmanager
def gmsh_session(model_name: str, terminal: int = 0) -> Iterator[None]:
    """
    Context manager that safely initializes/finalizes Gmsh.

    If Gmsh is already initialized, this is a no-op for initialization/finalize.
    """
    gmsh = require_gmsh()

    initialized_here = True
    if hasattr(gmsh, "isInitialized"):
        try:
            initialized_here = not bool(gmsh.isInitialized())
        except Exception:
            initialized_here = True

    if initialized_here:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", terminal)
        # Gmsh meshing can be slow for lofted OCC surfaces (e.g. morphed rectangular waveguides).
        # Enable multi-threading by default; allow override via env.
        try:
            env_threads = os.environ.get("BEMPPAUDIO_GMSH_THREADS")
            if env_threads is not None and str(env_threads).strip():
                n_threads = int(str(env_threads).strip())
            else:
                n_threads = int(min(8, os.cpu_count() or 1))
            if n_threads > 0:
                gmsh.option.setNumber("General.NumThreads", float(n_threads))
                gmsh.option.setNumber("Mesh.MaxNumThreads1D", float(n_threads))
                gmsh.option.setNumber("Mesh.MaxNumThreads2D", float(n_threads))
                gmsh.option.setNumber("Mesh.MaxNumThreads3D", float(n_threads))
                logger = get_logger()
                logger.debug(f"Gmsh threads: {n_threads}")
        except Exception:
            pass

    try:
        gmsh.model.add(model_name)
        yield
    finally:
        if initialized_here:
            gmsh.finalize()


def write_temp_msh() -> str:
    """Allocate a temporary `.msh` file in BEMPP's TMP_PATH."""
    bempp = require_bempp()
    fd, msh_name = tempfile.mkstemp(suffix=".msh", dir=bempp.TMP_PATH)
    os.close(fd)
    return msh_name


def import_msh_and_cleanup(msh_name: str):
    """Import a `.msh` into a BEMPP grid and remove the file."""
    bempp = require_bempp()
    grid = bempp.import_grid(msh_name)
    try:
        os.remove(msh_name)
    except OSError:
        pass
    return grid
