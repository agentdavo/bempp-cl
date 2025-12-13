"""
Optional dependency helpers.

This module centralizes "optional import" patterns so that importing
`bempp_audio` does not require heavy visualization/meshing dependencies
unless the corresponding functionality is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class UnavailableDependency:
    """
    Placeholder object for an optional dependency that is not installed.

    Accessing attributes or calling the object raises an ImportError that
    includes an installation hint.
    """

    name: str
    install_hint: str
    original_error: Optional[BaseException] = None

    def _raise(self) -> None:
        msg = f"Optional dependency '{self.name}' is required. {self.install_hint}"
        if self.original_error is not None:
            raise ImportError(msg) from self.original_error
        raise ImportError(msg)

    def __getattr__(self, _attr: str) -> Any:  # pragma: no cover
        self._raise()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        self._raise()


def optional_import(module: str) -> Tuple[Optional[Any], bool]:
    """Best-effort import: returns (module_or_none, available)."""
    try:
        return importlib.import_module(module), True
    except ImportError:
        return None, False


def require_module(module: str, install_hint: str) -> Any:
    """Import a required module or raise with a helpful installation hint."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(f"Required dependency '{module}' is not available. {install_hint}") from e


def require_bempp() -> Any:
    return require_module("bempp_cl.api", "Install bempp-cl (this repository) and ensure it is importable.")


def require_gmsh() -> Any:
    return require_module("gmsh", "Install with `pip install gmsh`.")


def require_meshio() -> Any:
    return require_module("meshio", "Install with `pip install meshio`.")


def require_matplotlib() -> Any:
    return require_module("matplotlib", "Install with `pip install matplotlib` or `pip install bempp-cl[audio]`.")


def require_plotly() -> Any:
    return require_module("plotly", "Install with `pip install plotly` or `pip install bempp-cl[audio]`.")

