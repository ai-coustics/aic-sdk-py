"""Locate and load the platform-specific SDK shared library.

This module resolves the correct shared object packaged within the wheel and
returns a loaded :class:`ctypes.CDLL` for use by the low-level bindings.
"""
from __future__ import annotations

import ctypes
import importlib.resources as res
import platform
from pathlib import Path

_PLAT = {
    "Linux":   ("libaic.so",   "linux"),
    "Darwin":  ("libaic.dylib","mac"),
    "Windows": ("aic.dll",     "windows"),
}

def _path() -> Path:
    """Return the filesystem path to the packaged SDK shared library.

    Returns
    -------
    pathlib.Path
        Absolute path to the platform-specific shared library within the package.

    Raises
    ------
    RuntimeError
        If the current operating system is not supported.
    """
    sysname = platform.system()
    try:
        libname, sub = _PLAT[sysname]
    except KeyError as exc:  # pragma: no cover
        raise RuntimeError(f"Unsupported OS: {sysname}") from exc

    pkg = f"{__package__}.libs.{sub}"
    with res.path(pkg, libname) as p:
        return p

def load() -> ctypes.CDLL:
    """Load and return the SDK shared library as a :class:`ctypes.CDLL`.

    Returns
    -------
    ctypes.CDLL
        Ready-to-use handle for calling C functions.
    """
    return ctypes.CDLL(str(_path()))
