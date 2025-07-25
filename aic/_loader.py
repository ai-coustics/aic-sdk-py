"""
Locate and dlopen the SDK shared library that ships inside the wheel.
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
    sysname = platform.system()
    try:
        libname, sub = _PLAT[sysname]
    except KeyError as exc:  # pragma: no cover
        raise RuntimeError(f"Unsupported OS: {sysname}") from exc

    pkg = f"{__package__}.libs.{sub}"
    with res.path(pkg, libname) as p:
        return p

def load() -> ctypes.CDLL:
    return ctypes.CDLL(str(_path()))
