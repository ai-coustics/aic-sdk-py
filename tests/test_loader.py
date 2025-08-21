from pathlib import Path

import pytest


class _DummyCtx:
    def __init__(self, path: Path):
        self._path = path

    def __enter__(self):
        return self._path

    def __exit__(self, exc_type, exc, tb):
        return False


def _stub_res_path(pkg: str, libname: str):
    # derive subdir name from package dotted path
    if ".linux" in pkg:
        sub = "linux"
    elif ".mac" in pkg:
        sub = "mac"
    elif ".windows" in pkg:
        sub = "windows"
    else:
        sub = "unknown"
    return _DummyCtx(Path(f"/tmp/{sub}/{libname}"))


def test_loader_path_for_platform(monkeypatch):
    import aic._loader as loader

    monkeypatch.setattr(loader.res, "path", _stub_res_path)

    # Linux
    monkeypatch.setattr(loader.platform, "system", lambda: "Linux")
    p = loader._path()
    assert isinstance(p, Path)
    assert "linux" in str(p)
    assert p.name.endswith(".so")

    # macOS
    monkeypatch.setattr(loader.platform, "system", lambda: "Darwin")
    p = loader._path()
    assert "mac" in str(p)
    assert p.name.endswith(".dylib")

    # Windows
    monkeypatch.setattr(loader.platform, "system", lambda: "Windows")
    p = loader._path()
    assert "windows" in str(p)
    assert p.name.endswith(".dll")


def test_loader_unsupported_os(monkeypatch):
    import aic._loader as loader

    monkeypatch.setattr(loader.res, "path", _stub_res_path)
    monkeypatch.setattr(loader.platform, "system", lambda: "Plan9")
    with pytest.raises(RuntimeError):
        loader._path()
