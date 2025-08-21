import os
import platform
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py


def _read_version_from_pyproject(project_root: Path) -> str:
    pyproject = project_root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    # Very small parser: look for a line like: version = "1.2.3"
    match = re.search(r"^version\s*=\s*\"([^\"]+)\"", text, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not determine version from pyproject.toml")
    return match.group(1)


def _read_sdk_version_from_pyproject(project_root: Path) -> str | None:
    """Return [tool.aic-sdk].sdk-version if present, else None."""
    pyproject = project_root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    # Look for: [tool.aic-sdk] ... sdk-version = "X.Y.Z"
    # Keep it simple and robust without adding a TOML parser.
    tool_block = re.search(r"(?ms)^\[tool\.aic-sdk\]\s+(.*?)(?=^\[|\Z)", text)
    if not tool_block:
        return None
    match = re.search(r'(?m)^\s*sdk-version\s*=\s*"([^"]+)"', tool_block.group(1))
    return match.group(1) if match else None


def _detect_platform_triplet() -> tuple[str, str, str]:
    """Return (os_key, asset_arch, asset_ext).

    os_key is one of: linux, mac, windows (used for destination path)
    asset_arch is the architecture string used in the SDK asset filenames
    asset_ext is either .tar.gz or .zip
    """
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Linux":
        os_key = "linux"
        if machine in ("x86_64", "amd64"):
            arch = "x86_64-unknown-linux-gnu"
        elif machine in ("aarch64", "arm64"):
            arch = "aarch64-unknown-linux-gnu"
        else:
            raise RuntimeError(f"Unsupported Linux architecture: {machine}")
        return os_key, arch, ".tar.gz"

    if system == "Darwin":
        os_key = "mac"
        if machine in ("arm64",):
            arch = "aarch64-apple-darwin"
        elif machine in ("x86_64",):
            arch = "x86_64-apple-darwin"
        else:
            raise RuntimeError(f"Unsupported macOS architecture: {machine}")
        return os_key, arch, ".tar.gz"

    if system == "Windows":
        os_key = "windows"
        # We only ship 64-bit for Windows
        arch = "x86_64-pc-windows-msvc"
        return os_key, arch, ".zip"

    raise RuntimeError(f"Unsupported OS: {system}")


def _compute_asset_name(version: str, arch: str, ext: str) -> str:
    return f"aic-sdk-{arch}-{version}{ext}"


def _download_bytes(url: str) -> bytes:
    with urlopen(url) as resp:  # nosec - URL is static to GitHub Releases
        if resp.status != 200:
            raise RuntimeError(f"Failed to download {url}: HTTP {resp.status}")
        return resp.read()


def _extract_sdk_archive(archive_bytes: bytes, ext: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        tmp.write(archive_bytes)
        tmp.flush()
        tmp.close()
        archive_path = Path(tmp.name)
        if ext == ".zip":
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(dest_dir)
        else:
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(dest_dir)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _install_shared_libs(extracted_root: Path, os_key: str, build_lib: Path) -> None:
    # Assets place binaries under a 'lib' directory
    lib_dir = extracted_root / "lib"
    if not lib_dir.exists():
        # Fallback: sometimes archives might not have a top-level lib/
        lib_dir = extracted_root
    dest = build_lib / "aic" / "libs" / os_key
    dest.mkdir(parents=True, exist_ok=True)

    patterns = {
        "linux": ["libaic*.so", "libaic*.so.*"],
        "mac": ["libaic*.dylib"],
        "windows": ["aic.dll", "libaic*.dll", "*.dll"],
    }[os_key]

    copied_any = False
    for pattern in patterns:
        for src in lib_dir.glob(pattern):
            shutil.copy2(src, dest / src.name)
            copied_any = True

    if not copied_any:
        raise RuntimeError(f"No SDK shared libraries found in {lib_dir} for platform {os_key}")

    # Ensure canonical filenames expected by loader exist
    if os_key == "linux":
        target = dest / "libaic.so"
        if not target.exists():
            # pick a suitable source (prefer exact .so, then highest-versioned)
            candidates = sorted(dest.glob("libaic*.so*"))
            if candidates:
                shutil.copy2(candidates[0], target)
    elif os_key == "mac":
        target = dest / "libaic.dylib"
        if not target.exists():
            candidates = sorted(dest.glob("libaic*.dylib"))
            if candidates:
                shutil.copy2(candidates[0], target)
    else:  # windows
        target = dest / "aic.dll"
        if not target.exists():
            candidates = sorted(dest.glob("*.dll"))
            if candidates:
                shutil.copy2(candidates[0], target)


class build_py(_build_py):
    """Custom build step that downloads SDK binaries at build/install time.

    This keeps PyPI artifacts small (sdist only) while ensuring end-users get
    the correct native libraries when installing from source.
    """

    def run(self):
        super().run()

        # Allow disabling via env for CI/source-only builds
        if os.environ.get("AIC_SDK_SKIP_DOWNLOAD", "").lower() in {"1", "true", "yes"}:
            print("[aic-sdk] Skipping SDK binary download due to AIC_SDK_SKIP_DOWNLOAD=1")
            return

        project_root = Path(__file__).parent.resolve()
        try:
            pkg_version = _read_version_from_pyproject(project_root)
            sdk_version = _read_sdk_version_from_pyproject(project_root) or pkg_version

            # Use the C library version for asset filename; strip any .post suffixes
            asset_version = re.sub(r"\.post\d+\Z", "", sdk_version)
            os_key, arch, ext = _detect_platform_triplet()
            asset_name = _compute_asset_name(asset_version, arch, ext)

            # Keep release tag tied to Python package version (upload assets there)
            base_url = f"https://github.com/ai-coustics/aic-sdk-py/releases/download/{pkg_version}"
            url = f"{base_url}/{asset_name}"

            print(f"[aic-sdk] Using SDK version: {sdk_version} (asset {asset_name})")
            print(f"[aic-sdk] Downloading SDK asset from {base_url}")
            data = _download_bytes(url)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                _extract_sdk_archive(data, ext, tmp_path)
                self.copy_tree(str(tmp_path), str(tmp_path))  # ensure content on disk
                _install_shared_libs(tmp_path, os_key, Path(self.build_lib))

            # Ensure downloaded binaries are packaged in any locally built wheel
            libs_root = Path(self.build_lib) / "aic" / "libs" / os_key
            if libs_root.exists():
                binaries = [
                    str(p.relative_to(Path(self.build_lib) / "aic")) for p in libs_root.glob("**/*") if p.is_file()
                ]
                self.distribution.package_data = self.distribution.package_data or {}
                pkg_list = self.distribution.package_data.get("aic", [])
                pkg_list.extend(binaries)
                # de-dupe while preserving order
                seen = set()
                pkg_list_unique = []
                for item in pkg_list:
                    if item not in seen:
                        seen.add(item)
                        pkg_list_unique.append(item)
                self.distribution.package_data["aic"] = pkg_list_unique

            print("[aic-sdk] SDK binaries installed into package tree")
        except Exception as exc:
            # Provide actionable error pointing users to the release page
            msg = (
                f"[aic-sdk] ERROR while fetching SDK binaries: {exc}\n"
                f"Ensure that a release exists at https://github.com/ai-coustics/aic-sdk-py/releases/tag/{sdk_version} "
                f"with the appropriate platform asset. You can bypass download by setting AIC_SDK_SKIP_DOWNLOAD=1."
            )
            raise RuntimeError(msg) from exc


setup(
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    has_ext_modules=lambda: True,
    cmdclass={
        "build_py": build_py,
    },
)
