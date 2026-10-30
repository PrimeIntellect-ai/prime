"""Real uv resolution against local wheels: no network or installation required."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
from prime_cli.commands.env import _build_install_command

pytestmark = pytest.mark.skipif(not shutil.which("uv"), reason="uv is required")


def _wheel(index: Path, name: str, version: str, requires: str = "") -> str:
    directory = index / name.replace("_", "-")
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{name}-{version}-py3-none-any.whl"
    wheel = directory / filename
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    if requires:
        metadata += f"Requires-Dist: {requires}\n"
    with ZipFile(wheel, "w") as archive:
        dist_info = f"{name}-{version}.dist-info"
        archive.writestr(f"{dist_info}/METADATA", metadata + "\n")
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    with (directory / "index.html").open("a") as index_file:
        index_file.write(f'<a href="{filename}">{filename}</a>\n')
    return wheel.as_uri()


@pytest.mark.parametrize(
    ("version", "prerelease", "expected"),
    [("latest", False, "0.2.0"), ("0.1.0", False, "0.1.0"), ("latest", True, "0.3.0rc1")],
)
def test_scoped_index_preserves_selection_and_dependency_sources(
    tmp_path, monkeypatch, version, prerelease, expected
):
    scoped = tmp_path / "scoped"
    default = tmp_path / "default"
    older_wheel = _wheel(scoped, "deep_swe", "0.1.0", "harbor==0.21.0")
    _wheel(scoped, "deep_swe", "0.2.0", "harbor==0.21.0")
    _wheel(scoped, "deep_swe", "0.3.0rc1", "harbor==0.21.0")
    _wheel(default, "harbor", "0.21.0")
    # An unrelated distribution on the default index must not replace the Hub env.
    _wheel(default, "deep_swe", "99.0.0")
    monkeypatch.setattr("prime_cli.commands.env.resolve_workspace_python", lambda: sys.executable)
    command = _build_install_command(
        "deep-swe", version, scoped.as_uri(), older_wheel, prerelease=prerelease
    )
    assert command is not None
    command += ["--dry-run", "--no-config", "--index-url", default.as_uri()]
    environment = {k: v for k, v in os.environ.items() if not k.startswith("UV_")}
    environment["UV_CACHE_DIR"] = str(tmp_path / "cache")
    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=30)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f"deep-swe=={expected}" in output
    assert "harbor==0.21.0" in output

    # The legacy owner index exposes a different Hub environment named harbor.
    # Reintroducing it reproduces the original resolver failure.
    _wheel(scoped, "harbor", "0.1.5")
    result = subprocess.run(
        command + ["--refresh"], capture_output=True, text=True, env=environment, timeout=30
    )
    assert result.returncode != 0
    assert "harbor==0.21.0" in result.stderr
