"""Volume put/get remote-path handling and SSH key hints (ENG-6451)."""

from types import SimpleNamespace

import pytest
import typer
from prime_cli.commands import volumes
from rich.console import Console


def _capture_console(monkeypatch) -> Console:
    console = Console(record=True, width=500, force_terminal=False)
    monkeypatch.setattr(volumes, "console", console)
    return console


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("datasets/my-data", "/volume/datasets/my-data"),
        ("/datasets/my-data", "/volume/datasets/my-data"),
        ("datasets/my-data/", "/volume/datasets/my-data/"),
        ("a/b/c.bin", "/volume/a/b/c.bin"),
        ("", "/volume/"),
        ("/", "/volume/"),
    ],
)
def test_remote_path_is_volume_relative(given, expected):
    assert volumes._remote_path(given) == expected


@pytest.mark.parametrize(
    "given",
    [
        "/volume/datasets/my-data",
        "volume/datasets/my-data",
        "volume/datasets/my-data/",
        "/volume",
        "volume/",
    ],
)
def test_remote_path_rejects_in_session_mount_path(given, monkeypatch):
    console = _capture_console(monkeypatch)
    with pytest.raises(typer.Exit) as exit_info:
        volumes._remote_path(given)
    assert exit_info.value.exit_code == 2
    text = console.export_text()
    assert "remote paths are relative to the volume root" in text
    assert "`prime` prepends /volume/ itself" in text
    assert "Did you mean" in text


def test_transfer_prints_key_and_remote_hint(monkeypatch, tmp_path):
    console = _capture_console(monkeypatch)
    session = SimpleNamespace(read_only=False)
    key = tmp_path / "id_rsa"
    monkeypatch.setattr(
        volumes,
        "_open_session",
        lambda *args, **kwargs: (session, "vol-ssh-abc", str(key), tmp_path / "cfg", False),
    )
    monkeypatch.setattr(
        volumes.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in ("ssh", "rsync") else None,
    )
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=23)

    monkeypatch.setattr(volumes.subprocess, "run", run)

    with pytest.raises(typer.Exit) as exit_info:
        volumes._transfer("my-vol", False, "datasets/", "./my-data", upload=True, direct=False)

    assert exit_info.value.exit_code == 23
    assert commands[0][-1] == "vol-ssh-abc:/volume/datasets/"
    text = console.export_text()
    assert f"Using SSH key: {key}" in text
    assert "prime config set-ssh-key-path" in text
    assert "Transfer failed." in text
    assert "primary key on the dashboard" in text
    assert "Tokens → SSH Keys" in text


def test_transfer_of_get_uses_volume_relative_path(monkeypatch, tmp_path):
    _capture_console(monkeypatch)  # silence the "Using SSH key" print
    session = SimpleNamespace(read_only=True)
    monkeypatch.setattr(
        volumes,
        "_open_session",
        lambda *args, **kwargs: (
            session,
            "vol-ssh-abc",
            str(tmp_path / "id_rsa"),
            tmp_path / "cfg",
            True,
        ),
    )
    monkeypatch.setattr(
        volumes.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in ("ssh", "rsync") else None,
    )
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(volumes.subprocess, "run", run)

    volumes._transfer("my-vol", True, "datasets/my-data", ".", upload=False, direct=False)

    assert commands[0][0].endswith("rsync")
    assert "vol-ssh-abc:/volume/datasets/my-data" in commands[0]
