import hashlib
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest
from prime_cli import __version__
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}

FAKE_INSTALLER = b"#!/bin/sh\nexit 0\n"


def trust_fake_installer(
    monkeypatch: pytest.MonkeyPatch,
    installer: bytes = FAKE_INSTALLER,
) -> None:
    monkeypatch.setattr(
        "prime_cli.commands.agent.PRIME_AGENT_INSTALLER_SHA256",
        hashlib.sha256(installer).hexdigest(),
    )


class FakeResponse:
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self, _size: int) -> bytes:
        return FAKE_INSTALLER


def test_agent_runs_installed_binary_with_passthrough_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        "prime_cli.commands.agent.shutil.which",
        lambda command: "/usr/local/bin/prime-agent" if command == "prime-agent" else None,
    )

    def fake_execv(executable: str, args: list[str]) -> None:
        calls.append((executable, args))

    monkeypatch.setattr("prime_cli.commands.agent.os.execv", fake_execv)

    result = runner.invoke(
        app,
        ["agent", "--model", "openai/gpt-5", "review this", "--help"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "/usr/local/bin/prime-agent",
            [
                "/usr/local/bin/prime-agent",
                "--model",
                "openai/gpt-5",
                "review this",
                "--help",
            ],
        )
    ]


def test_agent_installs_then_runs_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    agent_lookups = iter([None, "/new/bin/prime-agent"])
    installer_calls: list[list[str]] = []
    exec_calls: list[tuple[str, list[str]]] = []
    installer_paths: list[Path] = []
    download_requests: list[tuple[str, str | None, int]] = []

    def fake_which(command: str) -> str | None:
        if command == "prime-agent":
            return next(agent_lookups)
        if command == "sh":
            return "/bin/sh"
        return None

    def fake_run(command: list[str]) -> Any:
        installer_calls.append(command)
        if command[0] == "/bin/sh":
            installer_path = Path(command[1])
            installer_paths.append(installer_path)
            assert installer_path.read_bytes() == FAKE_INSTALLER
        return type("Result", (), {"returncode": 0})()

    trust_fake_installer(monkeypatch)
    monkeypatch.setattr("prime_cli.commands.agent.shutil.which", fake_which)

    def fake_urlopen(request: Request, timeout: int) -> FakeResponse:
        download_requests.append(
            (request.full_url, request.get_header("User-agent"), timeout)
        )
        return FakeResponse()

    monkeypatch.setattr("prime_cli.commands.agent.urlopen", fake_urlopen)
    monkeypatch.setattr("prime_cli.commands.agent.subprocess.run", fake_run)
    monkeypatch.setattr(
        "prime_cli.commands.agent.os.execv",
        lambda executable, args: exec_calls.append((executable, args)),
    )

    result = runner.invoke(app, ["agent", "agents"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Downloading the installer" in result.output
    assert installer_calls[0][0] == "/bin/sh"
    assert exec_calls == [
        ("/new/bin/prime-agent", ["/new/bin/prime-agent", "agents"])
    ]
    assert download_requests == [
        (
            "https://app.primeintellect.ai/prime-agent/install.sh",
            f"prime-cli/{__version__}",
            30,
        )
    ]
    assert installer_paths and not installer_paths[0].exists()


def test_agent_propagates_installer_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    trust_fake_installer(monkeypatch)
    monkeypatch.setattr(
        "prime_cli.commands.agent.shutil.which",
        lambda command: "/bin/sh" if command == "sh" else None,
    )
    monkeypatch.setattr(
        "prime_cli.commands.agent.urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    monkeypatch.setattr(
        "prime_cli.commands.agent.subprocess.run",
        lambda _command: type("Result", (), {"returncode": 23})(),
    )

    result = runner.invoke(app, ["agent"], env=TEST_ENV)

    assert result.exit_code == 23, result.output


def test_agent_finds_standalone_node_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standalone_agent = tmp_path / "prime-agent-node" / "current" / "bin" / "prime-agent"
    standalone_agent.parent.mkdir(parents=True)
    standalone_agent.touch()
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.commands.agent.shutil.which", lambda _command: None)
    monkeypatch.setattr(
        "prime_cli.commands.agent.os.execv",
        lambda executable, args: calls.append((executable, args)),
    )

    result = runner.invoke(app, ["agent", "doctor"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert calls == [
        (str(standalone_agent), [str(standalone_agent), "doctor"])
    ]


def test_agent_rejects_installer_checksum_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "prime_cli.commands.agent.shutil.which",
        lambda command: "/bin/sh" if command == "sh" else None,
    )
    monkeypatch.setattr(
        "prime_cli.commands.agent.urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    result = runner.invoke(app, ["agent"], env=TEST_ENV)

    assert result.exit_code == 1, result.output
    assert "installer checksum mismatch" in result.output


def test_agent_reports_invalid_installer(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_installer = b"not a shell script"

    class InvalidResponse(FakeResponse):
        def read(self, _size: int) -> bytes:
            return invalid_installer

    trust_fake_installer(monkeypatch, invalid_installer)
    monkeypatch.setattr(
        "prime_cli.commands.agent.shutil.which",
        lambda command: "/bin/sh" if command == "sh" else None,
    )
    monkeypatch.setattr(
        "prime_cli.commands.agent.urlopen", lambda *_args, **_kwargs: InvalidResponse()
    )

    result = runner.invoke(app, ["agent"], env=TEST_ENV)

    assert result.exit_code == 1, result.output
    assert "invalid installer response" in result.output


def test_agent_cleans_up_temp_file_when_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer_path = tmp_path / "prime-agent-install.sh"

    class FailingTempFile:
        name = str(installer_path)

        def __enter__(self) -> "FailingTempFile":
            installer_path.touch()
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def write(self, _installer: bytes) -> None:
            raise OSError("disk full")

    trust_fake_installer(monkeypatch)
    monkeypatch.setattr(
        "prime_cli.commands.agent.shutil.which",
        lambda command: "/bin/sh" if command == "sh" else None,
    )
    monkeypatch.setattr(
        "prime_cli.commands.agent.urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    monkeypatch.setattr(
        "prime_cli.commands.agent.tempfile.NamedTemporaryFile",
        lambda **_kwargs: FailingTempFile(),
    )

    result = runner.invoke(app, ["agent"], env=TEST_ENV)

    assert result.exit_code == 1, result.output
    assert "disk full" in result.output
    assert not installer_path.exists()
