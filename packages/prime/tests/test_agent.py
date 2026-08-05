from pathlib import Path
from typing import Any

import pytest
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}


class FakeResponse:
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self, _size: int) -> bytes:
        return b"#!/bin/sh\nexit 0\n"


def test_agent_runs_installed_binary_with_passthrough_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "prime_cli.commands.agent.shutil.which",
        lambda command: "/usr/local/bin/prime-agent" if command == "prime-agent" else None,
    )

    def fake_run(command: list[str]) -> Any:
        calls.append(command)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("prime_cli.commands.agent.subprocess.run", fake_run)

    result = runner.invoke(
        app,
        ["agent", "--model", "openai/gpt-5", "review this", "--help"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        [
            "/usr/local/bin/prime-agent",
            "--model",
            "openai/gpt-5",
            "review this",
            "--help",
        ]
    ]


def test_agent_installs_then_runs_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    agent_lookups = iter([None, "/new/bin/prime-agent"])
    calls: list[list[str]] = []
    installer_paths: list[Path] = []

    def fake_which(command: str) -> str | None:
        if command == "prime-agent":
            return next(agent_lookups)
        if command == "sh":
            return "/bin/sh"
        return None

    def fake_run(command: list[str]) -> Any:
        calls.append(command)
        if command[0] == "/bin/sh":
            installer_path = Path(command[1])
            installer_paths.append(installer_path)
            assert installer_path.read_bytes() == b"#!/bin/sh\nexit 0\n"
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("prime_cli.commands.agent.shutil.which", fake_which)
    monkeypatch.setattr(
        "prime_cli.commands.agent.urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    monkeypatch.setattr("prime_cli.commands.agent.subprocess.run", fake_run)

    result = runner.invoke(app, ["agent", "agents"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Downloading the installer" in result.output
    assert calls[0][0] == "/bin/sh"
    assert calls[1] == ["/new/bin/prime-agent", "agents"]
    assert installer_paths and not installer_paths[0].exists()


def test_agent_propagates_agent_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "prime_cli.commands.agent.shutil.which",
        lambda command: "/usr/local/bin/prime-agent" if command == "prime-agent" else None,
    )
    monkeypatch.setattr(
        "prime_cli.commands.agent.subprocess.run",
        lambda _command: type("Result", (), {"returncode": 17})(),
    )

    result = runner.invoke(app, ["agent", "doctor"], env=TEST_ENV)

    assert result.exit_code == 17, result.output


def test_agent_propagates_installer_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
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
    calls: list[list[str]] = []

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.commands.agent.shutil.which", lambda _command: None)
    monkeypatch.setattr(
        "prime_cli.commands.agent.subprocess.run",
        lambda command: calls.append(command) or type("Result", (), {"returncode": 0})(),
    )

    result = runner.invoke(app, ["agent", "doctor"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert calls == [[str(standalone_agent), "doctor"]]


def test_agent_reports_invalid_installer(monkeypatch: pytest.MonkeyPatch) -> None:
    class InvalidResponse(FakeResponse):
        def read(self, _size: int) -> bytes:
            return b"not a shell script"

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
