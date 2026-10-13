from types import SimpleNamespace
from typing import Any

from prime_cli.commands import env as env_command
from typer.testing import CliRunner

runner = CliRunner()


def test_env_init_uses_verifiers_v1_cli(monkeypatch: Any) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(env_command.subprocess, "run", fake_run)

    result = runner.invoke(
        env_command.app,
        ["init", "my-env", "--path", "./environments", "--some-v1-option"],
    )

    assert result.exit_code == 0, result.output
    assert commands == [
        ["uv", "run", "vf-init", "my-env", "--path", "./environments", "--some-v1-option"]
    ]


def test_env_init_can_use_legacy_v0_initializer(monkeypatch: Any) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(env_command.subprocess, "run", fake_run)
    monkeypatch.setattr(
        env_command,
        "resolve_workspace_python",
        lambda: "/workspace/.venv/bin/python",
    )

    result = runner.invoke(env_command.app, ["init", "my-env", "--v0", "--path", "./envs"])

    assert result.exit_code == 0, result.output
    assert commands == [
        [
            "/workspace/.venv/bin/python",
            "-m",
            "verifiers.cli.commands.init",
            "my-env",
            "--path",
            "./envs",
        ]
    ]


def test_env_init_propagates_verifiers_exit_code(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        env_command.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=17),
    )

    result = runner.invoke(env_command.app, ["init", "my-env"])

    assert result.exit_code == 17


def test_env_init_forwards_help_to_verifiers(monkeypatch: Any) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(env_command.subprocess, "run", fake_run)

    result = runner.invoke(env_command.app, ["init", "--help"])

    assert result.exit_code == 0, result.output
    assert commands == [["uv", "run", "vf-init", "--help"]]
