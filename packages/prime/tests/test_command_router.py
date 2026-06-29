from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path

import prime_cli.command_registry as registry
import prime_cli.command_router as router
import pytest
from click.testing import CliRunner
from prime_cli.command_registry import COMMANDS, GROUPS, Command, command_map
from prime_cli.main import app
from pydantic import ValidationError
from pydantic_config import BaseConfig, cli


def _command_id(command: Command) -> str:
    return " ".join(command.path)


def test_registry_commands_resolve() -> None:
    """Every registered command declares a resolvable callback or Verifiers exec."""
    for command in COMMANDS:
        if command.verifiers is not None:
            assert command.run_attr is None, command.path
            assert command.config_attr is None, command.path
            continue
        module = importlib.import_module(command.module)
        assert hasattr(module, command.run_attr), command.path
        assert command.config_attr is not None, command.path
        config_model = getattr(module, command.config_attr)
        assert issubclass(config_model, BaseConfig), command.path
    # No duplicate command paths, and GROUPS exactly covers every non-root prefix.
    assert set(GROUPS) == {
        command.path[:index] for command in COMMANDS for index in range(1, len(command.path))
    }


def test_registry_has_no_duplicate_or_prefix_paths() -> None:
    paths = [command.path for command in COMMANDS]

    assert len(paths) == len(set(paths))
    assert not any(
        left != right and right[: len(left)] == left for left in paths for right in paths
    )


def test_command_map_rejects_duplicate_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "COMMANDS", (*COMMANDS, COMMANDS[0]))

    with pytest.raises(RuntimeError, match="duplicate Prime CLI command path"):
        command_map()


@pytest.mark.parametrize(
    "command",
    [command for command in COMMANDS if command.verifiers is None],
    ids=_command_id,
)
def test_command_contract(command: Command) -> None:
    module = importlib.import_module(command.module)
    callback = getattr(module, command.run_attr)
    parameter = next(iter(inspect.signature(callback).parameters.values()))

    assert len(inspect.signature(callback).parameters) == 1
    assert parameter.name == "config"

    config_model = getattr(module, command.config_attr)
    assert issubclass(config_model, BaseConfig)
    assert config_model.model_config["extra"] == "forbid"
    config_model.model_rebuild()
    with pytest.raises(ValidationError) as exc_info:
        config_model.model_validate({"definitely_unknown": True})
    assert any(error["type"] == "extra_forbidden" for error in exc_info.value.errors())


@pytest.mark.parametrize(
    "command",
    [command for command in COMMANDS if command.verifiers is None],
    ids=_command_id,
)
def test_leaf_help_smoke(command: Command) -> None:
    result = CliRunner().invoke(app, [*command.path, "--help"])

    assert result.exit_code == 0, result.output
    assert "usage:" in result.output.lower()


@pytest.mark.parametrize("group", GROUPS, ids=lambda group: " ".join(group))
def test_group_help_smoke(group: tuple[str, ...]) -> None:
    result = CliRunner().invoke(app, [*group, "--help"])

    assert result.exit_code == 0, result.output
    assert "COMMAND [ARGS]" in result.output


@pytest.mark.parametrize(
    "command",
    [command for command in COMMANDS if command.verifiers is not None],
    ids=_command_id,
)
def test_verifiers_commands_receive_untouched_argv(
    command: Command,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple] = []
    monkeypatch.setattr(
        router,
        "exec_verifiers_process",
        lambda name, args, plain=False: captured.append((name, list(args), plain)),
    )

    result = CliRunner().invoke(app, [*command.path, "target", "--unknown", "value"])

    assert result.exit_code == 0, result.output
    assert captured == [(command.verifiers, ["target", "--unknown", "value"], False)]


def test_toml_and_cli_parse_to_the_same_config(tmp_path: Path) -> None:
    from prime_cli.commands.wallet import WalletConfig

    path = tmp_path / "wallet.toml"
    path.write_text('limit = 7\noutput = "json"\n', encoding="utf-8")

    from_toml = cli(WalletConfig, args=["@", str(path)])
    from_cli = cli(WalletConfig, args=["--limit", "7", "--output", "json"])

    assert from_toml == from_cli


def test_root_options_are_consumed_before_leaf_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    selected: list[str] = []
    monkeypatch.delenv("PRIME_CONTEXT", raising=False)

    def select_context(value: str) -> None:
        selected.append(value)
        os.environ["PRIME_CONTEXT"] = value

    monkeypatch.setattr(router, "_select_context", select_context)

    result = CliRunner().invoke(
        app,
        ["--context", "staging", "--plain", "wallet", "--help"],
    )

    assert result.exit_code == 0, result.output
    assert selected == ["staging"]
    assert os.getenv("PRIME_CONTEXT") is None


def test_version_is_a_root_option() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.startswith("Prime CLI version:")


def test_prime_cli_source_does_not_import_typer() -> None:
    source_root = Path(__file__).parents[1] / "src" / "prime_cli"

    for path in source_root.rglob("*.py"):
        assert "typer" not in path.read_text(encoding="utf-8").lower(), path
