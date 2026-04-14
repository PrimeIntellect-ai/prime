import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {"PRIME_DISABLE_VERSION_CHECK": "1"}


def test_train_help_promotes_config_run_path() -> None:
    result = runner.invoke(app, ["train", "--help"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "prime train [OPTIONS] CONFIG_PATH [ARGS]... | COMMAND [ARGS]..." in result.output
    assert "Launch and manage Hosted Training runs." in result.output
    assert "Path to a TOML config file to launch as a" in result.output
    assert "Hosted Training run." in result.output
    assert "logs" in result.output


def test_rl_alias_is_hidden_from_root_help() -> None:
    result = runner.invoke(app, ["--help"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Launch and manage Hosted Training runs." in result.output
    assert "Deprecated alias for `prime train`." not in result.output


def test_rl_alias_still_works_with_deprecation_warning(tmp_path: Path) -> None:
    output_path = tmp_path / "config.toml"

    result = runner.invoke(app, ["rl", "init", str(output_path)], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert (
        "[DEPRECATED] The 'rl' command is deprecated. Use 'prime train' instead."
    ) in result.output
    assert "Run with: prime train" in result.output
    assert output_path.exists()


def test_rl_alias_warning_uses_stderr_for_json_output() -> None:
    result = runner.invoke(app, ["rl", "configs", "--output", "json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert (
        "[DEPRECATED] The 'rl' command is deprecated. Use 'prime train' instead."
    ) in result.stderr
    assert "[DEPRECATED]" not in result.stdout
    data = json.loads(result.stdout)
    assert "configs" in data


def test_train_init_defaults_to_rl_toml() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["train", "init"], env=TEST_ENV)

        assert result.exit_code == 0, result.output
        assert "Created rl.toml" in result.output
        assert "Run with: prime train rl.toml" in result.output
        assert Path("rl.toml").exists()


def test_train_sft_config_forwards_teacher_payload_and_summary(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "sft.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\n'
        'loss = "sft"\n'
        "max_steps = 2\n"
        "[teacher]\n"
        'model = "openai/gpt-oss-120b"\n'
        "save = false\n"
        "[teacher.sampling]\n"
        "max_tokens = 2048\n"
        'reasoning_effort = "medium"\n'
        "[[env]]\n"
        'id = "primeintellect/wordle"\n'
        "[run_config]\n"
        'note = "cursor"\n'
    )
    captured: dict[str, Any] = {}

    class DummyRLClient:
        def __init__(self, api_client: object) -> None:
            pass

        def list_models(self, team_id: str | None = None) -> list[Any]:
            return []

        def create_run(self, **kwargs: Any) -> SimpleNamespace:
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                id="rft_test",
                status="RUNNING",
                runs_ahead=None,
                queue_reason=None,
            )

    class DummyConfig:
        team_id = None
        frontend_url = "https://app.primeintellect.ai"

    monkeypatch.setattr("prime_cli.commands.rl.APIClient", lambda: object())
    monkeypatch.setattr("prime_cli.commands.rl.Config", DummyConfig)
    monkeypatch.setattr("prime_cli.commands.rl.RLClient", DummyRLClient)

    result = runner.invoke(
        app,
        ["train", str(config_path), "--yes", "--skip-action-check"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["loss"] == "sft"
    assert captured["kwargs"]["rollouts_per_example"] == 1
    assert captured["kwargs"]["teacher"] == {
        "model": "openai/gpt-oss-120b",
        "save": False,
        "sampling": {
            "max_tokens": 2048,
            "reasoning_effort": "medium",
        },
    }
    assert "Loss:                sft" in result.output
    assert "Teacher" in result.output
    assert "Model:            openai/gpt-oss-120b" in result.output
    assert "Reasoning Effort: medium" in result.output
    assert "Run Config" in result.output
    assert "Values: {'note': 'cursor'}" in result.output
