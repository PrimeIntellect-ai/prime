import json
from pathlib import Path

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
