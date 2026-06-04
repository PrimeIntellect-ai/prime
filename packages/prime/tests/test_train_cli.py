import json
from pathlib import Path
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
    assert "request" in result.output


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


def test_train_help_lists_cluster_flag() -> None:
    # The flag has to be discoverable from `--help` so users don't have
    # to grep source to find it. Regression guard against future arg
    # reorders silently hiding the option.
    result = runner.invoke(app, ["train", "--help"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "--cluster" in result.output


def test_train_cluster_flag_overrides_config_cluster_name(monkeypatch, tmp_path: Path) -> None:
    # CLI `--cluster` should win over `cluster_name = "..."` in the TOML
    # so users can retarget a single dispatch without editing the config.
    # We don't validate the cluster name client-side: the backend's picker
    # is the source of truth — verify here only that the override reaches
    # the RLClient payload as `cluster_name`.
    captured: dict[str, Any] = {}

    def mock_create_run(self: Any, **kwargs: Any) -> Any:
        captured["kwargs"] = kwargs

        class _Run:
            id = "run-1"
            status = "QUEUED"
            runs_ahead = None

            def model_dump(self_inner) -> dict[str, Any]:
                return {"id": "run-1", "status": "QUEUED"}

        return _Run()

    def mock_list_models(self: Any, **kwargs: Any) -> list:
        return []

    monkeypatch.setattr(
        "prime_cli.api.rl.RLClient.create_run",
        mock_create_run,
    )
    monkeypatch.setattr(
        "prime_cli.api.rl.RLClient.list_models",
        mock_list_models,
    )

    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "Qwen/Qwen3-0.6B"\n'
        'cluster_name = "config-cluster"\n'
        "\n"
        "[[env]]\n"
        'id = "reverse-text"\n'
    )

    result = runner.invoke(
        app,
        [
            "train",
            str(config_path),
            "--cluster",
            "flag-cluster",
            "--output",
            "json",
            "--yes",
            "--skip-action-check",
        ],
        env={**TEST_ENV, "PRIME_API_KEY": "test-key"},
    )

    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["cluster_name"] == "flag-cluster"


def test_train_without_cluster_flag_uses_config_cluster_name(monkeypatch, tmp_path: Path) -> None:
    # Sanity check the inverse: with no --cluster, the TOML's
    # cluster_name is what reaches the backend. Without this we'd never
    # know if the override path silently took over the no-override path.
    captured: dict[str, Any] = {}

    def mock_create_run(self: Any, **kwargs: Any) -> Any:
        captured["kwargs"] = kwargs

        class _Run:
            id = "run-1"
            status = "QUEUED"
            runs_ahead = None

            def model_dump(self_inner) -> dict[str, Any]:
                return {"id": "run-1", "status": "QUEUED"}

        return _Run()

    def mock_list_models(self: Any, **kwargs: Any) -> list:
        return []

    monkeypatch.setattr(
        "prime_cli.api.rl.RLClient.create_run",
        mock_create_run,
    )
    monkeypatch.setattr(
        "prime_cli.api.rl.RLClient.list_models",
        mock_list_models,
    )

    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "Qwen/Qwen3-0.6B"\n'
        'cluster_name = "config-cluster"\n'
        "\n"
        "[[env]]\n"
        'id = "reverse-text"\n'
    )

    result = runner.invoke(
        app,
        [
            "train",
            str(config_path),
            "--output",
            "json",
            "--yes",
            "--skip-action-check",
        ],
        env={**TEST_ENV, "PRIME_API_KEY": "test-key"},
    )

    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["cluster_name"] == "config-cluster"


def test_train_request_submits_model_request(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def mock_post(self: Any, endpoint: str, json: dict[str, Any] | None = None) -> dict:
        captured["endpoint"] = endpoint
        captured["json"] = json
        return {"message": "Request submitted"}

    monkeypatch.setattr("prime_cli.client.APIClient.post", mock_post)

    result = runner.invoke(
        app,
        ["train", "request"],
        input="openai/gpt-oss-120b, meta-llama/Llama-4\nSFT distillation\n",
        env={**TEST_ENV, "PRIME_API_KEY": "test-key"},
    )

    assert result.exit_code == 0, result.output
    assert "Request submitted" in result.output
    assert captured["endpoint"] == "/feedback"
    payload = captured["json"]
    assert payload["category"] == "feature"
    assert payload["run_id"] is None
    assert payload["cli_version"]
    assert payload["message"] == (
        "Hosted Training model request\n\n"
        "Models:\nopenai/gpt-oss-120b, meta-llama/Llama-4\n\n"
        "Context:\nSFT distillation"
    )
