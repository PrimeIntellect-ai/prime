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


def _fake_run_payload(status: str, run_id: str = "run-1") -> dict[str, Any]:
    return {
        "id": run_id,
        "userId": "user-1",
        "status": status,
        "createdAt": "2026-08-25T00:00:00Z",
        "updatedAt": "2026-08-25T00:00:00Z",
    }


def test_train_stop_returns_immediately_when_already_terminal(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def mock_request(self, method, endpoint, params=None, json=None, timeout=None):
        calls.append((method, endpoint))
        return {"run": _fake_run_payload("STOPPED")}

    monkeypatch.setattr("prime_cli.core.client.APIClient.request", mock_request)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    result = runner.invoke(
        app,
        ["train", "stop", "run-1", "--force"],
        env={**TEST_ENV, "PRIME_API_KEY": "test-key"},
    )

    assert result.exit_code == 0, result.output
    assert "stopped successfully" in result.output
    assert calls == [("PUT", "/rft/runs/run-1/stop")], (
        "a stop that already returns a terminal status should not poll at all"
    )


def test_train_stop_polls_until_terminal(monkeypatch) -> None:
    statuses = iter(["RUNNING", "RUNNING", "STOPPED"])
    calls: list[tuple[str, str]] = []

    def mock_request(self, method, endpoint, params=None, json=None, timeout=None):
        calls.append((method, endpoint))
        return {"run": _fake_run_payload(next(statuses))}

    monkeypatch.setattr("prime_cli.core.client.APIClient.request", mock_request)
    # Poll loop now runs against a real monotonic deadline (so a stalled
    # request can't push the total wait past the advertised cap) - shrink
    # the interval instead of faking time.sleep away, or the deadline
    # check would busy-loop for real wall-clock seconds with no delay.
    monkeypatch.setattr("prime_cli.commands.rl.HOSTED_TRAINING_STOP_POLL_SECONDS", 0.01)

    result = runner.invoke(
        app,
        ["train", "stop", "run-1", "--force"],
        env={**TEST_ENV, "PRIME_API_KEY": "test-key"},
    )

    assert result.exit_code == 0, result.output
    assert "stopped successfully" in result.output
    assert calls == [
        ("PUT", "/rft/runs/run-1/stop"),
        ("GET", "/rft/runs/run-1"),
        ("GET", "/rft/runs/run-1"),
    ]


def test_train_stop_gives_up_after_max_polls(monkeypatch) -> None:
    def mock_request(self, method, endpoint, params=None, json=None, timeout=None):
        return {"run": _fake_run_payload("RUNNING")}

    monkeypatch.setattr("prime_cli.core.client.APIClient.request", mock_request)
    monkeypatch.setattr("prime_cli.commands.rl.HOSTED_TRAINING_STOP_POLL_SECONDS", 0.01)
    monkeypatch.setattr("prime_cli.commands.rl.HOSTED_TRAINING_STOP_MAX_POLLS", 2)

    result = runner.invoke(
        app,
        ["train", "stop", "run-1", "--force"],
        env={**TEST_ENV, "PRIME_API_KEY": "test-key"},
    )

    assert result.exit_code == 0, result.output
    assert "did not reach a terminal state" in result.output


def test_train_stop_reports_non_stopped_terminal_status_accurately(monkeypatch) -> None:
    """A run that transitions to FAILED/COMPLETED while stop is polling
    was not actively stopped - the CLI must not claim success."""
    statuses = iter(["RUNNING", "FAILED"])

    def mock_request(self, method, endpoint, params=None, json=None, timeout=None):
        return {"run": _fake_run_payload(next(statuses))}

    monkeypatch.setattr("prime_cli.core.client.APIClient.request", mock_request)
    monkeypatch.setattr("prime_cli.commands.rl.HOSTED_TRAINING_STOP_POLL_SECONDS", 0.01)

    result = runner.invoke(
        app,
        ["train", "stop", "run-1", "--force"],
        env={**TEST_ENV, "PRIME_API_KEY": "test-key"},
    )

    normalized_output = " ".join(result.output.split())

    assert result.exit_code == 0, result.output
    assert "stopped successfully" not in normalized_output
    assert "was not actively stopped" in normalized_output
    assert "FAILED" in normalized_output


def test_train_stop_survives_a_stalled_poll_request(monkeypatch) -> None:
    """A poll request that itself times out (deadline enforced at the
    transport level) must not be reported as a failed stop - stop_run
    already succeeded before polling started."""
    from prime_cli.core import APITimeoutError

    def mock_request(self, method, endpoint, params=None, json=None, timeout=None):
        if method == "PUT":
            return {"run": _fake_run_payload("RUNNING")}
        raise APITimeoutError("Request timed out")

    monkeypatch.setattr("prime_cli.core.client.APIClient.request", mock_request)
    monkeypatch.setattr("prime_cli.commands.rl.HOSTED_TRAINING_STOP_POLL_SECONDS", 0.01)

    result = runner.invoke(
        app,
        ["train", "stop", "run-1", "--force"],
        env={**TEST_ENV, "PRIME_API_KEY": "test-key"},
    )

    assert result.exit_code == 0, result.output
    assert "did not reach a terminal state" in result.output
    assert "Error:" not in result.output
