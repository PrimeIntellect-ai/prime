import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
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
    assert "logs" in result.output
    assert "request" in result.output
    # Retired with shared LoRA training: both only knew the LoRA schema.
    assert not re.search(r"^│\s+(init|configs)\s", result.output, re.M)
    # --fft is a deprecated no-op: still parsed, no longer advertised.
    assert "--full-finetune" not in result.output


def test_rl_alias_is_hidden_from_root_help() -> None:
    result = runner.invoke(app, ["--help"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Launch and manage Hosted Training runs." in result.output
    assert "Deprecated alias for `prime train`." not in result.output


def _mock_discovery(monkeypatch) -> None:
    from prime_cli.api.training import AvailableFFTModelsResponse

    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    monkeypatch.setattr(
        "prime_cli.api.training.HostedTrainingClient.get_available_fft",
        lambda self, team_id=None: AvailableFFTModelsResponse(models=[]),
    )


def test_rl_alias_still_works_with_deprecation_warning(monkeypatch) -> None:
    _mock_discovery(monkeypatch)

    result = runner.invoke(app, ["rl", "models"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert (
        "[DEPRECATED] The 'rl' command is deprecated. Use 'prime train' instead."
    ) in result.output
    assert "No models available for Hosted Training." in result.output


def test_rl_alias_warning_uses_stderr_for_json_output(monkeypatch) -> None:
    _mock_discovery(monkeypatch)

    result = runner.invoke(app, ["rl", "models", "--output", "json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert (
        "[DEPRECATED] The 'rl' command is deprecated. Use 'prime train' instead."
    ) in result.stderr
    assert "[DEPRECATED]" not in result.stdout
    assert json.loads(result.stdout) == {"available_fft_models": []}


_LORA_CONFIG = """\
model = "Qwen/Qwen3.5-0.8B"
max_steps = 100

[[env]]
id = "primeintellect/reverse-text"
"""


def test_train_rejects_lora_config_without_dispatch(monkeypatch, tmp_path: Path) -> None:
    """Shared LoRA no longer accepts runs, so a LoRA-shaped config fails
    locally with a pointer to prime-rl configs instead of reaching the
    dedicated endpoint and coming back as a validator 422."""
    captured = _capture_fft_dispatch(monkeypatch)
    cfg = tmp_path / "rl.toml"
    for body in (_LORA_CONFIG, 'model = "Qwen/Qwen3-0.6B"\n'):
        cfg.write_text(body)
        for args in ([], ["--fft"]):
            result = runner.invoke(app, ["train", str(cfg), *args, "-y"], env=TEST_ENV)
            assert result.exit_code == 1, (body, args)
            assert "shared LoRA training config" in result.output
            assert "prime-rl" in result.output
    assert captured == []


def test_train_rl_config_dispatches_without_deployment_block(monkeypatch, tmp_path: Path) -> None:
    """The same prime-rl config runs locally and hosted: an unset
    [deployment] falls through to prime-rl's own default sizing, with or
    without the now no-op --fft flag."""
    captured = _capture_fft_dispatch(monkeypatch)
    cfg = tmp_path / "rl.toml"
    cfg.write_text('max_steps = 20\n\n[model]\nname = "Qwen/Qwen3-0.6B"\n\n[orchestrator]\n')
    for args in ([], ["--fft"], ["--full-finetune"]):
        result = runner.invoke(app, ["train", str(cfg), *args, "-y", "-o", "json"], env=TEST_ENV)
        assert result.exit_code == 0, (args, result.output)
        assert json.loads(result.output)["run"]["runId"] == "r1"
    assert len(captured) == 3
    assert all("deployment" not in p["config"] for p in captured)
    assert all(p.get("mode") in (None, "rl") for p in captured)


def test_train_reports_missing_and_invalid_config(monkeypatch, tmp_path: Path) -> None:
    captured = _capture_fft_dispatch(monkeypatch)

    result = runner.invoke(app, ["train", str(tmp_path / "nope.toml"), "-y"], env=TEST_ENV)
    assert result.exit_code == 1
    assert "Config file not found" in result.output

    cfg = tmp_path / "rl.toml"
    cfg.write_text("[model\n")
    result = runner.invoke(app, ["train", str(cfg), "-y"], env=TEST_ENV)
    assert result.exit_code == 1
    assert "Invalid TOML" in result.output
    assert captured == []


def test_train_request_submits_model_request(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def mock_post(self: Any, endpoint: str, json: dict[str, Any] | None = None, **_: Any) -> dict:
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


def test_train_stop_survives_any_poll_api_error(monkeypatch) -> None:
    """Not just timeouts - any APIError on a follow-up status poll (a
    transient connection failure, a 5xx, ...) is inconclusive about the
    stop, not a failed stop, since stop_run already succeeded."""
    from prime_cli.core import APIError

    def mock_request(self, method, endpoint, params=None, json=None, timeout=None):
        if method == "PUT":
            return {"run": _fake_run_payload("RUNNING")}
        raise APIError("HTTP 502: Bad Gateway")

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


_FFT_BODY = (
    '[model]\nname = "Qwen/Qwen3-0.6B"\n\n[deployment]\nnum_train_gpus = 1\nnum_infer_gpus = 1\n'
)


def _capture_fft_dispatch(monkeypatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def fake_create_run(self, payload):
        captured.append(payload)
        from prime_cli.api.training import HostedTrainingRunResponse

        return HostedTrainingRunResponse(run_id="r1", token_value="t")

    monkeypatch.setattr("prime_cli.api.training.HostedTrainingClient.create_run", fake_create_run)
    return captured


def test_train_volume_flag_and_toml_key_reach_the_fft_payload(monkeypatch, tmp_path: Path) -> None:
    _mock_volumes(monkeypatch, [_vol("my-ckpts"), _vol("from-toml")])
    captured = _capture_fft_dispatch(monkeypatch)  # after _mock_volumes: last create_run patch wins
    cfg = tmp_path / "rl.toml"
    cfg.write_text(_FFT_BODY)
    result = runner.invoke(
        app, ["train", str(cfg), "--volume", "my-ckpts", "-y", "-o", "json"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    cfg.write_text('volume = "from-toml"\n' + _FFT_BODY)
    result = runner.invoke(app, ["train", str(cfg), "-y", "-o", "json"], env=TEST_ENV)
    assert result.exit_code == 0, result.output

    assert [p.get("volume") for p in captured] == ["my-ckpts", "from-toml"]
    assert "volume" not in captured[1]["config"]


def _vol(name: str, status: str = "RUNNING"):
    from prime_cli.api.training import Volume

    return Volume(name=name, size="1Ti", status=status, clusterId="c", namespace="n", pvcName="p")


def _mock_volumes(monkeypatch, existing, created_status="RUNNING", create_error=None):
    """Patch volume + dispatch calls. Returns (creates, dispatched)."""
    from prime_cli.api.training import HostedTrainingRunResponse
    from prime_cli.client import APIError

    state = {"vols": list(existing)}
    creates: list[tuple[str, str]] = []
    dispatched: list[dict[str, Any]] = []

    def create_volume(self, name, size, team_id=None):
        creates.append((name, size))
        if create_error:
            raise APIError(create_error)
        v = _vol(name, "PENDING")
        state["vols"].append(_vol(name, created_status))
        return v

    def create_run(self, payload):
        dispatched.append(payload)
        return HostedTrainingRunResponse(run_id="r1", token_value="t")

    p = "prime_cli.api.training.HostedTrainingClient."
    monkeypatch.setattr(p + "list_volumes", lambda self, team_id=None: list(state["vols"]))
    monkeypatch.setattr(p + "create_volume", create_volume)
    monkeypatch.setattr(p + "create_run", create_run)
    monkeypatch.setattr("prime_cli.commands.rl.time.sleep", lambda s: None)
    return creates, dispatched


def _run_volume(tmp_path: Path, *extra: str, toml_extra: str = ""):
    cfg = tmp_path / "rl.toml"
    cfg.write_text(
        toml_extra + '[model]\nname = "Qwen/Qwen3-0.6B"\n\n'
        "[deployment]\nnum_train_gpus = 1\nnum_infer_gpus = 1\n"
    )
    return runner.invoke(app, ["train", str(cfg), "--volume", "ckpts", "-y", *extra], env=TEST_ENV)


def test_train_volume_exists_does_not_create(monkeypatch, tmp_path: Path) -> None:
    creates, dispatched = _mock_volumes(monkeypatch, [_vol("ckpts")])
    assert _run_volume(tmp_path).exit_code == 0
    assert creates == [] and len(dispatched) == 1


def test_train_missing_volume_is_created_then_dispatched(monkeypatch, tmp_path: Path) -> None:
    creates, dispatched = _mock_volumes(monkeypatch, [])
    result = _run_volume(tmp_path)
    assert result.exit_code == 0, result.output
    assert "Volume 'ckpts' doesn't exist, creating it (5Ti)" in result.output
    assert creates == [("ckpts", "5Ti")] and len(dispatched) == 1


def test_train_volume_create_failure_exits_without_dispatch(monkeypatch, tmp_path: Path) -> None:
    creates, dispatched = _mock_volumes(monkeypatch, [], create_error="no cluster assigned")
    result = _run_volume(tmp_path)
    assert result.exit_code == 1 and "no cluster assigned" in result.output
    assert len(creates) == 1 and dispatched == []


def test_train_volume_failed_status_exits_without_dispatch(monkeypatch, tmp_path: Path) -> None:
    creates, dispatched = _mock_volumes(monkeypatch, [], created_status="FAILED")
    result = _run_volume(tmp_path)
    assert result.exit_code == 1 and "FAILED" in result.output
    assert dispatched == []


def test_train_existing_deploying_volume_is_not_recreated(monkeypatch, tmp_path: Path) -> None:
    creates, dispatched = _mock_volumes(monkeypatch, [_vol("ckpts", "DEPLOYING")])
    assert _run_volume(tmp_path).exit_code == 0
    assert creates == [] and len(dispatched) == 1


def test_train_volume_size_flag_and_toml_key_set_the_created_size(
    monkeypatch, tmp_path: Path
) -> None:
    creates, dispatched = _mock_volumes(monkeypatch, [])
    result = _run_volume(tmp_path, "--volume-size", "500Gi")
    assert result.exit_code == 0, result.output
    assert "creating it (500Gi)" in result.output
    assert creates == [("ckpts", "500Gi")]
    # The TOML key works too, and never reaches the prime-rl config.
    creates, dispatched = _mock_volumes(monkeypatch, [])
    result = _run_volume(tmp_path, toml_extra='volume_size = "2Ti"\n')
    assert result.exit_code == 0, result.output
    assert creates == [("ckpts", "2Ti")]
    assert "volume_size" not in str(dispatched[0])


def test_train_volume_size_is_ignored_for_an_existing_volume(monkeypatch, tmp_path: Path) -> None:
    creates, dispatched = _mock_volumes(monkeypatch, [_vol("ckpts")])
    result = _run_volume(tmp_path, "--volume-size", "5Ti")
    assert result.exit_code == 0, result.output
    assert creates == [] and len(dispatched) == 1
    assert "already exists (1Ti); --volume-size 5Ti is ignored" in result.output


def test_train_volume_size_needs_volume(tmp_path: Path) -> None:
    cfg = tmp_path / "rl.toml"
    cfg.write_text(
        '[model]\nname = "Qwen/Qwen3-0.6B"\n\n'
        "[deployment]\nnum_train_gpus = 1\nnum_infer_gpus = 1\n"
    )
    result = runner.invoke(app, ["train", str(cfg), "--volume-size", "1Ti", "-y"], env=TEST_ENV)
    assert result.exit_code == 1
    assert "--volume-size" in result.output and "needs" in result.output


def test_train_ref_flag_and_toml_key_reach_the_fft_payload(monkeypatch, tmp_path: Path) -> None:
    captured = _capture_fft_dispatch(monkeypatch)
    cfg = tmp_path / "rl.toml"
    cfg.write_text(_FFT_BODY)
    result = runner.invoke(
        app, ["train", str(cfg), "--ref", "feat/my-branch", "-y", "-o", "json"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["run"]["sourceRef"] == "feat/my-branch"

    cfg.write_text('source_ref = "from-toml"\n' + _FFT_BODY)
    result = runner.invoke(app, ["train", str(cfg), "-y", "-o", "json"], env=TEST_ENV)
    assert result.exit_code == 0, result.output

    # CLI flag wins over the TOML key.
    result = runner.invoke(
        app, ["train", str(cfg), "--ref", "abc123", "-y", "-o", "json"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output

    assert [p.get("sourceRef") for p in captured] == ["feat/my-branch", "from-toml", "abc123"]
    assert all("source_ref" not in p["config"] for p in captured)


def test_train_pr_flag_resolves_head_sha_and_conflicts_with_ref(
    monkeypatch, tmp_path: Path
) -> None:
    captured = _capture_fft_dispatch(monkeypatch)
    sha = "0123456789abcdef0123456789abcdef01234567"
    seen: list[int] = []

    def fake_resolve(pr_number: int) -> str:
        seen.append(pr_number)
        return sha

    monkeypatch.setattr("prime_cli.api.training.resolve_pull_request_head", fake_resolve)
    cfg = tmp_path / "rl.toml"
    cfg.write_text(_FFT_BODY)
    result = runner.invoke(app, ["train", str(cfg), "--pr", "42", "-y", "-o", "json"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert seen == [42]
    assert captured[0]["sourceRef"] == sha
    # --output json must stay pure JSON: no "Resolved PR" line on stdout.
    assert json.loads(result.output)["run"]["sourceRef"] == sha

    # Table mode shows the resolved sha once, in the build banner, with the
    # PR it came from.
    result = runner.invoke(app, ["train", str(cfg), "--pr", "42", "-y"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert f"Source ref: {sha} (PR #42)" in result.output
    assert result.output.count(sha) == 1

    # A run that fails a local check never reaches GitHub.
    def never(pr_number: int) -> str:
        raise AssertionError("resolved a PR for a dispatch that fails locally")

    monkeypatch.setattr("prime_cli.api.training.resolve_pull_request_head", never)
    result = runner.invoke(
        app, ["train", str(cfg), "--pr", "42", "-e", "FOO=bar", "-y"], env=TEST_ENV
    )
    assert result.exit_code == 1
    assert "secret(s): FOO" in result.output
    monkeypatch.setattr("prime_cli.api.training.resolve_pull_request_head", fake_resolve)

    result = runner.invoke(
        app, ["train", str(cfg), "--pr", "42", "--ref", "main", "-y"], env=TEST_ENV
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output
    assert len(captured) == 2  # the json + table dispatches above, nothing since


def test_train_help_hides_source_overlay_flags() -> None:
    # --ref / --pr are restricted (granted per team on the platform): they
    # still parse (see the payload tests above) but must not appear in the
    # public help panel.
    result = runner.invoke(app, ["train", "--help"], env=TEST_ENV)
    assert result.exit_code == 0
    text = " ".join(result.output.split())
    assert not re.search(r"--(ref|pr)(?![\w-])", text), text
    assert "sourceRef access" not in text


def test_train_ref_shape_is_checked_before_dispatch(monkeypatch, tmp_path: Path) -> None:
    captured = _capture_fft_dispatch(monkeypatch)
    cfg = tmp_path / "rl.toml"
    cfg.write_text(_FFT_BODY)
    for bad in (
        "pull/42/head",
        "refs/heads/x",
        "",
        "a b",
        "feat/../main",
        "feat//x",
        "feat/.hidden",
        "-x",
        "x.lock",
        "feat/x.lock/y",  # any component ending in .lock, not just the last
        "feat/x.",  # trailing dot
        "a" * 201,
    ):
        result = runner.invoke(app, ["train", str(cfg), "--ref", bad, "-y"], env=TEST_ENV)
        assert result.exit_code == 1, bad
        assert "--ref" in result.output, bad
    # The TOML key gets the same check, named by its origin.
    cfg.write_text('source_ref = "pull/42/head"\n' + _FFT_BODY)
    result = runner.invoke(app, ["train", str(cfg), "-y"], env=TEST_ENV)
    assert result.exit_code == 1
    assert "source_ref in" in result.output and "--pr" in result.output
    assert captured == []

    # An empty --ref no longer slips past the --pr exclusivity check.
    result = runner.invoke(app, ["train", str(cfg), "--ref", "", "--pr", "42", "-y"], env=TEST_ENV)
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_source_ref_check_keeps_valid_git_refs() -> None:
    # Dots and `.lock`-like text are fine wherever git allows them; the
    # check must only reject what git-check-ref-format rejects.
    from prime_cli.commands.rl import _source_ref_error

    for ok in (
        "main",
        "v1.2.3",
        "feat/my.branch",
        "release/1.0.lockfile",
        "lock/x",
        "0123456789abcdef0123456789abcdef01234567",
        "a" * 200,
    ):
        assert _source_ref_error(ok) is None, ok


def test_train_source_overlay_reaches_the_sft_payload(monkeypatch, tmp_path: Path) -> None:
    # SFT shares the dedicated-run dispatch with full-FT, so --ref / --pr
    # are forwarded there too (never silently dropped).
    captured = _capture_fft_dispatch(monkeypatch)
    sha = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr("prime_cli.api.training.resolve_pull_request_head", lambda pr_number: sha)
    cfg = tmp_path / "sft.toml"
    sft = '[model]\nname = "Qwen/Qwen3-0.6B"\n\n[data]\nname = "/volume/datasets/x"\n'
    cfg.write_text(sft)
    result = runner.invoke(
        app, ["train", str(cfg), "--sft", "--ref", "feat/x", "-y", "-o", "json"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app, ["train", str(cfg), "--sft", "--pr", "42", "-y", "-o", "json"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert [(p["mode"], p["sourceRef"]) for p in captured] == [("sft", "feat/x"), ("sft", sha)]


def _gh_pull(monkeypatch, status: int = 200, body: Any = None, headers: dict | None = None):
    """Fake the one GitHub call resolve_pull_request_head makes; returns the
    request headers it was sent so tests can assert on auth."""
    import httpx

    sent: dict[str, Any] = {}

    def fake_get(url, headers=None, timeout=None):
        sent.update(headers or {})
        return httpx.Response(
            status, json=body, headers=headers_ or {}, request=httpx.Request("GET", url)
        )

    headers_ = headers
    monkeypatch.setattr("prime_cli.api.training.httpx.get", fake_get)
    return sent


def _pull_body(sha: str = "a" * 40, repo: str = "PrimeIntellect-ai/prime-rl", **extra) -> dict:
    return {"head": {"sha": sha, "repo": {"full_name": repo}}, **extra}


def test_resolve_pull_request_head_refuses_a_merged_pr(monkeypatch) -> None:
    import pytest
    from prime_cli.api.training import resolve_pull_request_head
    from prime_cli.core import APIError

    _gh_pull(monkeypatch, body=_pull_body(merged=True, merge_commit_sha="b" * 40))
    with pytest.raises(APIError, match=r"already merged.*--ref main.*" + "b" * 40):
        resolve_pull_request_head(7)

    # `merged: false` on an open PR resolves as before.
    _gh_pull(monkeypatch, body=_pull_body(merged=False))
    assert resolve_pull_request_head(7) == "a" * 40


def test_resolve_pull_request_head_sends_a_github_token_when_set(monkeypatch) -> None:
    from prime_cli.api.training import resolve_pull_request_head

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    sent = _gh_pull(monkeypatch, body=_pull_body())
    resolve_pull_request_head(7)
    assert "Authorization" not in sent

    monkeypatch.setenv("GH_TOKEN", "ghp_x")
    sent = _gh_pull(monkeypatch, body=_pull_body())
    resolve_pull_request_head(7)
    assert sent["Authorization"] == "Bearer ghp_x"

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_y")  # GITHUB_TOKEN wins over GH_TOKEN
    sent = _gh_pull(monkeypatch, body=_pull_body())
    resolve_pull_request_head(7)
    assert sent["Authorization"] == "Bearer ghp_y"


def test_resolve_pull_request_head_error_mapping(monkeypatch) -> None:
    import pytest
    from prime_cli.api.training import resolve_pull_request_head
    from prime_cli.core import APIError

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    cases = [
        (404, {}, {}, r"not found"),
        (429, {}, {}, r"rate limit.*GITHUB_TOKEN"),
        (403, {"x-ratelimit-remaining": "0"}, {}, r"rate limit"),
        (403, {"retry-after": "60"}, {}, r"rate limit"),
        # A secondary limit may carry neither header (GitHub documents both
        # as optional); the error body is then the only signal.
        (
            403,
            {"x-ratelimit-remaining": "57"},
            {"message": "You have exceeded a secondary rate limit. Please wait a few minutes."},
            r"rate limit",
        ),
        (
            403,
            {},
            {"message": "You have triggered an abuse detection mechanism. Please wait."},
            r"rate limit",
        ),
        (
            403,
            {},
            {
                "message": "Forbidden",
                "documentation_url": "https://docs.github.com/rest/overview/"
                "rate-limits-for-the-rest-api#about-secondary-rate-limits",
            },
            r"rate limit",
        ),
        # A 403 that isn't a quota is a refusal, reported with GitHub's reason.
        (403, {}, {"message": "Resource protected by organization SAML"}, r"refused.*SAML"),
        (403, {}, "not an object", r"refused"),
        (500, {}, {}, r"HTTP 500"),
        (200, {}, {"nope": True}, r"Unexpected GitHub response"),
        (200, {}, _pull_body(sha=""), r"no head commit sha"),
    ]
    for status, resp_headers, body, pattern in cases:
        _gh_pull(monkeypatch, status=status, body=body, headers=resp_headers)
        with pytest.raises(APIError, match=pattern):
            resolve_pull_request_head(7)

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_y")
    _gh_pull(monkeypatch, status=403, body={}, headers={"x-ratelimit-remaining": "0"})
    with pytest.raises(APIError, match=r"rate limit") as err:
        resolve_pull_request_head(7)
    assert "GITHUB_TOKEN" not in str(err.value)  # hint only when no token is set

    def boom(url, headers=None, timeout=None):
        import httpx

        raise httpx.ConnectError("refused")

    monkeypatch.setattr("prime_cli.api.training.httpx.get", boom)
    with pytest.raises(APIError, match=r"Could not reach GitHub"):
        resolve_pull_request_head(7)


def test_resolve_pull_request_head_rejects_forks(monkeypatch) -> None:
    import pytest
    from prime_cli.api.training import resolve_pull_request_head
    from prime_cli.core import APIError

    _gh_pull(monkeypatch, body=_pull_body(sha="f" * 40, repo="someone/prime-rl"))
    with pytest.raises(APIError, match=r"fork"):
        resolve_pull_request_head(7)

    # A deleted fork leaves `head.repo` null: still a fork, never a crash.
    _gh_pull(monkeypatch, body={"head": {"sha": "f" * 40, "repo": None}})
    with pytest.raises(APIError, match=r"fork \(unknown\)"):
        resolve_pull_request_head(7)

    _gh_pull(monkeypatch, body=_pull_body())
    assert resolve_pull_request_head(7) == "a" * 40


def test_train_get_prints_the_dedicated_runs_cluster(monkeypatch) -> None:
    """Dedicated runs carry their cluster as primeClusterId (rftClusterId
    is NULL on those rows); the get table must show it, not stay silent."""
    from prime_cli.api.rl import RLRun

    run = RLRun.model_validate(
        {
            "id": "kuvz6ncuc5qzezo3uk3afrur",
            "userId": "user-1",
            "status": "RUNNING",
            "kind": "DEDICATED_FULL_FT",
            "primeClusterId": "ccl9wyl7n0002f2dn0g0x3p0q",
            "rftClusterId": "",
            "createdAt": "2026-05-17T00:00:00Z",
            "updatedAt": "2026-05-17T00:00:00Z",
        }
    )
    monkeypatch.setattr(
        "prime_cli.commands.rl.RLClient.get_run", lambda self, run_id, timeout=None: run
    )

    result = runner.invoke(app, ["train", "get", "kuvz6ncuc5qzezo3uk3afrur"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Cluster: ccl9wyl7n0002f2dn0g0x3p0q" in result.output


def test_train_get_prints_a_shared_runs_cluster(monkeypatch) -> None:
    from prime_cli.api.rl import RLRun

    run = RLRun.model_validate(
        {
            "id": "shared-run",
            "userId": "user-1",
            "status": "RUNNING",
            "kind": "SHARED_RFT_HOSTED",
            "rftClusterId": "oes1yd0lubfsku6iekofnoau",
            "createdAt": "2026-05-17T00:00:00Z",
            "updatedAt": "2026-05-17T00:00:00Z",
        }
    )
    monkeypatch.setattr(
        "prime_cli.commands.rl.RLClient.get_run", lambda self, run_id, timeout=None: run
    )

    result = runner.invoke(app, ["train", "get", "shared-run"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Cluster: oes1yd0lubfsku6iekofnoau" in result.output


# --- on-demand billed launch output -----------------------------------------
#
# The backend derives placement server-side, so the CLI echoes what the
# run costs from the create-response wire fields. These tests mock
# APIClient.post for /training/runs (the wire layer) rather than
# HostedTrainingClient.create_run, so the camelCase aliases are parsed by
# the real response model.

_BILLED_RESPONSE = {
    "runId": "run-priced",
    "tokenValue": "SECRET",
    "onDemand": True,
    "gpuType": "H100_80GB",
    "pricePerGpuHour": 1.5,
    "estimatedCostPerHour": 12.0,
    "isBeta": True,
}


def _mock_run_post(monkeypatch, response: dict[str, Any]) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    # `timeout` is part of the dispatch POST's signature now: the backend's
    # validator hop needs more than the client's 30s default, and a mock
    # that swallowed the kwarg would hide a regression in that budget.
    def mock_post(
        self: Any,
        endpoint: str,
        json: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> dict:
        captured["endpoint"] = endpoint
        captured["json"] = json
        captured["timeout"] = timeout
        return response

    monkeypatch.setattr("prime_cli.core.APIClient.post", mock_post)
    return captured


@pytest.mark.parametrize("is_beta", [True, False], ids=["beta", "stable"])
def test_train_fft_billed_dispatch_prints_price_and_beta_label(
    monkeypatch, tmp_path: Path, is_beta: bool
) -> None:
    _mock_run_post(monkeypatch, dict(_BILLED_RESPONSE, isBeta=is_beta))
    cfg = tmp_path / "fft.toml"
    cfg.write_text(_FFT_BODY)

    result = runner.invoke(
        app, ["train", str(cfg), "--yes"], env={**TEST_ENV, "PRIME_API_KEY": "test-key"}
    )

    assert result.exit_code == 0, result.output
    assert "Dispatched" in result.output
    assert "run-priced" in result.output
    assert "H100_80GB" in result.output
    assert "$1.50/GPU-hr" in result.output
    assert "~$12.00/hr" in result.output
    # Readiness wording: billable time starts at pod readiness.
    assert "Billing starts when the pods are ready" in result.output
    # The (beta) badge is server-driven and retires without a CLI release.
    if is_beta:
        assert "On-demand (beta)" in result.output
    else:
        assert "On-demand" in result.output
        assert "(beta)" not in result.output
    # The per-run token is bound into a k8s Secret server-side and must
    # never reach stdout.
    assert "SECRET" not in result.output


def test_train_fft_billed_json_keeps_cost_facts_and_hides_token(
    monkeypatch, tmp_path: Path
) -> None:
    _mock_run_post(monkeypatch, _BILLED_RESPONSE)
    cfg = tmp_path / "fft.toml"
    cfg.write_text(_FFT_BODY)

    result = runner.invoke(
        app,
        ["train", str(cfg), "--yes", "--output", "json"],
        env={**TEST_ENV, "PRIME_API_KEY": "test-key"},
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["run"] == {
        "runId": "run-priced",
        "onDemand": True,
        "gpuType": "H100_80GB",
        "pricePerGpuHour": 1.5,
        "estimatedCostPerHour": 12.0,
        "isBeta": True,
    }
    assert "tokenValue" not in data["run"]
    assert "SECRET" not in result.stdout


def test_train_fft_reserved_response_prints_no_billing_line(monkeypatch, tmp_path: Path) -> None:
    """A reserved run has no hourly rate — no billing line may be
    fabricated for it."""
    captured = _mock_run_post(monkeypatch, {"runId": "r1", "tokenValue": "t"})
    cfg = tmp_path / "fft.toml"
    cfg.write_text(_FFT_BODY)

    result = runner.invoke(
        app, ["train", str(cfg), "--yes"], env={**TEST_ENV, "PRIME_API_KEY": "test-key"}
    )

    assert result.exit_code == 0, result.output
    assert captured["endpoint"] == "/training/runs"
    assert "On-demand" not in result.output
    assert "GPU-hr" not in result.output


def test_train_fft_on_demand_false_with_null_costs_prints_no_billing_line(
    monkeypatch, tmp_path: Path
) -> None:
    """`onDemand: false` may carry null cost fields (older backends omit
    them entirely): must parse, exit 0, and print no billing line."""
    response = {
        "runId": "r1",
        "tokenValue": "t",
        "onDemand": False,
        "gpuType": None,
        "pricePerGpuHour": None,
        "estimatedCostPerHour": None,
        "isBeta": False,
    }
    _mock_run_post(monkeypatch, response)
    cfg = tmp_path / "fft.toml"
    cfg.write_text(_FFT_BODY)

    result = runner.invoke(
        app, ["train", str(cfg), "--yes"], env={**TEST_ENV, "PRIME_API_KEY": "test-key"}
    )

    assert result.exit_code == 0, result.output
    assert "On-demand" not in result.output


def test_train_fft_billed_zero_rate_is_visible(monkeypatch, tmp_path: Path) -> None:
    """A billed run at $0 is a real fact, not an unknown — it must render
    as $0.00 rather than being swallowed."""
    _mock_run_post(
        monkeypatch, dict(_BILLED_RESPONSE, pricePerGpuHour=0.0, estimatedCostPerHour=0.0)
    )
    cfg = tmp_path / "fft.toml"
    cfg.write_text(_FFT_BODY)

    result = runner.invoke(
        app, ["train", str(cfg), "--yes"], env={**TEST_ENV, "PRIME_API_KEY": "test-key"}
    )

    assert result.exit_code == 0, result.output
    assert "$0.00/GPU-hr" in result.output
    assert "~$0.00/hr" in result.output


# --- ambiguous dispatch: client timeout on a POST that may have committed ----
#
# `prime train <config>` returned `Error: Request timed out: The read
# operation timed out` while the backend had already created
# cyac46dphnrcw0pcbqcdg6od (observed in PENDING right after). The CLI
# called that a failure, so the natural reaction - re-run the command -
# dispatched a SECOND paid run. The POST's own budget has to clear the
# backend's 60s validator hop, and whatever timeout is left has to be
# reported as "outcome unknown", never as a plain error.


def test_dispatch_post_outlasts_the_backend_validator_budget(monkeypatch, tmp_path: Path) -> None:
    """The shared 30s client default expires before the backend's 60s
    validator budget, which is what made the timeout routine."""
    from prime_cli.api.training import CREATE_RUN_TIMEOUT
    from prime_cli.core.client import DEFAULT_CONNECT_TIMEOUT_S

    captured = _mock_run_post(monkeypatch, dict(_BILLED_RESPONSE))
    cfg = tmp_path / "fft.toml"
    cfg.write_text(_FFT_BODY)

    result = runner.invoke(
        app, ["train", str(cfg), "--yes"], env={**TEST_ENV, "PRIME_API_KEY": "test-key"}
    )

    assert result.exit_code == 0, result.output
    assert captured["endpoint"] == "/training/runs"
    # The READ budget must outlast the server's own validator budget,
    # while the connect budget stays short — a host that never answers
    # is a plain failure, not an ambiguous outcome, and must not hang
    # for the full dispatch window.
    assert captured["timeout"] is CREATE_RUN_TIMEOUT
    assert captured["timeout"].read == CREATE_RUN_TIMEOUT.read
    assert captured["timeout"].connect == DEFAULT_CONNECT_TIMEOUT_S
    assert captured["timeout"].read > 60, (
        "the dispatch POST must outlast the backend's _VALIDATOR_HTTP_TIMEOUT_S "
        "(60s) or a cold validator times out the client while the run is created"
    )


def _timeout_dispatch(
    monkeypatch, tmp_path: Path, *, runs, name=None, extra=(), legacy_total_none=False
):
    """Dispatch an FFT config whose POST times out, with `runs` as the
    recovery lookup's answer (an exception class re-raises instead)."""
    from prime_cli.api.rl import RLRun, RLRunPage
    from prime_cli.client import APITimeoutError

    def mock_post(
        self: Any,
        endpoint: str,
        json: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> dict:
        raise APITimeoutError("Request timed out: The read operation timed out")

    monkeypatch.setattr("prime_cli.core.APIClient.post", mock_post)

    def list_runs(self, team_id=None, *, mine=False, page=1, limit=20):
        if isinstance(runs, type) and issubclass(runs, Exception):
            raise runs("boom")
        # `total` set: a modern backend that honoured `mine`, so recovery
        # trusts the server-side filter. `legacy_total_none=True` models
        # the older backend that ignored page/limit/mine and returned the
        # whole team history, which recovery has to re-filter itself.
        return RLRunPage(
            runs=[RLRun.model_validate(r) for r in runs],
            total=None if legacy_total_none else len(runs),
        )

    monkeypatch.setattr("prime_cli.api.rl.RLClient.list_runs", list_runs)

    cfg = tmp_path / "fft.toml"
    cfg.write_text((f'name = "{name}"\n' if name else "") + _FFT_BODY)
    return runner.invoke(
        app,
        ["train", str(cfg), "--yes", *extra],
        env={**TEST_ENV, "PRIME_API_KEY": "test-key"},
    )


def _candidate_run(**overrides) -> dict[str, Any]:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    row = {
        "id": "cyac46dphnrcw0pcbqcdg6od",
        "userId": "u1",
        "status": "PENDING",
        "kind": "DEDICATED_FULL_FT",
        "createdAt": now,
        "updatedAt": now,
    }
    row.update(overrides)
    return row


def test_timed_out_dispatch_never_reads_as_a_plain_error(monkeypatch, tmp_path: Path) -> None:
    """`Error: Request timed out` is the exact output that caused the
    double dispatch. The outcome is unknown, and the user must be told
    not to re-run."""
    result = _timeout_dispatch(monkeypatch, tmp_path, runs=[])

    assert result.exit_code == 1, result.output
    assert "Error: Request timed out" not in result.output
    assert "Dispatch outcome unknown" in result.output
    assert "Do not re-run this command yet" in result.output
    assert "prime train list" in result.output


def test_timed_out_dispatch_names_the_run_it_created(monkeypatch, tmp_path: Path) -> None:
    result = _timeout_dispatch(monkeypatch, tmp_path, runs=[_candidate_run()])

    assert result.exit_code == 1, result.output
    # "most likely", never "did": this is a time-window heuristic, not a
    # correlation id, so claiming certainty could send the user to stop
    # an unrelated run.
    assert "most likely" in result.output
    assert "did create a run" not in result.output
    assert "cyac46dphnrcw0pcbqcdg6od" in result.output
    assert "before" in result.output and "retrying" in result.output


def test_timed_out_dispatch_matches_on_the_config_name(monkeypatch, tmp_path: Path) -> None:
    """With a named config the match is near-certain, so a run carrying a
    different name must not be claimed as this dispatch's."""
    result = _timeout_dispatch(
        monkeypatch,
        tmp_path,
        name="my-run",
        runs=[_candidate_run(name="someone-elses-run")],
    )

    assert "most likely" not in result.output
    assert "No matching run found yet" in result.output

    result = _timeout_dispatch(
        monkeypatch, tmp_path, name="my-run", runs=[_candidate_run(name="my-run")]
    )
    assert "most likely" in result.output


def test_timed_out_dispatch_ignores_terminal_and_stale_runs(monkeypatch, tmp_path: Path) -> None:
    """A run that already finished, or one from an hour ago, belongs to an
    older dispatch - offering it would send the user chasing the wrong id."""
    from datetime import datetime, timedelta, timezone

    # 30 min back: comfortably outside the clock-skew allowance, yet
    # recent enough that a loose scan window would have matched it.
    stale = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    result = _timeout_dispatch(
        monkeypatch,
        tmp_path,
        runs=[
            _candidate_run(id="old-finished", status="COMPLETED"),
            _candidate_run(id="old-pending", createdAt=stale, updatedAt=stale),
            _candidate_run(id="lora-run", kind="SHARED_RFT_HOSTED"),
        ],
    )

    assert "No matching run found yet" in result.output
    for rejected in ("old-finished", "old-pending", "lora-run"):
        assert rejected not in result.output


def test_timed_out_dispatch_lists_every_candidate_when_ambiguous(
    monkeypatch, tmp_path: Path
) -> None:
    result = _timeout_dispatch(
        monkeypatch,
        tmp_path,
        runs=[_candidate_run(id="run-a"), _candidate_run(id="run-b")],
    )

    assert "could be yours" in result.output
    assert "run-a" in result.output
    assert "run-b" in result.output
    assert "most likely" not in result.output


def test_timed_out_dispatch_says_so_when_the_lookup_itself_fails(
    monkeypatch, tmp_path: Path
) -> None:
    """ "We could not check" must never render as "nothing was created" -
    that is the sentence that licenses the duplicate dispatch."""
    from prime_cli.client import APIError

    result = _timeout_dispatch(monkeypatch, tmp_path, runs=APIError)

    assert result.exit_code == 1, result.output
    assert "Could not reach the API to check" in result.output
    assert "No matching run found yet" not in result.output
    assert "Do not re-run this command yet" in result.output


def test_timed_out_dispatch_keeps_json_stdout_clean(monkeypatch, tmp_path: Path) -> None:
    """Automation parses stdout as the run payload. An ambiguous dispatch
    emits no payload, so the whole report goes to stderr and the
    non-zero exit is the only signal on stdout."""
    result = _timeout_dispatch(monkeypatch, tmp_path, runs=[_candidate_run()], extra=("-o", "json"))

    assert result.exit_code == 1
    assert "Dispatch outcome unknown" in result.stderr
    assert "cyac46dphnrcw0pcbqcdg6od" in result.stderr
    assert "Dispatch outcome unknown" not in result.stdout


def test_timed_out_dispatch_report_is_plain_mode_clean(monkeypatch, tmp_path: Path) -> None:
    """--plain is the AI/non-interactive surface: the guidance has to
    survive markup stripping, with no rich tags left behind."""
    result = _timeout_dispatch(monkeypatch, tmp_path, runs=[_candidate_run()], extra=("--plain",))

    assert result.exit_code == 1, result.output
    assert "Do not re-run this command yet" in result.output
    assert "cyac46dphnrcw0pcbqcdg6od" in result.output
    assert "[bold red]" not in result.output
    assert "[cyan]" not in result.output


def test_dispatch_timeout_keeps_the_connect_budget_short() -> None:
    """Only the READ budget is extended for dispatch.

    Handing httpx a bare number sets connect/read/write/pool alike,
    which would stretch the 10s connect limit to the full dispatch
    budget — so a host that blackholes connections would hang for the
    whole window before reporting an ambiguous outcome, when it never
    reached the backend and the outcome is simply "failed".
    """
    from prime_cli.api.training import CREATE_RUN_TIMEOUT
    from prime_cli.core.client import DEFAULT_CONNECT_TIMEOUT_S

    assert CREATE_RUN_TIMEOUT.connect == DEFAULT_CONNECT_TIMEOUT_S
    assert CREATE_RUN_TIMEOUT.connect < CREATE_RUN_TIMEOUT.read
    assert CREATE_RUN_TIMEOUT.read > 60, (
        "the dispatch read budget must outlast the backend's 60s validator hop"
    )


def test_recovery_ignores_a_teammates_run_on_a_legacy_backend(monkeypatch, tmp_path: Path) -> None:
    """A backend that ignored `mine` must not get a teammate's run named.

    `total is None` means the backend predates pagination and returned
    the full team history, so `mine=True` bought nothing. Without a
    client-side owner filter the only candidate left standing could
    belong to someone else — and recovery would offer to stop it.
    """
    result = _timeout_dispatch(
        monkeypatch,
        tmp_path,
        runs=[_candidate_run(userId="someone-else")],
        legacy_total_none=True,
    )

    assert result.exit_code == 1, result.output
    out = result.output
    assert "cyac46dphnrcw0pcbqcdg6od" not in out, (
        f"a teammate's run must never be named as this dispatch:\n{out}"
    )


def test_create_run_waits_out_a_cold_validator():
    """POST /training/runs validates through the rl-validator, which can take
    its full 60s when cold (ENG-6696); the 30s client default gave up first."""
    from unittest.mock import MagicMock

    from prime_cli.api.training import CREATE_RUN_TIMEOUT, HostedTrainingClient

    api = MagicMock()
    api.post.return_value = {"id": "r1"}
    try:
        HostedTrainingClient(api).create_run({"x": 1})
    except Exception:
        pass  # response model shape is not what's under test
    assert api.post.call_args.args == ("/training/runs",)
    timeout = api.post.call_args.kwargs["timeout"]
    assert timeout is CREATE_RUN_TIMEOUT
    # The read wait outlasts the validator; connecting still fails fast.
    assert timeout.read > 60 and timeout.connect == 10.0


def _paging_timeout_dispatch(monkeypatch, tmp_path: Path, *, pages, total=None):
    """Like `_timeout_dispatch`, but the lookup honours `page` so the
    recovery walk can be observed. `pages` is a list of per-page run
    lists; `requested` records every page number asked for."""
    from prime_cli.api.rl import RLRun, RLRunPage
    from prime_cli.client import APITimeoutError

    requested: list[int] = []

    def mock_post(
        self: Any,
        endpoint: str,
        json: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> dict:
        raise APITimeoutError("Request timed out: The read operation timed out")

    monkeypatch.setattr("prime_cli.core.APIClient.post", mock_post)

    flat = [r for page in pages for r in page]

    def list_runs(self, team_id=None, *, mine=False, page=1, limit=20):
        requested.append(page)
        rows = pages[page - 1] if page - 1 < len(pages) else []
        return RLRunPage(
            runs=[RLRun.model_validate(r) for r in rows],
            total=len(flat) if total is None else total,
        )

    monkeypatch.setattr("prime_cli.api.rl.RLClient.list_runs", list_runs)

    cfg = tmp_path / "fft.toml"
    cfg.write_text(_FFT_BODY)
    result = runner.invoke(
        app,
        ["train", str(cfg), "--yes"],
        env={**TEST_ENV, "PRIME_API_KEY": "test-key"},
    )
    return result, requested


def _filler_runs(count: int, *, prefix: str = "filler") -> list[dict[str, Any]]:
    """Fresh in-window runs that are not the lost dispatch."""
    return [_candidate_run(id=f"{prefix}-{i}", name=f"other-{i}") for i in range(count)]


def test_recovery_pages_past_the_first_twenty_runs(monkeypatch, tmp_path: Path) -> None:
    """A user who dispatches often can push the lost run off page one.

    Stopping at 20 reports "No matching run found yet", which is exactly
    the answer that invites the duplicate dispatch this path exists to
    prevent.
    """
    lost = _candidate_run()
    result, requested = _paging_timeout_dispatch(
        monkeypatch,
        tmp_path,
        pages=[_filler_runs(20), [*_filler_runs(4, prefix="p2"), lost]],
    )

    assert result.exit_code == 1, result.output
    assert requested == [1, 2], f"recovery must walk on to page 2:\n{requested}"
    assert "cyac46dphnrcw0pcbqcdg6od" in result.output, (
        f"the lost run was on page 2 and must still be named:\n{result.output}"
    )
    assert "No matching run found yet" not in result.output


def test_recovery_stops_at_one_page_in_the_common_case(monkeypatch, tmp_path: Path) -> None:
    """A short first page settles it — no extra round trips."""
    result, requested = _paging_timeout_dispatch(monkeypatch, tmp_path, pages=[[_candidate_run()]])

    assert result.exit_code == 1, result.output
    assert requested == [1], f"a partial page means there is no more to fetch:\n{requested}"
    assert "cyac46dphnrcw0pcbqcdg6od" in result.output


def test_recovery_stops_paging_once_past_the_recovery_window(monkeypatch, tmp_path: Path) -> None:
    """Runs are ordered createdAt DESC, so a pre-cutoff page ends the walk.

    Without this the walk would keep paging through a long run history
    that cannot possibly contain the dispatch.
    """
    from datetime import datetime, timedelta, timezone

    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    stale_page = [_candidate_run(id=f"old-{i}", name=f"old-{i}", createdAt=old) for i in range(20)]
    result, requested = _paging_timeout_dispatch(
        monkeypatch,
        tmp_path,
        pages=[stale_page, [_candidate_run()]],
        total=40,
    )

    assert result.exit_code == 1, result.output
    assert requested == [1], (
        f"the first page was already older than the cutoff; nothing newer "
        f"can follow it:\n{requested}"
    )
    assert "No matching run found yet" in result.output


def test_recovery_bounds_the_page_walk(monkeypatch, tmp_path: Path) -> None:
    """A backend that keeps answering must not turn recovery into a crawl.

    The user is already waiting on a failed command, so the walk is
    capped rather than trusting `total` to terminate it.
    """
    from prime_cli.commands.rl import _DISPATCH_RECOVERY_MAX_PAGES

    result, requested = _paging_timeout_dispatch(
        monkeypatch,
        tmp_path,
        pages=[_filler_runs(20, prefix=f"p{i}") for i in range(40)],
        total=10_000,
    )

    assert result.exit_code == 1, result.output
    assert requested == list(range(1, _DISPATCH_RECOVERY_MAX_PAGES + 1)), (
        f"the walk must stop at the cap:\n{requested}"
    )


def test_connect_timeout_is_a_plain_failure_not_an_ambiguous_dispatch(
    monkeypatch, tmp_path: Path
) -> None:
    """A connect timeout never transmitted, so no run was created.

    `APIClient.request` used to wrap ConnectTimeout in the same
    APITimeoutError as a read timeout, so this path told the user the
    outcome was unknown and not to re-run — about a dispatch that
    provably never happened.
    """
    from prime_cli.client import APIConnectTimeoutError

    looked_up: list[int] = []

    def mock_post(
        self: Any,
        endpoint: str,
        json: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> dict:
        raise APIConnectTimeoutError("Request timed out: timed out while connecting")

    def list_runs(self, team_id=None, *, mine=False, page=1, limit=20):
        looked_up.append(page)
        raise AssertionError("a connect timeout must not trigger a recovery lookup")

    monkeypatch.setattr("prime_cli.core.APIClient.post", mock_post)
    monkeypatch.setattr("prime_cli.api.rl.RLClient.list_runs", list_runs)

    cfg = tmp_path / "fft.toml"
    cfg.write_text(_FFT_BODY)
    result = runner.invoke(
        app,
        ["train", str(cfg), "--yes"],
        env={**TEST_ENV, "PRIME_API_KEY": "test-key"},
    )

    assert result.exit_code == 1, result.output
    assert looked_up == []
    assert "Error:" in result.output
    assert "Do not re-run this command yet" not in result.output, (
        f"nothing was transmitted, so re-running is safe:\n{result.output}"
    )


def test_connect_timeouts_are_raised_as_the_safe_subtype() -> None:
    """httpx collapses every timeout into TimeoutException; the client
    must preserve the distinction callers branch on."""
    import httpx as _httpx
    from prime_cli.core.client import (
        APIClient,
        APIConnectTimeoutError,
        APITimeoutError,
    )

    assert issubclass(APIConnectTimeoutError, APITimeoutError), (
        "existing `except APITimeoutError` handlers must keep catching it"
    )

    for exc, expected in (
        (_httpx.ConnectTimeout("connect"), APIConnectTimeoutError),
        (_httpx.PoolTimeout("pool"), APIConnectTimeoutError),
        (_httpx.ReadTimeout("read"), APITimeoutError),
        (_httpx.WriteTimeout("write"), APITimeoutError),
    ):
        client = APIClient(api_key="k")

        def boom(*a, _exc=exc, **kw):
            raise _exc

        client.client.request = boom  # type: ignore[method-assign]
        try:
            client.request("POST", "/x")
        except APITimeoutError as raised:
            assert type(raised) is expected, (
                f"{type(exc).__name__} must map to {expected.__name__}, got {type(raised).__name__}"
            )
        else:
            raise AssertionError(f"{type(exc).__name__} did not raise")

    # And the read/write pair must NOT be the safe subtype, or the
    # dispatch command would call an ambiguous outcome a clean failure.
    client = APIClient(api_key="k")

    def read_boom(*a, **kw):
        raise _httpx.ReadTimeout("read")

    client.client.request = read_boom  # type: ignore[method-assign]
    try:
        client.request("POST", "/x")
    except APIConnectTimeoutError:
        raise AssertionError("a read timeout must stay ambiguous")
    except APITimeoutError:
        pass
