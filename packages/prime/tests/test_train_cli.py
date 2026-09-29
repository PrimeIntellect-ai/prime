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
        assert "stop accepting new runs on" in result.output
        assert "October 5, 2026" in result.output
        assert "Hosted Training is transitioning to dedicated runs." in result.output
        assert Path("rl.toml").exists()
        assert "stop accepting new runs on October 5, 2026" in Path("rl.toml").read_text()
        assert (
            "# Hosted Training is transitioning to dedicated runs." in Path("rl.toml").read_text()
        )


def test_train_legacy_run_warns_before_confirmation_without_warning_fft(
    monkeypatch, tmp_path: Path
) -> None:
    from prime_cli.commands import rl

    legacy_config = tmp_path / "legacy.toml"
    legacy_config.write_text(rl.generate_rl_config_template())
    monkeypatch.setattr(rl, "APIClient", lambda: object())
    monkeypatch.setattr(
        rl, "RLClient", lambda _: type("Client", (), {"list_models": lambda self, team_id: []})()
    )
    monkeypatch.setattr(rl, "Config", lambda: type("Config", (), {"team_id": None})())
    seen: list[str] = []

    def cancel(_message: str, _yes: bool, *, default: bool) -> bool:
        seen.append("confirm")
        return False

    monkeypatch.setattr(rl, "confirm_or_skip", cancel)
    result = runner.invoke(app, ["train", str(legacy_config)], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "stop accepting new runs on" in result.output
    assert "October 5, 2026" in result.output
    assert "Hosted Training is transitioning to dedicated runs." in result.output
    assert result.output.index("stop accepting new runs") < result.output.index("Configuration:")
    assert seen == ["confirm"]

    fft_config = tmp_path / "fft.toml"
    fft_config.write_text(
        '[model]\nname = "Qwen/Qwen3-0.6B"\n[deployment]\nnum_train_gpus = 1\nnum_infer_gpus = 1\n'
    )
    monkeypatch.setattr(rl, "_dispatch_full_finetune_run", lambda **kwargs: None)
    result = runner.invoke(app, ["train", str(fft_config)], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "stop accepting new runs" not in result.output


def test_train_legacy_json_warning_uses_stderr(monkeypatch, tmp_path: Path) -> None:
    from prime_cli.commands import rl

    config = tmp_path / "legacy.toml"
    config.write_text(rl.generate_rl_config_template())
    monkeypatch.setattr(rl, "APIClient", lambda: object())
    monkeypatch.setattr(
        rl, "RLClient", lambda _: type("Client", (), {"list_models": lambda self, team_id: []})()
    )
    monkeypatch.setattr(rl, "Config", lambda: type("Config", (), {"team_id": None})())
    monkeypatch.setattr(rl, "confirm_or_skip", lambda *args, **kwargs: False)
    result = runner.invoke(app, ["train", str(config), "--output", "json"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "stop accepting new runs on October 5, 2026" in result.stderr
    assert "Hosted Training is transitioning to dedicated runs." in result.stderr
    assert "stop accepting new runs" not in result.stdout


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
    assert "Volume 'ckpts' doesn't exist, creating it (1Ti)" in result.output
    assert creates == [("ckpts", "1Ti")] and len(dispatched) == 1


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


def test_train_help_marks_source_overlay_flags_internal() -> None:
    result = runner.invoke(app, ["train", "--help"], env=TEST_ENV)
    assert result.exit_code == 0
    text = " ".join(result.output.split())
    assert "--ref" in text and "--pr" in text
    assert text.count("Internal only") == 2


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
        "-x",
        "x.lock",
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


def test_train_source_overlay_is_rejected_on_the_lora_path(monkeypatch, tmp_path: Path) -> None:
    def never(pr_number: int) -> str:
        raise AssertionError("the LoRA path must not resolve a PR")

    monkeypatch.setattr("prime_cli.api.training.resolve_pull_request_head", never)
    lora = '[model]\nname = "Qwen/Qwen3-0.6B"\n'
    cfg = tmp_path / "rl.toml"
    for body, args in (
        (lora, ["--ref", "feat/x"]),
        (lora, ["--pr", "42"]),
        ('source_ref = "feat/x"\n' + lora, []),
    ):
        cfg.write_text(body)
        result = runner.invoke(app, ["train", str(cfg), *args, "-y"], env=TEST_ENV)
        assert result.exit_code == 1, args
        assert "--ref / --pr" in result.output and "full-FT" in result.output, args


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
        # A 403 that isn't a quota is a refusal, reported with GitHub's reason.
        (403, {}, {"message": "Resource protected by organization SAML"}, r"refused.*SAML"),
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
