"""The traces commands must build their client from the CLI config, so that
`prime --context <env> traces ...` talks to that context's deployment instead
of silently falling back to the SDK's static config."""

import json
from unittest.mock import Mock

import pytest
from prime_cli.commands import traces as traces_cmd
from prime_cli.core import Config
from prime_cli.core.config import ConfigModel
from prime_cli.main import app as main_app
from prime_traces import APIError, Batch, TraceListPage, TraceSummary, UploadReceipt
from typer.testing import CliRunner

runner = CliRunner()


class _StubConfig:
    api_key = "ctx-key"
    traces_url = "https://traces.staging.primeintellect.ai"
    team_id = "team-ctx"


def test_traces_client_uses_cli_config(monkeypatch):
    monkeypatch.setattr(traces_cmd, "Config", _StubConfig)
    api = traces_cmd._traces_client().client
    assert api.api_key == "ctx-key"
    assert api.base_url == "https://traces.staging.primeintellect.ai"
    assert api.team_id == "team-ctx"
    assert api.client.headers["X-Prime-Team-ID"] == "team-ctx"


class _EmptyContextConfig:
    """A context with no API key and no team — must stay that way."""

    api_key = ""
    traces_url = "https://traces.ctx.primeintellect.ai"
    team_id = None


class _SdkFileConfig:
    """What the SDK's static ~/.prime/config.json would resolve to."""

    api_key = "file-key"
    traces_url = "https://file.primeintellect.ai"
    team_id = "file-team"


def test_empty_context_never_falls_back_to_sdk_config(monkeypatch):
    """An unset api_key/team in the active context must not re-resolve from
    the SDK's static config — that would attribute traffic to the default
    context's credentials and team."""
    import prime_traces.core.client as sdk_client

    monkeypatch.setattr(traces_cmd, "Config", _EmptyContextConfig)
    monkeypatch.setattr(sdk_client, "Config", _SdkFileConfig)

    api = traces_cmd._traces_client().client
    assert api.api_key == ""
    assert api.team_id == ""
    assert "Authorization" not in api.client.headers
    assert "X-Prime-Team-ID" not in api.client.headers


def test_config_model_round_trips_traces_url():
    """traces_url must survive the load path, which round-trips the config
    file through ConfigModel — a missing field there is silently dropped."""
    config = Config.__new__(Config)
    config.config = ConfigModel(traces_url="https://traces.x.ai/api/v1/").model_dump()
    assert config.traces_url == "https://traces.x.ai"


def test_cli_config_traces_url_precedence(monkeypatch):
    monkeypatch.delenv("PRIME_TRACES_URL", raising=False)
    monkeypatch.delenv("PRIME_API_BASE_URL", raising=False)
    monkeypatch.delenv("PRIME_BASE_URL", raising=False)

    # Bypass __init__ so the test never touches ~/.prime on the dev machine.
    config = Config.__new__(Config)
    config.config = {"base_url": "https://api.staging.primeintellect.ai"}

    # No traces_url anywhere: fall back to the context's platform base URL.
    assert config.traces_url == "https://api.staging.primeintellect.ai"

    # Context file value wins over the fallback; /api/v1 is normalized away
    # like base_url does (the client appends the prefix itself).
    config.config["traces_url"] = "https://traces.staging.primeintellect.ai/api/v1"
    assert config.traces_url == "https://traces.staging.primeintellect.ai"

    # Env var wins over everything.
    monkeypatch.setenv("PRIME_TRACES_URL", "http://localhost:8083")
    assert config.traces_url == "http://localhost:8083"


def test_traces_config_outputs_treat_url_as_literal_text(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    traces_url = "https://traces.example/[/]"

    configured = runner.invoke(main_app, ["config", "set-traces-url", traces_url])
    viewed = runner.invoke(main_app, ["config", "view"])
    viewed_plain = runner.invoke(main_app, ["config", "view", "--plain"])

    assert configured.exit_code == 0, configured.output
    assert traces_url in configured.output
    assert viewed.exit_code == 0, viewed.output
    assert traces_url in viewed.output
    assert viewed_plain.exit_code == 0, viewed_plain.output
    assert traces_url in viewed_plain.output


def test_set_traces_url_prompt_can_clear_existing_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    config = Config()
    config.set_traces_url("https://traces.example")

    result = runner.invoke(main_app, ["config", "set-traces-url"], input="-\n")

    assert result.exit_code == 0, result.output
    assert "override cleared" in result.output
    assert Config()._configured_traces_url() is None


def test_set_traces_url_persists_in_active_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    config = Config()
    config.save_environment("staging")
    config.load_environment("staging")

    traces_url = "https://traces.staging.example"
    result = runner.invoke(main_app, ["config", "set-traces-url", traces_url])

    assert result.exit_code == 0, result.output
    config = Config()
    config.load_environment("production")
    config.load_environment("staging")
    assert config.traces_url == traces_url


def test_set_traces_url_does_not_persist_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    config = Config()
    config.save_environment("staging")
    config.load_environment("staging")
    monkeypatch.setenv("PRIME_TRACES_URL", "https://temporary.example")

    traces_url = "https://traces.staging.example"
    result = runner.invoke(main_app, ["config", "set-traces-url", traces_url])

    assert result.exit_code == 0, result.output
    monkeypatch.delenv("PRIME_TRACES_URL")
    config = Config()
    config.load_environment("production")
    config.load_environment("staging")
    assert config.traces_url == traces_url


def test_set_traces_url_with_context_does_not_replace_default_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    # The CLI callback sets this directly; record the pre-test value so the
    # monkeypatch fixture restores it even after the callback replaces it.
    monkeypatch.setenv("PRIME_CONTEXT", "")
    config = Config()
    config.set_api_key("staging-key")
    config.set_base_url("https://api.staging.example")
    config.save_environment("staging")
    staging_file = config.environments_dir / "staging.json"
    staging_before = json.loads(staging_file.read_text())

    config.load_environment("production")
    config.set_api_key("production-key")
    root_before = json.loads(config.config_file.read_text())

    traces_url = "https://traces.staging.example"
    result = runner.invoke(
        main_app,
        ["--context", "staging", "config", "set-traces-url", traces_url],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(config.config_file.read_text()) == root_before
    assert json.loads(staging_file.read_text()) == {
        **staging_before,
        "traces_url": traces_url,
    }


def test_set_traces_url_does_not_persist_unrelated_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    config = Config()
    config.set_api_key("saved-key")
    config.set_team("saved-team")
    config.set_base_url("https://api.saved.example")
    config.save_environment("staging")
    config.load_environment("staging")
    staging_file = config.environments_dir / "staging.json"
    staging_before = json.loads(staging_file.read_text())

    monkeypatch.setenv("PRIME_API_KEY", "temporary-key")
    monkeypatch.setenv("PRIME_TEAM_ID", "temporary-team")
    monkeypatch.setenv("PRIME_API_BASE_URL", "https://api.temporary.example")

    traces_url = "https://traces.staging.example"
    result = runner.invoke(main_app, ["config", "set-traces-url", traces_url])

    assert result.exit_code == 0, result.output
    assert json.loads(staging_file.read_text()) == {
        **staging_before,
        "traces_url": traces_url,
    }


def test_set_traces_url_rejects_unstored_production_context(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setenv("PRIME_CONTEXT", "")
    config = Config()
    config.save_environment("staging")
    config.load_environment("staging")
    root_before = json.loads(config.config_file.read_text())

    result = runner.invoke(
        main_app,
        [
            "--context",
            "production",
            "config",
            "set-traces-url",
            "https://traces.production.example",
        ],
    )

    assert result.exit_code == 1
    assert "run 'prime config use production' first" in result.output
    assert json.loads(config.config_file.read_text()) == root_before


def test_logout_preserves_active_environment_traces_url(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    config = Config()
    config.set_api_key("secret")
    config.set_traces_url("https://traces.staging.example")
    config.save_environment("staging")
    config.load_environment("staging")

    result = runner.invoke(main_app, ["logout", "--yes"])

    assert result.exit_code == 0, result.output
    config = Config()
    config.load_environment("production")
    config.load_environment("staging")
    assert config.traces_url == "https://traces.staging.example"


# ---------------------------------------------------------------------------
# Command smoke tests: every `prime traces` command exercised through Typer
# with a stubbed TracesClient, mirroring test_tunnel_cli.py. These catch
# signature drift between the CLI options and the SDK methods, and pin that
# `--output json` emits parseable JSON and nothing else.
# ---------------------------------------------------------------------------


def _summary(**overrides):
    fields = {
        "trace_id": "8d3f1a2b",
        "upload_id": "5ee85e41",
        "episode_id": None,
        "created_at": "2026-07-20T18:02:11.482Z",
        "ingested_at": "2026-07-20T18:06:02.117Z",
        "run_id": "run_9f3k2m",
        "environment_id": None,
        "model": {"provider": "prime", "id": "deepseek-v4-flash"},
        "task_id": "tb2-0187",
        "agent_name": "solver",
        "score": {"reward": 0.85, "outcome": "done"},
        "execution": {"has_error": False, "is_truncated": False},
        "duration_ms": 215537,
        "total_tokens": 84213,
        "size_bytes": 417284,
        "context": {"source": "hosted_eval"},
    }
    fields.update(overrides)
    return TraceSummary.model_validate(fields)


class FakeTracesClient:
    def __init__(self):
        self.calls: dict = {}
        self.receipt = UploadReceipt(upload_id="a" * 64, status="committed")

    def upload_file(self, path, **kwargs):
        self.calls["upload_file"] = {"path": path, "data": path.read_bytes(), **kwargs}
        on_batch = kwargs.get("on_batch")
        batch = Batch(data=b"{}\n", digest="a" * 64, num_lines=1, first_line_number=1)
        if on_batch is not None:
            on_batch(batch, self.receipt)
        return [self.receipt]

    def list(self, **kwargs):
        self.calls["list"] = kwargs
        return TraceListPage(items=[_summary()], next_cursor="cursor-1")

    def get(self, trace_id):
        self.calls["get"] = trace_id
        return _summary(trace_id=trace_id)

    def get_raw(self, trace_id):
        self.calls["get_raw"] = trace_id
        return b'{"version":4,"id":"%s"}' % trace_id.encode()

    def download_raw(self, trace_id, dest):
        self.calls["download_raw"] = (trace_id, dest)
        return 29

    def delete(self, trace_id, created_at=None):
        self.calls["delete"] = (trace_id, created_at)

    def delete_run(self, run_id):
        self.calls["delete_run"] = run_id


@pytest.fixture()
def fake_client(monkeypatch):
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    client = FakeTracesClient()
    monkeypatch.setattr(traces_cmd, "_traces_client", lambda: client)
    return client


def test_upload_command_table_output(fake_client, tmp_path):
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_bytes(b'{"id":"a"}\n')

    result = runner.invoke(
        main_app,
        ["traces", "upload", str(traces_file), "-c", "source=hosted_eval", "-c", "suite=s1"],
    )

    assert result.exit_code == 0, result.output
    assert "Uploaded 1 batch(es)" in result.output
    call = fake_client.calls["upload_file"]
    assert call["path"].parent.parent == tmp_path.resolve()
    assert call["context"] == {"source": "hosted_eval", "suite": "s1"}
    assert call["compress"] is True
    assert call["line_format"].value == "trace"


def test_upload_command_episodes_json_output(fake_client, tmp_path):
    traces_file = tmp_path / "episodes.jsonl"
    traces_file.write_bytes(b'{"id":"ep"}\n')

    result = runner.invoke(
        main_app,
        ["traces", "upload", str(traces_file), "--episodes", "--no-compress", "-o", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["num_batches"] == 1
    assert payload["receipts"][0]["status"] == "committed"
    call = fake_client.calls["upload_file"]
    assert call["line_format"].value == "episode"
    assert call["compress"] is False


def test_upload_command_rejects_malformed_context(fake_client, tmp_path):
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_bytes(b'{"id":"a"}\n')

    result = runner.invoke(main_app, ["traces", "upload", str(traces_file), "-c", "no-equals"])

    assert result.exit_code == 1
    assert "upload_file" not in fake_client.calls


def test_upload_command_redacts_a_copy_and_keeps_review_data(fake_client, tmp_path):
    secret = "opaque-user-key-0123456789"
    traces_file = tmp_path / "traces.jsonl"
    original = (
        json.dumps(
            {
                "prompt": f"accidentally repeated {secret}",
                "answer": "reference answer",
                "rubric": "compare against the reference answer",
            }
        ).encode()
        + b"\n"
    )
    traces_file.write_bytes(original)
    secrets_file = tmp_path / "secrets.txt"
    secrets_file.write_text(secret + "\n")

    result = runner.invoke(
        main_app,
        [
            "traces",
            "upload",
            str(traces_file),
            "-c",
            f"authorization={secret}",
            "--secrets-file",
            str(secrets_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Preflight redacted 2 credential-bearing location(s)" in result.output
    assert secret not in fake_client.calls["upload_file"]["data"].decode()
    assert fake_client.calls["upload_file"]["context"] == {"authorization": "[REDACTED]"}
    uploaded = json.loads(fake_client.calls["upload_file"]["data"])
    assert uploaded["answer"] == "reference answer"
    assert uploaded["rubric"] == "compare against the reference answer"
    assert traces_file.read_bytes() == original


def test_upload_command_falls_back_from_read_only_source_directory(
    fake_client, tmp_path, monkeypatch
):
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_text('{"answer":"keep"}\n')
    real_temporary_directory = traces_cmd.tempfile.TemporaryDirectory
    fallback = real_temporary_directory(prefix="prime-traces-upload-test-")
    temporary_directory = Mock(side_effect=[PermissionError(), fallback])
    monkeypatch.setattr(traces_cmd.tempfile, "TemporaryDirectory", temporary_directory)

    result = runner.invoke(main_app, ["traces", "upload", str(traces_file)])

    assert result.exit_code == 0, result.output
    assert temporary_directory.call_args_list[0].kwargs["dir"] == traces_file.resolve().parent
    assert "dir" not in temporary_directory.call_args_list[1].kwargs


def test_upload_command_fails_before_client_creation_on_invalid_json(tmp_path, monkeypatch):
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_text('{"password":}\n')

    def fail_if_called():
        raise AssertionError("client must not be created before preflight succeeds")

    monkeypatch.setattr(traces_cmd, "_traces_client", fail_if_called)
    result = runner.invoke(main_app, ["traces", "upload", str(traces_file)])

    assert result.exit_code == 1
    assert "Preflight failed: invalid JSON on JSONL line 1" in result.output
    assert not isinstance(result.exception, AssertionError)


def test_unexpected_error_does_not_dump_sdk_locals(fake_client, tmp_path):
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_bytes(b'{"id":"a"}\n')

    def fail_upload(*args, **kwargs):
        secret_trace = "sensitive-trace-" + "payload"
        assert secret_trace
        raise RuntimeError("malformed receipt")

    fake_client.upload_file = fail_upload
    result = runner.invoke(main_app, ["traces", "upload", str(traces_file)])

    assert result.exit_code == 1
    assert "Unexpected error: malformed receipt" in result.output
    assert "sensitive-trace-payload" not in result.output


def test_list_command_forwards_filters_and_renders_table(fake_client):
    result = runner.invoke(
        main_app,
        ["traces", "list", "--run-id", "run_9f3k2m", "--reward-min", "0.5", "--limit", "10"],
    )

    assert result.exit_code == 0, result.output
    assert "8d3f1a2b" in result.output
    assert "cursor-1" in result.output  # next-page hint
    call = fake_client.calls["list"]
    assert call["run_id"] == "run_9f3k2m"
    assert call["reward_min"] == 0.5
    assert call["limit"] == 10


def test_list_command_renders_full_trace_id(fake_client):
    trace_id = "trace-0123456789abcdef0123456789abcdef"
    fake_client.list = lambda **kwargs: TraceListPage(
        items=[_summary(trace_id=trace_id)],
        next_cursor=None,
    )

    result = runner.invoke(main_app, ["traces", "list"])

    assert result.exit_code == 0, result.output
    assert trace_id in result.output


def test_list_command_json_output_is_parseable(fake_client):
    result = runner.invoke(main_app, ["traces", "list", "-o", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["items"][0]["trace_id"] == "8d3f1a2b"
    assert payload["next_cursor"] == "cursor-1"


def test_api_error_messages_are_rendered_as_literal_text(fake_client):
    def fail_list(**kwargs):
        raise APIError("invalid filter [/]")

    fake_client.list = fail_list

    result = runner.invoke(main_app, ["traces", "list"])

    assert result.exit_code == 1
    assert "Error: invalid filter [/]" in result.output
    assert "MarkupError" not in result.output


def test_json_error_keeps_stdout_machine_readable(fake_client):
    def fail_list(**kwargs):
        raise APIError("invalid filter")

    fake_client.list = fail_list

    result = runner.invoke(main_app, ["traces", "list", "-o", "json"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Error: invalid filter" in result.stderr


def test_table_output_treats_trace_values_as_literal_text(fake_client):
    markup = "[/]"
    fake_client.list = lambda **kwargs: TraceListPage(
        items=[
            _summary(
                trace_id=markup,
                run_id=markup,
                task_id=markup,
                score={"reward": 0.85, "outcome": markup},
            )
        ],
        next_cursor=None,
    )
    fake_client.get = lambda trace_id: _summary(trace_id=trace_id, run_id=markup)

    listed = runner.invoke(main_app, ["traces", "list"])
    fetched = runner.invoke(main_app, ["traces", "get", "[red]trace"])

    assert listed.exit_code == 0, listed.output
    assert markup in listed.output
    assert fetched.exit_code == 0, fetched.output
    assert "[red]trace" in fetched.output
    assert markup in fetched.output


def test_get_command_summary_and_raw(fake_client):
    result = runner.invoke(main_app, ["traces", "get", "8d3f1a2b"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls["get"] == "8d3f1a2b"

    result = runner.invoke(main_app, ["traces", "get", "8d3f1a2b", "--raw"])
    assert result.exit_code == 0, result.output
    assert '"version":4' in result.output
    assert fake_client.calls["get_raw"] == "8d3f1a2b"


def test_get_command_raw_stdout_preserves_exact_bytes(fake_client):
    raw = b'{"version":4}\n\xff'
    fake_client.get_raw = lambda trace_id: raw

    result = runner.invoke(main_app, ["traces", "get", "8d3f1a2b", "--raw"])

    assert result.exit_code == 0
    assert result.stdout_bytes == raw


def test_get_command_raw_error_keeps_stdout_clean(fake_client):
    def fail_get_raw(trace_id):
        raise APIError(f"trace {trace_id} not found")

    fake_client.get_raw = fail_get_raw

    result = runner.invoke(main_app, ["traces", "get", "missing", "--raw"])

    assert result.exit_code == 1
    assert result.stdout_bytes == b""
    assert "Error: trace missing not found" in result.stderr


def test_get_command_rejects_dest_without_raw(fake_client, tmp_path):
    dest = tmp_path / "trace.json"

    result = runner.invoke(main_app, ["traces", "get", "8d3f1a2b", "--dest", str(dest)])

    assert result.exit_code == 1
    assert "--dest requires --raw" in result.output
    assert "get" not in fake_client.calls
    assert not dest.exists()


def test_get_command_raw_to_dest_streams(fake_client, tmp_path):
    dest = tmp_path / "trace.json"
    result = runner.invoke(main_app, ["traces", "get", "8d3f1a2b", "--raw", "--dest", str(dest)])

    assert result.exit_code == 0, result.output
    trace_id, streamed_dest = fake_client.calls["download_raw"]
    assert trace_id == "8d3f1a2b"
    assert streamed_dest == dest


def test_get_command_raw_to_dest_honors_json_output(fake_client, tmp_path):
    dest = tmp_path / "trace.json"
    result = runner.invoke(
        main_app,
        ["traces", "get", "8d3f1a2b", "--raw", "--dest", str(dest), "-o", "json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"dest": str(dest), "bytes_written": 29}


def test_export_command_is_not_registered(fake_client, tmp_path):
    """Exports are unimplemented server-side (every handler raises, answering
    500), so the command is deliberately absent rather than shipped broken."""
    result = runner.invoke(main_app, ["traces", "export", str(tmp_path / "out.jsonl")])
    assert result.exit_code != 0


def test_delete_command_requires_exactly_one_target(fake_client):
    both = runner.invoke(
        main_app, ["traces", "delete", "8d3f1a2b", "--run-id", "run_9f3k2m", "--yes"]
    )
    neither = runner.invoke(main_app, ["traces", "delete", "--yes"])

    assert both.exit_code == 1
    assert neither.exit_code == 1
    assert "delete" not in fake_client.calls
    assert "delete_run" not in fake_client.calls


def test_delete_command_trace_and_run(fake_client):
    result = runner.invoke(main_app, ["traces", "delete", "8d3f1a2b", "--yes"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls["delete"] == ("8d3f1a2b", None)

    result = runner.invoke(main_app, ["traces", "delete", "--run-id", "run_9f3k2m", "--yes"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls["delete_run"] == "run_9f3k2m"
    assert "accepted" in result.output


def test_delete_success_treats_target_as_literal_text(fake_client):
    result = runner.invoke(main_app, ["traces", "delete", "[red]", "--yes"])

    assert result.exit_code == 0, result.output
    assert fake_client.calls["delete"] == ("[red]", None)
    assert "Deletion of trace [red] accepted" in result.output
