"""The traces commands must build their client from the CLI config, so that
`prime --context <env> traces ...` talks to that context's deployment instead
of silently falling back to the SDK's static config."""

import json

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


def test_set_traces_url_with_context_is_read_only(monkeypatch, tmp_path):
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

    assert result.exit_code == 1, result.output
    assert "Temporary context 'staging' is read-only" in result.output
    assert json.loads(config.config_file.read_text()) == root_before
    assert json.loads(staging_file.read_text()) == staging_before


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
    assert "prime config use production" in result.output
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
        self.calls["upload_file"] = {"path": path, **kwargs}
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
    assert "Use --page 2 to see more." in result.output
    # The exact-boundary resume stays on offer next to the page hint.
    assert "--cursor cursor-1" in result.output
    call = fake_client.calls["list"]
    assert call["run_id"] == "run_9f3k2m"
    assert call["reward_min"] == 0.5
    assert call["limit"] == 10
    assert call["cursor"] is None


def test_list_command_first_page_without_more_has_no_footer(fake_client):
    fake_client.list = lambda **kwargs: TraceListPage(items=[_summary()], next_cursor=None)

    result = runner.invoke(main_app, ["traces", "list"])

    assert result.exit_code == 0, result.output
    assert "Page 1" not in result.output
    assert "--page" not in result.output


def test_list_command_cursor_resume_keeps_cursor_hint(fake_client):
    result = runner.invoke(main_app, ["traces", "list", "--cursor", "cursor-0"])

    assert result.exit_code == 0, result.output
    assert fake_client.calls["list"]["cursor"] == "cursor-0"
    assert "More results: --cursor cursor-1" in result.output
    assert "--page" not in result.output


def _paged_client(fake_client, pages):
    """Serve ``pages`` keyed by the cursor that reaches them; records each call."""
    calls = []

    def list_(**kwargs):
        calls.append(kwargs)
        items, next_cursor = pages[kwargs.get("cursor")]
        return TraceListPage(items=items, next_cursor=next_cursor)

    fake_client.list = list_
    return calls


def test_list_command_page_walks_cursors_to_the_requested_page(fake_client):
    calls = _paged_client(
        fake_client,
        {
            None: ([_summary(trace_id="p1a"), _summary(trace_id="p1b")], "c1"),
            "c1": ([_summary(trace_id="p2a"), _summary(trace_id="p2b")], "c2"),
            "c2": ([_summary(trace_id="p3a")], None),
        },
    )

    result = runner.invoke(main_app, ["traces", "list", "--page", "3", "--limit", "2"])

    assert result.exit_code == 0, result.output
    assert [call["cursor"] for call in calls] == [None, "c1", "c2"]
    assert all(call["limit"] == 2 for call in calls)
    assert "p3a" in result.output
    assert "p1a" not in result.output
    assert "p2a" not in result.output
    assert "Page 3 • showing 5-5" in result.output
    assert "--page 4" not in result.output


def test_list_command_page_with_more_pages_hints_the_next_page(fake_client):
    _paged_client(
        fake_client,
        {
            None: ([_summary(trace_id="p1a"), _summary(trace_id="p1b")], "c1"),
            "c1": ([_summary(trace_id="p2a"), _summary(trace_id="p2b")], "c2"),
        },
    )

    result = runner.invoke(main_app, ["traces", "list", "-p", "2", "--limit", "2"])

    assert result.exit_code == 0, result.output
    assert "Page 2 • showing 3-4" in result.output
    assert "Use --page 3 to see more." in result.output
    assert "--cursor c2" in result.output


def test_list_command_page_past_the_end_stops_walking(fake_client):
    calls = _paged_client(
        fake_client,
        {
            None: ([_summary(trace_id="p1a")], "c1"),
            "c1": ([_summary(trace_id="p2a")], None),
        },
    )

    result = runner.invoke(main_app, ["traces", "list", "--page", "5"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 2  # stops at the last real page instead of requesting five
    assert "No traces on page 5." in result.output
    assert "--page 1" in result.output
    assert "p2a" not in result.output
    assert "showing" not in result.output


def test_list_command_page_json_output_is_the_requested_page(fake_client):
    _paged_client(
        fake_client,
        {
            None: ([_summary(trace_id="p1a")], "c1"),
            "c1": ([_summary(trace_id="p2a")], "c2"),
        },
    )

    result = runner.invoke(main_app, ["traces", "list", "--page", "2", "-o", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [item["trace_id"] for item in payload["items"]] == ["p2a"]
    assert payload["next_cursor"] == "c2"


def test_list_command_rejects_page_with_cursor(fake_client):
    result = runner.invoke(main_app, ["traces", "list", "--page", "2", "--cursor", "c1"])

    assert result.exit_code == 1
    assert "--page cannot be combined with --cursor" in result.output
    assert "list" not in fake_client.calls


def test_list_command_rejects_page_below_one(fake_client):
    result = runner.invoke(main_app, ["traces", "list", "--page", "0"])

    assert result.exit_code == 1
    assert "--page must be at least 1" in result.output
    assert "list" not in fake_client.calls


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
