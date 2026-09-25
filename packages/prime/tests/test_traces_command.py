"""The traces commands must build their client from the CLI config, so that
`prime --context <env> traces ...` talks to that context's deployment instead
of silently falling back to the SDK's static config."""

import json

import pytest
from prime_cli.commands import traces as traces_cmd
from prime_cli.core import Config
from prime_cli.core.config import ConfigModel
from prime_cli.main import app as main_app
from prime_traces import (
    APIError,
    Batch,
    EpisodeDetail,
    EpisodeListPage,
    EpisodeSummary,
    NotFoundError,
    TraceListPage,
    TraceSummary,
    UploadReceipt,
)
from typer.testing import CliRunner

runner = CliRunner()


def test_search_json_preserves_continuation_coverage_and_scope(monkeypatch):
    from prime_traces import TraceSearchCoverage, TraceSearchPage

    calls = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def search(self, query, **kwargs):
            calls.append((query, kwargs))
            return TraceSearchPage(
                items=[],
                next_cursor="cursor",
                coverage=TraceSearchCoverage(
                    examined_traces=256,
                    unindexed_trace_ids=["pending"],
                    partial_index=False,
                ),
            )

    monkeypatch.setattr(traces_cmd, "_traces_client", Client)
    result = runner.invoke(
        traces_cmd.app,
        ["search", "connection refused", "--run-id", "run", "--run-step", "0", "-o", "json"],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["next_cursor"] == "cursor"
    assert body["coverage"]["unindexed_trace_ids"] == ["pending"]
    assert len(calls) == 1 and calls[0][0] == "connection refused"
    assert calls[0][1]["run_id"] == "run" and calls[0][1]["run_step"] == 0


@pytest.mark.parametrize(
    "args",
    [
        ["search", "hello"],
        ["search", " ", "--run-id", "run"],
        ["search", "he", "--run-id", "run"],
        ["search", "hello", "--run-id", "team/run"],
        ["search", "hello", "--run-id", "run", "--field", "sql"],
    ],
)
def test_search_rejects_invalid_arguments_before_network(monkeypatch, args):
    def fail():
        pytest.fail("invalid input must not make a network request")

    monkeypatch.setattr(traces_cmd, "_traces_client", fail)
    assert runner.invoke(traces_cmd.app, args).exit_code != 0


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

    # No traces_url anywhere: the traces service default, never the platform
    # base URL (which does not serve /api/v1/traces), even with a base_url override.
    assert config.traces_url == "https://prime-traces.pintel.dev"
    assert config.traces_url == Config.DEFAULT_TRACES_URL

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


def _episode(**overrides):
    fields = {
        "episode_id": "ep_4c1d",
        "upload_id": "5ee85e41",
        "schema_version": 1,
        "created_at": "2026-07-20T18:02:11.482Z",
        "ingested_at": "2026-07-20T18:06:02.117Z",
        "run_id": "run_9f3k2m",
        "environment_id": "tb2",
        "outcome": "failed",
        "has_error": True,
        "error": {"type": "EnvHookError", "message": "teardown timed out"},
    }
    fields.update(overrides)
    return EpisodeSummary.model_validate(fields)


def _episode_detail(**overrides):
    traces = {
        "trace_count": 1,
        "total_tokens": 84213,
        "total_duration_ms": 215537,
        "any_trace_error": False,
        "agent_names": ["solver"],
    }
    traces.update(overrides.pop("traces", {}))
    return EpisodeDetail.model_validate(
        {**_episode(**overrides).model_dump(mode="json"), "traces": traces}
    )


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

    def list_episodes(self, **kwargs):
        self.calls["list_episodes"] = kwargs
        return EpisodeListPage(items=[_episode()], next_cursor="ep-cursor-1")

    def get_episode(self, episode_id):
        self.calls["get_episode"] = episode_id
        return _episode_detail(episode_id=episode_id)

    def get_episode_raw(self, episode_id):
        self.calls["get_episode_raw"] = episode_id
        return b'{"id":"%s","traces":["8d3f1a2b"]}' % episode_id.encode()

    def list_episode_traces(self, episode_id, **kwargs):
        self.calls["list_episode_traces"] = {"episode_id": episode_id, **kwargs}
        return TraceListPage(
            items=[_summary(episode_id=episode_id, activity={"model_turns": 12})],
            next_cursor=None,
        )


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


@pytest.mark.parametrize("code", [None, "run_not_found", "search_limit_exceeded"])
def test_search_error_keeps_json_stdout_clean(monkeypatch, code):
    from prime_traces import NotFoundError

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def search(self, *args, **kwargs):
            raise NotFoundError("Run [red]missing[/] not found", code=code)

    monkeypatch.setattr(traces_cmd, "_traces_client", Client)
    result = runner.invoke(traces_cmd.app, ["search", "hello", "--run-id", "run", "-o", "json"])
    assert result.exit_code == 1
    assert result.stdout_bytes == b""
    if code is None:
        assert "requires the Prime Traces search API" in result.stderr
    else:
        assert "Search failed: Run [red]missing[/] not found" in result.stderr
        assert "requires the Prime Traces search API" not in result.stderr


@pytest.mark.parametrize("cursor", ["cursor[red]opaque[/red]", "cursor[/]"])
@pytest.mark.parametrize("output", ["table", "json"])
def test_search_preserves_opaque_cursor(monkeypatch, cursor, output):
    from prime_traces import TraceSearchPage

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def search(self, *args, **kwargs):
            return TraceSearchPage(
                items=[],
                next_cursor=cursor,
                coverage=None,
            )

    monkeypatch.setattr(traces_cmd, "_traces_client", Client)
    result = runner.invoke(traces_cmd.app, ["search", "hello", "--run-id", "run", "-o", output])
    assert result.exit_code == 0, result.output
    if output == "json":
        assert json.loads(result.stdout)["next_cursor"] == cursor
    else:
        assert f"--cursor {cursor}" in result.stdout


@pytest.mark.parametrize(
    "args,warns",
    [
        ([], True),  # first page without coverage: completeness is unknown
        (["--cursor", "resume"], False),  # resumed pages omit coverage by design
    ],
)
def test_search_warns_when_first_page_coverage_is_unknown(monkeypatch, args, warns):
    from prime_traces import TraceSearchPage

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def search(self, *args, **kwargs):
            return TraceSearchPage(items=[], next_cursor=None, coverage=None)

    monkeypatch.setattr(traces_cmd, "_traces_client", Client)
    result = runner.invoke(traces_cmd.app, ["search", "hello", "--run-id", "run", *args])
    assert result.exit_code == 0, result.output
    assert ("Index coverage is unknown" in result.stderr) is warns
    assert "Search exhausted" in result.stdout


def _search_match(text, query="the", trace_id="trace", node_idx=1, role="user"):
    from prime_traces import TraceSearchMatch

    position = text.index(query)
    start = max(0, position - 64)
    return TraceSearchMatch(
        trace_id=trace_id,
        upload_id="upload",
        generation=1,
        episode_id=None,
        node_idx=node_idx,
        role=role,
        field="content",
        representation="text",
        match_start=position,
        match_end=position + len(query),
        excerpt=text[start : start + len(query) + 128],
        excerpt_start=start,
    )


@pytest.mark.parametrize(
    "text,query,highlighted,plain",
    [
        ("x" * 70 + "the literal\nlives here " + "y" * 80, "the", "the", "…"),  # clipped edges
        ("before check the\nstatus after", "check the\nstatus", "check the status", "before"),
    ],
)
def test_search_excerpt_highlights_by_offset(text, query, highlighted, plain):
    rendered = traces_cmd._search_excerpt(_search_match(text, query), 60)
    spans = [span for span in rendered.spans if "yellow" in str(span.style)]
    assert [rendered.plain[span.start : span.end] for span in spans] == [highlighted]
    assert rendered.plain.startswith(plain) and "\n" not in rendered.plain


def test_search_excerpt_ignores_offsets_outside_the_excerpt():
    match = _search_match("the end").model_copy(update={"match_start": 40, "match_end": 43})
    rendered = traces_cmd._search_excerpt(match, 60)
    assert rendered.plain == "the end" and not rendered.spans


@pytest.mark.parametrize("width", [80, 50])  # narrow terminals keep the match visible
def test_search_table_groups_rows_by_trace(monkeypatch, width):
    from prime_traces import TraceSearchCoverage, TraceSearchPage

    monkeypatch.setattr(traces_cmd.console, "width", width)

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def search(self, *args, **kwargs):
            return TraceSearchPage(
                items=[
                    _search_match("first the one", trace_id="aaaa", role="system"),
                    _search_match("second the two", trace_id="aaaa", node_idx=7),
                    _search_match("third the three", trace_id="bbbb", role="tool"),
                ],
                next_cursor=None,
                coverage=TraceSearchCoverage(
                    examined_traces=9, unindexed_trace_ids=[], partial_index=False
                ),
            )

    monkeypatch.setattr(traces_cmd, "_traces_client", Client)
    result = runner.invoke(traces_cmd.app, ["search", "the", "--run-id", "run"])
    assert result.exit_code == 0, result.output
    assert result.stdout.count("aaaa") == 1 and result.stdout.count("bbbb") == 1
    assert "the one" in result.stdout
    assert "3 matches in 2 traces on this page · 9 traces" in result.stdout


# ---------------------------------------------------------------------------
# Episodes: `prime traces episodes list|get` and `prime traces list --episode`
# ---------------------------------------------------------------------------


def test_episodes_list_forwards_filters_and_moves_the_run_to_the_title(fake_client):
    result = runner.invoke(
        main_app,
        [
            "traces",
            "episodes",
            "list",
            "--run-id",
            "run_9f3k2m",
            "--env",
            "tb2",
            "--outcome",
            "failed",
            "--has-error",
            "--limit",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_client.calls["list_episodes"] == {
        "run_id": "run_9f3k2m",
        "environment_id": "tb2",
        "outcome": "failed",
        "has_error": True,
        "created_after": None,
        "created_before": None,
        "limit": 10,
        "cursor": None,
    }
    assert "Episodes · run run_9f3k2m · tb2" in result.output
    assert "Run " not in result.output  # the run is in the title, not a column
    assert "ep_4c1d" in result.output
    assert "EnvHookError" in result.output
    assert "teardown timed out" not in result.output  # the message is left to `get`
    assert "2026-07-20 18:02:11Z" in result.output
    assert "Use --page 2 to see more." in result.output
    assert "--cursor ep-cursor-1" in result.output
    assert "prime traces episodes get <episode_id>" in result.output


def test_episodes_list_moves_shared_values_to_the_title_and_keeps_differing_ones(fake_client):
    fake_client.list_episodes = lambda **kwargs: EpisodeListPage(
        items=[
            _episode(episode_id="ep_a", run_id="run_a", has_error=False),
            _episode(episode_id="ep_b", run_id="run_b", has_error=False),
        ],
        next_cursor=None,
    )

    result = runner.invoke(main_app, ["traces", "episodes", "list"])

    assert result.exit_code == 0, result.output
    assert "Episodes · tb2" in result.output  # every row shares the environment
    assert "Environment" not in result.output
    assert "Run" in result.output  # the runs differ, so they keep their column
    assert "run_a" in result.output and "run_b" in result.output
    assert "--page" not in result.output


def test_episodes_list_page_walks_cursors_and_stops_past_the_end(fake_client):
    calls = []
    pages = {None: ([_episode(episode_id="ep_p1")], "c1"), "c1": ([], None)}

    def list_episodes(**kwargs):
        calls.append(kwargs)
        items, next_cursor = pages[kwargs["cursor"]]
        return EpisodeListPage(items=items, next_cursor=next_cursor)

    fake_client.list_episodes = list_episodes

    result = runner.invoke(main_app, ["traces", "episodes", "list", "--page", "4"])

    assert result.exit_code == 0, result.output
    assert [call["cursor"] for call in calls] == [None, "c1"]
    assert "No episodes on page 4." in result.output
    assert "ep_p1" not in result.output


def test_episodes_list_rejects_page_with_cursor(fake_client):
    result = runner.invoke(
        main_app, ["traces", "episodes", "list", "--page", "2", "--cursor", "c1"]
    )

    assert result.exit_code == 1
    assert "--page cannot be combined with --cursor" in result.output
    assert "list_episodes" not in fake_client.calls


def test_episodes_list_json_output_is_the_page(fake_client):
    result = runner.invoke(main_app, ["traces", "episodes", "list", "-o", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["items"][0]["episode_id"] == "ep_4c1d"
    assert payload["items"][0]["error"]["type"] == "EnvHookError"
    assert payload["next_cursor"] == "ep-cursor-1"


def test_episodes_get_shows_both_error_sources_and_member_traces(fake_client):
    result = runner.invoke(main_app, ["traces", "episodes", "get", "ep_4c1d"])

    assert result.exit_code == 0, result.output
    assert fake_client.calls["get_episode"] == "ep_4c1d"
    assert fake_client.calls["list_episode_traces"] == {"episode_id": "ep_4c1d", "limit": 20}
    # An environment hook failure with every trace green: both flags are shown.
    assert "episode error yes" in result.output
    assert "trace errors no" in result.output
    assert "EnvHookError: teardown timed out" in result.output
    assert "task         tb2-0187" in result.output
    assert "8d3f1a2b" in result.output
    assert "solver" in result.output
    assert "84,213" in result.output
    assert "prime traces transcript <trace_id>" in result.output
    assert "prime traces list --episode" not in result.output


def test_episodes_get_points_at_the_full_member_listing(fake_client):
    fake_client.get_episode = lambda episode_id: _episode_detail(
        episode_id=episode_id, traces={"trace_count": 45}
    )
    fake_client.list_episode_traces = lambda episode_id, **kwargs: TraceListPage(
        items=[_summary(trace_id=f"t{i}") for i in range(20)], next_cursor="more"
    )

    result = runner.invoke(main_app, ["traces", "episodes", "get", "ep_4c1d"])

    assert result.exit_code == 0, result.output
    assert "Showing the newest 20 of 45 traces." in result.output
    assert "prime traces list --episode ep_4c1d" in result.output


def test_episodes_get_lists_member_traces_in_the_order_they_ran(fake_client):
    # The service returns newest first; the episode view reads oldest first.
    fake_client.list_episode_traces = lambda episode_id, **kwargs: TraceListPage(
        items=[
            _summary(trace_id="t-late", agent_name="reviewer"),
            _summary(trace_id="t-early", agent_name="planner"),
        ],
        next_cursor=None,
    )

    result = runner.invoke(main_app, ["traces", "episodes", "get", "ep_4c1d"])

    assert result.exit_code == 0, result.output
    assert result.output.index("t-early") < result.output.index("t-late")


def test_episodes_get_omits_the_task_when_a_member_records_none(fake_client):
    fake_client.list_episode_traces = lambda episode_id, **kwargs: TraceListPage(
        items=[_summary(trace_id="t1"), _summary(trace_id="t2", task_id=None)],
        next_cursor=None,
    )

    result = runner.invoke(main_app, ["traces", "episodes", "get", "ep_4c1d"])

    assert result.exit_code == 0, result.output
    assert "tb2-0187" not in result.output


def test_episodes_get_without_traces_says_so(fake_client):
    fake_client.get_episode = lambda episode_id: _episode_detail(
        episode_id=episode_id, traces={"trace_count": 0, "agent_names": []}
    )
    fake_client.list_episode_traces = lambda episode_id, **kwargs: TraceListPage(
        items=[], next_cursor=None
    )

    result = runner.invoke(main_app, ["traces", "episodes", "get", "ep_4c1d"])

    assert result.exit_code == 0, result.output
    assert "No traces were recorded for this episode." in result.output
    assert "Trace ID" not in result.output


def test_episodes_get_json_is_the_detail_alone(fake_client):
    result = runner.invoke(main_app, ["traces", "episodes", "get", "ep_4c1d", "-o", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["episode_id"] == "ep_4c1d"
    assert payload["traces"]["trace_count"] == 1
    assert "list_episode_traces" not in fake_client.calls


def test_episodes_get_raw_preserves_exact_bytes(fake_client):
    raw = b'{"id":"ep_4c1d","traces":["a"]}\n\xff'
    fake_client.get_episode_raw = lambda episode_id: raw

    result = runner.invoke(main_app, ["traces", "episodes", "get", "ep_4c1d", "--raw"])

    assert result.exit_code == 0
    assert result.stdout_bytes == raw
    assert "get_episode" not in fake_client.calls


def test_episodes_get_raw_to_dest_writes_the_file(fake_client, tmp_path):
    dest = tmp_path / "episode.json"

    result = runner.invoke(
        main_app,
        ["traces", "episodes", "get", "ep_4c1d", "--raw", "--dest", str(dest), "-o", "json"],
    )

    assert result.exit_code == 0, result.output
    assert dest.read_bytes() == b'{"id":"ep_4c1d","traces":["8d3f1a2b"]}'
    assert json.loads(result.output) == {"dest": str(dest), "bytes_written": 38}


def test_episodes_get_rejects_dest_without_raw(fake_client, tmp_path):
    dest = tmp_path / "episode.json"

    result = runner.invoke(main_app, ["traces", "episodes", "get", "ep_4c1d", "--dest", str(dest)])

    assert result.exit_code == 1
    assert "--dest requires --raw" in result.output
    assert "get_episode" not in fake_client.calls


class _TeamConfig:
    team_id = "team_123"
    team_name = "Research"
    team_id_from_env = False


def _episode_missing(episode_id, **kwargs):
    raise NotFoundError(
        f"No episode '{episode_id}' for this owner", status_code=404, code="episode_not_found"
    )


@pytest.mark.parametrize("args", [["--raw"], ["-o", "json"], []])
def test_episodes_get_not_found_names_the_account_it_searched(fake_client, monkeypatch, args):
    monkeypatch.setattr(traces_cmd, "Config", _TeamConfig)
    fake_client.get_episode = _episode_missing
    fake_client.get_episode_raw = _episode_missing

    result = runner.invoke(main_app, ["traces", "episodes", "get", "ep_gone", *args])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Not found: no episode ep_gone in team Research." in result.stderr
    assert "prime switch" in result.stderr


def test_episode_not_found_ignores_the_stored_name_under_a_team_id_override(
    fake_client, monkeypatch
):
    class _EnvTeamConfig(_TeamConfig):
        team_id = "team_from_env"
        team_id_from_env = True

    monkeypatch.setattr(traces_cmd, "Config", _EnvTeamConfig)
    fake_client.get_episode = _episode_missing

    result = runner.invoke(main_app, ["traces", "episodes", "get", "ep_gone"])

    assert result.exit_code == 1
    assert "no episode ep_gone in team team_from_env." in result.stderr
    assert "Research" not in result.stderr


def test_traces_list_episode_lists_member_traces_with_the_agent_column(fake_client):
    result = runner.invoke(
        main_app, ["traces", "list", "--episode", "ep_4c1d", "--has-error", "--limit", "5"]
    )

    assert result.exit_code == 0, result.output
    call = fake_client.calls["list_episode_traces"]
    assert call["episode_id"] == "ep_4c1d"
    assert call["has_error"] is True
    assert call["limit"] == 5
    assert call["cursor"] is None
    assert "sort" not in call
    assert "list" not in fake_client.calls
    assert "Traces · episode ep_4c1d" in result.output
    assert "Agent" in result.output
    assert "solver" in result.output


def test_traces_list_episode_rejects_sort(fake_client):
    result = runner.invoke(main_app, ["traces", "list", "--episode", "ep_4c1d", "--sort", "reward"])

    assert result.exit_code == 1
    assert "--sort cannot be combined with --episode" in result.output
    assert "list_episode_traces" not in fake_client.calls


def test_traces_list_episode_not_found_names_the_personal_account(fake_client, monkeypatch):
    class _PersonalConfig:
        team_id = None
        team_name = None
        team_id_from_env = False

    monkeypatch.setattr(traces_cmd, "Config", _PersonalConfig)
    fake_client.list_episode_traces = _episode_missing

    result = runner.invoke(main_app, ["traces", "list", "--episode", "ep_gone"])

    assert result.exit_code == 1
    assert "Not found: no episode ep_gone in your personal account." in result.stderr


def test_episode_tables_treat_values_as_literal_text(fake_client):
    markup = "[/]"
    fake_client.list_episodes = lambda **kwargs: EpisodeListPage(
        items=[_episode(episode_id=markup, environment_id=markup, outcome=markup)],
        next_cursor=None,
    )
    fake_client.get_episode = lambda episode_id: _episode_detail(
        episode_id=episode_id, run_id=markup, traces={"agent_names": [markup]}
    )

    listed = runner.invoke(main_app, ["traces", "episodes", "list"])
    fetched = runner.invoke(main_app, ["traces", "episodes", "get", "[red]ep"])

    assert listed.exit_code == 0, listed.output
    assert markup in listed.output
    assert fetched.exit_code == 0, fetched.output
    assert "[red]ep" in fetched.output
    assert markup in fetched.output
