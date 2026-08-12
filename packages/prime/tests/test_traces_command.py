"""The traces commands must build their client from the CLI config, so that
`prime --context <env> traces ...` talks to that context's deployment instead
of silently falling back to the SDK's static config."""

import json

import pytest
from prime_cli.commands import traces as traces_cmd
from prime_cli.core import Config
from prime_cli.core.config import ConfigModel
from prime_cli.main import app as main_app
from prime_traces import Batch, TraceListPage, TraceSummary, UploadReceipt
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
    assert "cursor-1" in result.output  # next-page hint
    call = fake_client.calls["list"]
    assert call["run_id"] == "run_9f3k2m"
    assert call["reward_min"] == 0.5
    assert call["limit"] == 10


def test_list_command_json_output_is_parseable(fake_client):
    result = runner.invoke(main_app, ["traces", "list", "-o", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["items"][0]["trace_id"] == "8d3f1a2b"
    assert payload["next_cursor"] == "cursor-1"


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
