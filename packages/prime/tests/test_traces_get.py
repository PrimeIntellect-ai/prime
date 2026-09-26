"""`prime traces get` summary layout."""

import json

import pytest
from prime_cli.commands import traces as traces_cmd
from prime_cli.main import app as main_app
from prime_traces import TraceSummary
from typer.testing import CliRunner

runner = CliRunner()

TOOLS = [
    {
        "type": "function",
        "name": "Bash",
        "description": "Executes a bash command.\n\nLong usage notes follow.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
    {
        "type": "function",
        "name": "Read",
        "description": "Reads a file.",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
    },
]


def _summary(**overrides):
    fields = {
        "trace_id": "8d3f1a2b",
        "upload_id": "5ee85e41",
        "episode_id": None,
        "created_at": "2026-07-20T18:02:11.482Z",
        "ingested_at": "2026-07-20T18:06:02.117Z",
        "run_id": "run_9f3k2m",
        "environment_id": "terminal-bench-2",
        "model": {"provider": "prime", "id": "deepseek-v4-flash"},
        "task_id": "tb2-0187",
        "agent_name": "solver",
        "score": {"reward": 0.85, "outcome": "done"},
        "execution": {"has_error": False, "is_truncated": False},
        "duration_ms": 258000,
        "total_tokens": 84213,
        "size_bytes": 417284,
        "context": {"source": "hosted_eval", "run_type": "eval"},
        "user_id": "user_1",
        "run_step": None,
        "task_key": "k" * 64,
        "task_hash": "h" * 64,
        "metrics": {"graded": 0.0, "smell_pass": 1.0},
        "activity": {"total_entries": 5, "user_messages": 1, "model_turns": 2, "tool_calls": 2},
        "tool_definitions": TOOLS,
    }
    fields.update(overrides)
    return TraceSummary.model_validate(fields)


class FakeClient:
    def get(self, trace_id):
        return _summary(trace_id=trace_id)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setattr(traces_cmd.console, "width", 120)
    fake = FakeClient()
    monkeypatch.setattr(traces_cmd, "_traces_client", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# prime traces get
# ---------------------------------------------------------------------------


def test_get_shows_fixed_fields_without_dumping_tool_schemas(client):
    result = runner.invoke(main_app, ["traces", "get", "8d3f1a2b"])

    assert result.exit_code == 0, result.output
    out = result.output
    for expected in (
        "run_9f3k2m  ·  eval",
        "prime/deepseek-v4-flash",
        "done",
        "reward 0.85",
        "4m 18s",
        "84,213",
        "407.5 KiB",
        "2 model turns · 2 tool calls · 1 user messages · 5 entries",
        "2  Bash, Read",
        "source=hosted_eval",
        "smell_pass",
        "ingested 2026-07-20 18:06:02Z",
        "prime traces get 8d3f1a2b --raw",
    ):
        assert expected in out, expected
    # The schemas, dict reprs and internal keys stay out of the table.
    assert "Executes a bash command" not in out
    assert "{'provider'" not in out
    assert "task_hash" not in out and "k" * 64 not in out


def test_get_json_still_returns_every_field(client):
    result = runner.invoke(main_app, ["traces", "get", "8d3f1a2b", "-o", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tool_definitions"][0]["name"] == "Bash"
    assert payload["task_hash"] == "h" * 64


def test_get_summary_without_extra_fields(client):
    bare = _summary()
    fields = bare.model_dump(mode="json")
    for key in ("metrics", "activity", "tool_definitions", "user_id", "run_step"):
        fields.pop(key)
    client.get = lambda trace_id: TraceSummary.model_validate(fields)

    result = runner.invoke(main_app, ["traces", "get", "8d3f1a2b"])

    assert result.exit_code == 0, result.output
    assert "metrics" not in result.output
    assert "tools" not in result.output


@pytest.mark.parametrize("args", [["transcript", "8d3f1a2b"], ["episodes", "list"]])
def test_removed_commands_are_not_registered(client, args):
    result = runner.invoke(main_app, ["traces", *args])

    assert result.exit_code == 2
    assert "No such command" in result.output
