"""`prime traces get` summary layout and `prime traces transcript`.

The transcript reads the node/call index and falls back to the raw document
when the index is not ready or is capped, so both sources are exercised here.
"""

import json

import pytest
from prime_cli.commands import traces as traces_cmd
from prime_cli.commands import traces_transcript as transcript_module
from prime_cli.commands.traces_transcript import parse_node_range
from prime_cli.main import app as main_app
from prime_traces import (
    APIError,
    LineFormatConflictError,
    TraceCallPage,
    TraceNodePage,
    TraceNotIndexedError,
    TraceSummary,
)
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


def _msg(role, content=None, **extra):
    message = {
        "role": role,
        "content": content,
        "reasoning_content": None,
        "tool_calls": None,
        "tool_call_id": None,
        "name": None,
        "tool_name": None,
    }
    message.update(extra)
    return message


LONG_OUTPUT = "\n".join(f"line {i}" for i in range(1, 11))

MESSAGES = [
    _msg("system", "You are a careful agent."),
    _msg("user", "List the workspace."),
    _msg(
        "assistant",
        reasoning_content="Start by listing files.",
        tool_calls=[
            {"id": "c1", "type": "function", "name": "Bash", "arguments": '{"command": "ls -la"}'}
        ],
    ),
    _msg("tool", LONG_OUTPUT, tool_call_id="c1"),
    _msg(
        "assistant",
        tool_calls=[
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "Read", "arguments": {"file_path": "/workspace/a.sv"}},
            }
        ],
    ),
    _msg("tool", "Error: file not found", tool_call_id="c2"),
    _msg("assistant", [{"type": "text", "text": "The workspace has one file."}]),
]


def _nodes(start=0, stop=None):
    return [
        {
            "node_idx": i,
            "parent_idx": i - 1 if i else None,
            "timestamp": 1000.0 + i * 2,
            "sampled": m["role"] == "assistant",
            "message": m,
        }
        for i, m in enumerate(MESSAGES)
    ][start:stop]


INDEX_CALLS = [
    {
        "call_idx": 0,
        "node_idx": 2,
        "time_start": 1000.0,
        "time_end": 1003.5,
        "model": "m",
        "endpoint": "/v1/messages",
        "finish_reason": "tool_calls",
    },
    {
        "call_idx": 1,
        "node_idx": 4,
        "time_start": 1005.0,
        "time_end": 1006.0,
        "model": "m",
        "endpoint": "/v1/messages",
        "finish_reason": "tool_calls",
    },
    {
        "call_idx": 2,
        "node_idx": 6,
        "time_start": 1008.0,
        "time_end": 1015.0,
        "model": "m",
        "endpoint": "/v1/messages",
        "finish_reason": "stop",
    },
]

DOCUMENT = {
    "version": 1,
    "id": "8d3f1a2b",
    "tools": TOOLS,
    "nodes": [
        {
            "tools": TOOLS if i == 0 else [],
            "semantic_parents": [i - 1] if i else [],
            "message": m,
            "sampled": m["role"] == "assistant",
            "timestamp": 1000.0 + i * 2,
        }
        for i, m in enumerate(MESSAGES)
    ],
    "calls": [
        {
            "node": c["node_idx"],
            "model": "m",
            "endpoint": "/v1/messages",
            "finish_reason": c["finish_reason"],
            "usage": {
                "prompt_tokens": 2000 + c["call_idx"],
                "completion_tokens": 100 + c["call_idx"],
                "cached_input_tokens": 16384,
                "reasoning_tokens": 7,
            },
            "time": {"start": c["time_start"], "end": c["time_end"]},
        }
        for c in INDEX_CALLS
    ],
}


class FakeClient:
    """Serves MESSAGES from the index in two pages, or from DOCUMENT."""

    def __init__(self, *, not_indexed=False, partial_index=False):
        self.not_indexed = not_indexed
        self.partial_index = partial_index
        self.requests = []

    def get(self, trace_id):
        self.requests.append(("get", trace_id))
        return _summary(trace_id=trace_id)

    def list_nodes(self, trace_id, *, limit=None, cursor=None, **_):
        self.requests.append(("nodes", cursor))
        if self.not_indexed:
            raise TraceNotIndexedError("not indexed yet", status_code=409, code="trace_not_indexed")
        if cursor is None:
            return TraceNodePage(items=_nodes(0, 4), next_cursor="n1", partial_index=False)
        return TraceNodePage(items=_nodes(4), next_cursor=None, partial_index=self.partial_index)

    def list_calls(self, trace_id, *, limit=None, cursor=None, **_):
        self.requests.append(("calls", cursor))
        return TraceCallPage(items=INDEX_CALLS, next_cursor=None, partial_index=False)

    def get_raw(self, trace_id):
        self.requests.append(("raw", trace_id))
        return json.dumps(DOCUMENT).encode()


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
        "prime traces transcript 8d3f1a2b",
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


# ---------------------------------------------------------------------------
# prime traces transcript
# ---------------------------------------------------------------------------


def test_transcript_reads_every_index_page(client):
    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b"])

    assert result.exit_code == 0, result.output
    assert ("nodes", None) in client.requests and ("nodes", "n1") in client.requests
    assert not any(kind == "raw" for kind, _ in client.requests)
    out = result.output
    assert "system prompt · 24 chars · hidden" in out
    assert "You are a careful agent." not in out
    assert "List the workspace." in out
    assert "● turn 1" in out and "● turn 3" in out
    assert "3.5s" in out and "tool_calls" in out
    assert "Start by listing files." in out
    assert "→ Bash  ls -la" in out
    assert "→ Read  /workspace/a.sv" in out
    assert "line 4" in out and "line 5" not in out
    assert "… 6 more lines (--full, or --node 3)" in out
    assert "Error: file not found" in out
    assert "The workspace has one file." in out
    assert "Read from the full document" not in out


def test_transcript_full_includes_system_prompt(client):
    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b", "--full"])

    assert result.exit_code == 0, result.output
    assert "You are a careful agent." in result.output
    assert "line 10" in result.output
    assert "more lines" not in result.output


def test_transcript_falls_back_to_document_when_not_indexed(monkeypatch, client):
    client.not_indexed = True

    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b"])

    assert result.exit_code == 0, result.output
    assert ("raw", "8d3f1a2b") in client.requests
    assert "Read from the full document: the trace is still being indexed." in result.output
    assert "● turn 3" in result.output and "3.5s" in result.output


def test_transcript_falls_back_to_document_when_index_is_capped(client):
    client.partial_index = True

    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b", "-o", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["source"] == "document"
    assert len(payload["nodes"]) == len(MESSAGES)
    assert payload["nodes"][3]["parent_idx"] == 2
    # Both sources produce the same call shape, so JSON consumers see one schema.
    assert set(payload["calls"][0]) == set(INDEX_CALLS[0])
    assert not any(kind == "calls" for kind, _ in client.requests)


def test_transcript_tools_only(client):
    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b", "--tools-only"])

    assert result.exit_code == 0, result.output
    assert "ls -la" in result.output and "10 lines" in result.output
    assert "/workspace/a.sv" in result.output and "1 line" in result.output
    # A result that looks like an error is only colored, never labelled.
    assert " error" not in result.output
    assert "Start by listing files." not in result.output


@pytest.mark.parametrize(
    ("args", "present", "absent"),
    [
        (["--node", "3"], ["line 1", "line 10"], ["List the workspace.", "turn 1", "more lines"]),
        (["--node", "4:"], ["turn 2", "turn 3"], ["turn 1 "]),
        (["--node", "0"], ["You are a careful agent."], ["List the workspace."]),
    ],
)
def test_transcript_selection(client, args, present, absent):
    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b", *args])

    assert result.exit_code == 0, result.output
    body = result.output.split("╰", 1)[1]  # after the header panel
    for text in present:
        assert text in body, text
    for text in absent:
        assert text not in body, text


def test_transcript_json_respects_node_range(client):
    result = runner.invoke(
        main_app, ["traces", "transcript", "8d3f1a2b", "--node", "4:", "-o", "json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["source"] == "index"
    assert [n["node_idx"] for n in payload["nodes"]] == [4, 5, 6]
    assert len(payload["calls"]) == 3


def test_transcript_treats_content_as_literal_text(client):
    MESSAGES_WITH_MARKUP = [_msg("user", "[red]not markup[/red] [/]")]
    client.list_nodes = lambda trace_id, **_: TraceNodePage(
        items=[
            {
                "node_idx": 0,
                "parent_idx": None,
                "timestamp": None,
                "sampled": False,
                "message": MESSAGES_WITH_MARKUP[0],
            }
        ],
        next_cursor=None,
        partial_index=False,
    )

    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b"])

    assert result.exit_code == 0, result.output
    assert "[red]not markup[/red] [/]" in result.output


def test_transcript_plain_mode_has_no_panels(client):
    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b", "--plain"])

    assert result.exit_code == 0, result.output
    assert "╭" not in result.output and "│" not in result.output
    assert "List the workspace." in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["--node", "x"],
        ["--node", "9:3"],
    ],
)
def test_transcript_rejects_invalid_arguments_before_network(client, args):
    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b", *args])

    assert result.exit_code == 1
    assert client.requests == []


def test_transcript_rejects_document_without_nodes(client):
    client.not_indexed = True
    client.get_raw = lambda trace_id: b'{"version": 1}'

    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b"])

    assert result.exit_code == 1
    assert "no `nodes` array" in result.output


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("12", (12, 12)),
        ("30:", (30, None)),
        (":9", (None, 9)),
        ("5:9", (5, 9)),
        (":", (None, None)),
    ],
)
def test_parse_node_range(value, expected):
    assert parse_node_range(value) == expected


def test_transcript_falls_back_on_not_indexed_code_from_older_sdk(client):
    # prime-traces 0.0.5 raised every 409 as LineFormatConflictError; the code decides.
    def not_indexed(trace_id, **_):
        raise LineFormatConflictError("not indexed", status_code=409, code="trace_not_indexed")

    client.list_nodes = not_indexed

    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b"])

    assert result.exit_code == 0, result.output
    assert ("raw", "8d3f1a2b") in client.requests


def test_transcript_other_index_errors_do_not_fall_back(client):
    def conflict(trace_id, **_):
        raise APIError("invalid cursor", status_code=400, code="invalid_cursor")

    client.list_nodes = conflict

    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b"])

    assert result.exit_code == 1
    assert "invalid cursor" in result.output
    assert not any(kind == "raw" for kind, _ in client.requests)


def test_transcript_asks_for_upgrade_on_sdk_without_node_reads(monkeypatch, client):
    class OldClient:
        def get(self, trace_id):
            raise AssertionError("should stop before any request")

    monkeypatch.setattr(traces_cmd, "_traces_client", OldClient)

    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b"])

    assert result.exit_code == 1
    assert "prime-traces 0.0.6 or newer" in result.output


def test_traces_command_module_imports_only_names_from_prime_traces_0_0_5():
    # The CLI's dependency floor is prime-traces>=0.0.5 until the release PR raises it,
    # and `prime_cli.main` imports this module, so a newer-only import breaks every command.
    import ast
    import inspect

    added_in_0_0_6 = {
        "NodeMessage",
        "TraceCall",
        "TraceCallPage",
        "TraceNode",
        "TraceNodePage",
        "TraceNotIndexedError",
    }
    for module in (traces_cmd, transcript_module):
        tree = ast.parse(inspect.getsource(module))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "prime_traces"
            for alias in node.names
        }
        assert not imported & added_in_0_0_6, module.__name__


def test_transcript_empty_node_range(client):
    result = runner.invoke(main_app, ["traces", "transcript", "8d3f1a2b", "--node", "50:"])

    assert result.exit_code == 0, result.output
    assert "No nodes in that range." in result.output


def test_get_lists_tool_names_and_counts_the_rest(client):
    many = [{**TOOLS[0], "name": f"tool_number_{i:02d}"} for i in range(12)]
    client.get = lambda trace_id: _summary(tool_definitions=many)

    result = runner.invoke(main_app, ["traces", "get", "8d3f1a2b"])

    assert result.exit_code == 0, result.output
    assert "12  tool_number_00, tool_number_01" in result.output
    assert "more" in result.output and "tool_number_11" not in result.output
