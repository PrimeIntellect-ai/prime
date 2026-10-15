"""Tests for `prime train logs` (orchestrator + env-server) and components."""

import json
import re
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, List

import pytest
import typer
from prime_cli.api.rl import RLClient
from prime_cli.client import APIError
from prime_cli.commands import rl as rl_commands
from prime_cli.main import app
from rich.console import Console
from rich.text import Text
from typer.testing import CliRunner

RUN_ID = "rl-run-123"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _plain(output: str) -> str:
    """Rich styles each char of an option separately in terminal mode, so a
    raw '--tail' never appears contiguously under CI's color-forced runs."""
    return _ANSI_RE.sub("", output)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("prime_cli.commands.rl.time.sleep", lambda _: None)


def _run_payload(
    status: str = "RUNNING",
    loss: str = "rl",
    run_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "runConfig": run_config,
        "id": RUN_ID,
        "name": "demo",
        "userId": "u1",
        "status": status,
        "loss": loss,
        "rolloutsPerExample": 8,
        "seqLen": 4096,
        "maxSteps": 100,
        "batchSize": 32,
        "baseModel": "some-model",
        "environments": [{"name": "reverse-text"}],
        "createdAt": now,
        "updatedAt": now,
    }


def _make_mock_get(
    responses: Dict[str, Any],
    orch_calls: List[Dict[str, Any]],
    env_calls: List[Dict[str, Any]],
    list_calls: List[Dict[str, Any]],
):
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if endpoint == f"/rft/runs/{RUN_ID}":
            return {
                "run": _run_payload(
                    responses.get("run_status", "RUNNING"),
                    responses.get("run_loss", "rl"),
                    responses.get("run_config"),
                )
            }
        if endpoint == f"/rft/runs/{RUN_ID}/logs":
            call_params = params or {}
            orch_calls.append(call_params)
            if call_params.get("component") == "env-server":
                return {"logs": responses.get("env_server_logs", "")}
            return {"logs": responses.get("orchestrator_logs", "")}
        if endpoint == f"/rft/runs/{RUN_ID}/env-server-logs":
            env_calls.append(params or {})
            return {"logs": responses.get("env_server_logs", "")}
        if endpoint == f"/rft/runs/{RUN_ID}/env-servers":
            list_calls.append(params or {})
            return {"env_servers": responses.get("env_servers", [])}
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    return mock_get


# ---------- prime train logs (orchestrator default) ----------


def _json_log(**fields: Any) -> str:
    return json.dumps(
        {"timestamp": "2026-10-15T00:22:16.662071+00:00", "level": "INFO", "message": "Ready"}
        | fields
    )


@pytest.mark.parametrize(
    "prefix,context",
    [
        ("", ""),
        ("[adminlog] ", "[adminlog] "),
        ("[trainer0]:", "[trainer0] "),
        ("(APIServer pid=927) ", "(APIServer pid=927) "),
        (
            "[adminlog] [trainer7]:(Worker_TP0 pid=970) ",
            "[adminlog] [trainer7] (Worker_TP0 pid=970) ",
        ),
        ("\x1b[32m[adminlog] \x1b[0m", "[adminlog] "),
    ],
)
def test_json_log_formats_known_prefixes_and_preserves_context(prefix: str, context: str) -> None:
    # The internal URL mirrors Sami's screenshot: the API adds [adminlog]
    # for ADMIN/MANAGER readers, which used to bypass JSON detection.
    line = prefix + _json_log(message="Waiting for http://inference.svc.cluster.local:8000")
    rendered = rl_commands.clean_logs(line)
    assert len(rendered) == 1
    plain = Text.from_markup(rendered[0], emoji=False).plain
    assert plain == (
        f"00:22:16    INFO {context}Waiting for http://inference.svc.cluster.local:8000"
    )


def test_json_log_keeps_nested_message_worker_context_and_exception() -> None:
    line = _json_log(
        message={"event": "step", "values": [0.5, {"loss": 0.1}]},
        tag="T_3",
        extra={"rank": 3, "metrics": {"tokens": 128}},
        exception="Traceback:\nValueError: [red]literal[/red]",
    )
    plain = Text.from_markup(rl_commands.clean_logs(line)[0], emoji=False).plain
    assert '[T_3] {"event": "step", "values": [0.5, {"loss": 0.1}]}' in plain
    assert '"rank": 3, "metrics": {"tokens": 128}' in plain
    assert "Traceback:\n    ValueError: [red]literal[/red]" in plain


@pytest.mark.parametrize("prefix", ["", "[adminlog] ", "[trainer7]:(Worker_TP0 pid=970) "])
def test_json_log_keeps_foreign_top_level_context(prefix: str) -> None:
    # On the base branch, prefixed records fell back to raw JSON and kept
    # these fields. Recognizing the prefix must not make that context vanish.
    line = prefix + _json_log(
        rank=0,
        process=851,
        context={"pod": "synthetic-worker", "labels": ["[red]literal[/red]"]},
        module="clients",
        function="_check_health",
        line=347,
    )
    rendered = rl_commands.clean_logs(line)
    header, details = Text.from_markup(rendered[0], emoji=False).plain.split("\n", 1)
    assert header.startswith("00:22:16    INFO ")
    assert header.endswith("Ready")
    assert details.startswith("    ")
    assert json.loads(details) == {
        "rank": 0,
        "process": 851,
        "context": {"pod": "synthetic-worker", "labels": ["[red]literal[/red]"]},
        "module": "clients",
        "function": "_check_health",
        "line": 347,
    }


@pytest.mark.parametrize("level", ["debug", "INFO", "WARNING", "ERROR", "CRITICAL", "custom"])
def test_json_log_keeps_levels_and_literal_message(level: str) -> None:
    rendered = rl_commands.clean_logs(_json_log(level=level, message="[red]literal[/red] :rocket:"))
    assert Text.from_markup(rendered[0], emoji=False).plain == (
        f"00:22:16 {level.upper():>7} [red]literal[/red] :rocket:"
    )


def test_json_log_uses_native_level_alignment_before_distinct_sources() -> None:
    info = "[trainer0]:" + _json_log(tag="T_7")
    warning = "(Worker_TP1 pid=4002) " + _json_log(level="WARNING", tag="BATCH_7")
    rendered = [
        Text.from_markup(line, emoji=False).plain
        for line in rl_commands.clean_logs(info + "\n" + warning)
    ]
    assert rendered == [
        "00:22:16    INFO [trainer0] [T_7] Ready",
        "00:22:16 WARNING (Worker_TP1 pid=4002) [BATCH_7] Ready",
    ]
    assert rendered[0].index("[trainer0]") == rendered[1].index("(Worker_TP1")


def test_json_log_moves_complete_diagnostics_off_the_message_row() -> None:
    fields = {
        "module": "trainer",
        "function": "save",
        "line": 347,
        "type": "event",
        "extra": {"step": 20, "timing": {"seconds": 1.2}},
        "rank": 0,
        "context": {"pod": "synthetic-worker", "labels": ["[red]literal[/red]"]},
    }
    rendered = rl_commands.clean_logs(_json_log(message="Checkpoint saved", **fields))[0]
    text = Text.from_markup(rendered, emoji=False)
    header, details = text.plain.split("\n", 1)
    assert header == "00:22:16    INFO Checkpoint saved"
    assert details.startswith("    ")
    assert json.loads(details) == fields
    assert text.get_style_at_offset(rl_commands.console, text.plain.index('"module"')).dim


def test_json_log_preserves_empty_and_null_diagnostics() -> None:
    fields = {"tag": None, "extra": {}, "exception": False, "module": "", "rank": 0}
    rendered = rl_commands.clean_logs(_json_log(**fields))[0]
    _, details = Text.from_markup(rendered, emoji=False).plain.split("\n", 1)
    assert json.loads(details) == fields


@pytest.mark.parametrize("follow", [False, True])
def test_wrapped_diagnostics_stay_indented_without_losing_fields(
    monkeypatch: pytest.MonkeyPatch, follow: bool
) -> None:
    output = StringIO()
    monkeypatch.setattr(rl_commands, "console", Console(file=output, width=42, color_system=None))
    metadata = {"extra": {"rank": 0, "tensor": "x" * 100}, "module": "trainer"}
    responses = iter([_json_log(**metadata)])

    def fetch(tail: int) -> str:
        try:
            return next(responses)
        except StopIteration:
            raise KeyboardInterrupt

    if follow:
        with pytest.raises(KeyboardInterrupt):
            rl_commands._stream_logs(fetch, 1000, False, True, "trainer")
    else:
        rl_commands._stream_logs(fetch, 1000, False, False, "trainer")
    lines = output.getvalue().splitlines()
    header_index = lines.index("00:22:16    INFO Ready")
    details = lines[header_index + 1 :]
    assert len(details) > 1
    assert all(line.startswith("    ") for line in details)
    assert json.loads("".join(line[4:] for line in details)) == metadata


@pytest.mark.parametrize(
    "line",
    [
        '[adminlog] {"timestamp": "broken"',
        _json_log(timestamp=None),
        _json_log(timestamp=123),
        _json_log(level={"name": "INFO"}),
        '{"timestamp": "12:34:56", "level": "INFO", "record": {"message": "nested"}}',
        '{"payload": {"message": "nested"}}',
        "[trainer3]:plain [red]literal[/red] :rocket:",
        "[adminlog] ordinary text",
    ],
)
def test_unrecognized_or_malformed_log_is_literal_fallback(line: str) -> None:
    assert rl_commands.format_json_log_line(line) is None
    assert Text.from_markup(rl_commands.clean_logs(line)[0], emoji=False).plain == line


@pytest.fixture(params=["integer-digit-limit", "deep-nesting"])
def unsupported_json_log(request: pytest.FixtureRequest) -> str:
    # These are syntactically JSON but exceed Python's decoder limits.
    value = (
        "1" * 5000 if request.param == "integer-digit-limit" else "[" * 10000 + "0" + "]" * 10000
    )
    return '[adminlog] {"timestamp":"12:34:56","level":"INFO","message":' + value + "}"


def test_decoder_limit_is_literal_fallback(unsupported_json_log: str) -> None:
    assert rl_commands.format_json_log_line(unsupported_json_log) is None
    rendered = rl_commands.clean_logs(unsupported_json_log)
    assert Text.from_markup(rendered[0], emoji=False).plain == unsupported_json_log


@pytest.mark.parametrize("follow", [False, True])
def test_decoder_limit_does_not_abort_cli_output(
    monkeypatch: pytest.MonkeyPatch, unsupported_json_log: str, follow: bool
) -> None:
    calls = 0

    def get_logs(*args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return unsupported_json_log + "\n" + _json_log(message="Continued after decoder limit")
        raise KeyboardInterrupt

    monkeypatch.setattr(RLClient, "get_logs", get_logs)
    result = CliRunner().invoke(app, ["train", "logs", RUN_ID, *(["--follow"] if follow else [])])
    assert result.exit_code == 0, type(result.exception).__name__
    assert unsupported_json_log in _plain(result.output).replace("\n", "")
    assert "INFO Continued after decoder limit" in _plain(result.output)
    assert calls == (2 if follow else 1)
    if follow:
        assert "Stopped watching logs" in result.output


def test_progress_filter_still_applies_to_prefixed_json() -> None:
    logs = "\n".join(
        [
            "[adminlog] " + _json_log(type="progress", percent=50),
            "[trainer0]:" + _json_log(message="Done"),
            " 50%|█████     | 5/10",
            "100%|██████████| 10/10",
        ]
    )
    lines = [Text.from_markup(line, emoji=False).plain for line in rl_commands.clean_logs(logs)]
    assert lines == ["00:22:16    INFO [trainer0] Done", "100%|██████████| 10/10"]


@pytest.mark.parametrize(
    "args",
    [[], ["-c", "trainer"], ["-c", "inference"], ["--env", "reverse-text"]],
)
def test_logs_default_renders_api_admin_json_for_each_component(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    logs = "[adminlog] " + _json_log(message="Connecting", tag="T_0")
    orch: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"orchestrator_logs": logs, "env_server_logs": logs}, orch, [], [])
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)
    result = CliRunner().invoke(app, ["train", "logs", RUN_ID, *args])
    assert result.exit_code == 0, result.output
    assert "00:22:16    INFO [adminlog] [T_0] Connecting" in _plain(result.output)
    assert '"timestamp"' not in result.output
    assert len(orch) == 1


@pytest.mark.parametrize("command", ["train", "rl"])
@pytest.mark.parametrize("component", [None, "trainer"])
@pytest.mark.parametrize("follow", [False, True])
@pytest.mark.parametrize("raw", [False, True])
def test_sft_trainer_commands_preserve_mixed_log_streams(
    monkeypatch: pytest.MonkeyPatch, command: str, component: str | None, follow: bool, raw: bool
) -> None:
    ordinary = [
        "[trainer0]:Starting SFT [red]literal[/red] :rocket:",
        '[adminlog] {"timestamp": "broken", "message": "[green]literal[/green]"',
        '{"payload": {"message": "[blue]nested[/blue] :rocket:"}}',
        '{"timestamp":"12:34:56","level":"INFO","record":{"message":"nested"}}',
        '"[yellow]JSON string[/yellow] :rocket:"',
        '[1, "[bold]array[/bold]"]',
        "true",
        "null",
        "42",
    ]
    structured = "[trainer0]:" + _json_log(
        message="SFT checkpoint [red]literal[/red] :rocket:",
        module="train",
        function="train",
        line=82,
    )
    logs = "\n".join([*ordinary, structured, "SFT finished [green]literal[/green]"])
    calls: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"run_loss": "sft", "orchestrator_logs": logs}, calls, [], [])

    def get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if endpoint.endswith("/logs") and calls:
            raise KeyboardInterrupt
        return mock_get(self, endpoint, params)

    output = StringIO()
    monkeypatch.setattr("prime_cli.core.APIClient.get", get)
    monkeypatch.setattr(rl_commands, "console", Console(file=output, width=2000, color_system=None))
    args = [command, "logs", RUN_ID, *(["--component", component] if component else [])]
    result = CliRunner().invoke(
        app, args + (["--follow"] if follow else []) + (["--raw"] if raw else [])
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    # No explicit component lets the API select the primary pod (trainer
    # for SFT, orchestrator for RL); an explicit trainer is forwarded.
    assert calls[0].get("component") == component
    rendered = output.getvalue()
    if raw:
        assert logs + "\n" in rendered
    else:
        assert "\n".join(ordinary) + "\n" in rendered
        assert "00:22:16    INFO [trainer0] SFT checkpoint [red]literal[/red] :rocket:" in rendered
        assert '    {"module": "train", "function": "train", "line": 82}' in rendered
        assert structured not in rendered
        assert "SFT finished [green]literal[/green]\n" in rendered
    if follow:
        assert "Stopped watching logs" in rendered


@pytest.mark.parametrize("raw", [False, True])
def test_logs_keep_prefixed_foreign_context_with_api_response(
    monkeypatch: pytest.MonkeyPatch, raw: bool
) -> None:
    line = "[adminlog] " + _json_log(rank=0, process=851, context={"pod": "synthetic-worker"})
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get", _make_mock_get({"orchestrator_logs": line}, [], [], [])
    )
    result = CliRunner().invoke(app, ["train", "logs", RUN_ID, *(["--raw"] if raw else [])])
    assert result.exit_code == 0, result.output
    if raw:
        assert result.stdout == line + "\n"
    else:
        plain = _plain(result.output)
        assert "00:22:16    INFO [adminlog] Ready" in plain
        for field in ['"rank": 0', '"process": 851', '"synthetic-worker"']:
            assert field in plain


def test_logs_raw_preserves_json_without_markup_emoji_or_wrapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    line = _json_log(message="[red]literal[/red] :rocket: " + "long " * 50)
    mock_get = _make_mock_get({"orchestrator_logs": line}, [], [], [])
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)
    result = CliRunner().invoke(app, ["train", "logs", RUN_ID, "--raw"])
    assert result.exit_code == 0, result.output
    assert result.stdout == line + "\n"
    assert json.loads(result.stdout) == json.loads(line)


def test_logs_raw_keeps_admin_prefix_and_malformed_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    logs = "[adminlog] " + _json_log() + '\n[trainer0]:plain\n{"malformed"\n'
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get", _make_mock_get({"orchestrator_logs": logs}, [], [], [])
    )
    result = CliRunner().invoke(app, ["train", "logs", RUN_ID, "-r"])
    assert result.exit_code == 0, result.output
    assert result.stdout == logs


@pytest.mark.parametrize("raw", [False, True])
def test_follow_compares_original_records_before_formatting(
    monkeypatch: pytest.MonkeyPatch, raw: bool
) -> None:
    first = "[adminlog] " + _json_log(timestamp="2026-10-15T00:22:16.100000+00:00")
    second = "[adminlog] " + _json_log(timestamp="2026-10-15T00:22:16.200000+00:00")
    progress = _json_log(type="progress")
    responses = iter([first + "\n" + progress, progress + "\n" + second])

    def get_logs(*args: Any, **kwargs: Any) -> str:
        try:
            return next(responses)
        except StopIteration:
            raise KeyboardInterrupt

    monkeypatch.setattr(RLClient, "get_logs", get_logs)
    result = CliRunner().invoke(app, ["train", "logs", RUN_ID, "-f", *(["--raw"] if raw else [])])
    assert result.exit_code == 0, result.output
    assert result.output.count('"message": "Ready"' if raw else "Ready") == (3 if raw else 2)
    assert "Stopped watching logs" in result.output


def test_follow_surfaces_api_errors_after_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def get_logs(*args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "[adminlog] " + _json_log()
        raise APIError("HTTP 403: denied")

    monkeypatch.setattr(RLClient, "get_logs", get_logs)
    result = CliRunner().invoke(app, ["train", "logs", RUN_ID, "--follow"])
    assert result.exit_code == 1
    assert "INFO [adminlog] Ready" in _plain(result.output)
    assert "HTTP 403: denied" in result.output


def test_follow_recovers_startup_and_rate_limit_without_reprinting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = "[adminlog] " + _json_log(message="First")
    second = "[trainer0]:" + _json_log(message="Second")
    responses = iter(
        [
            APIError("HTTP 404: run is queued"),
            first,
            APIError("HTTP 429: rate limited"),
            APIError("HTTP 429: rate limited"),
            APIError("HTTP 429: rate limited"),
            first + "\n" + second,
        ]
    )
    sleeps: list[int] = []

    def get_logs(*args: Any, **kwargs: Any) -> str:
        try:
            response = next(responses)
        except StopIteration:
            raise KeyboardInterrupt
        if isinstance(response, APIError):
            raise response
        return response

    monkeypatch.setattr(RLClient, "get_logs", get_logs)
    monkeypatch.setattr(rl_commands.time, "sleep", sleeps.append)
    result = CliRunner().invoke(app, ["train", "logs", RUN_ID, "-f"])
    assert result.exit_code == 0, result.output
    assert result.output.count("First") == 1
    assert result.output.count("Second") == 1
    assert "waiting for logs" in result.output
    assert "Rate limited. Waiting 30s" in result.output
    assert "Stopped watching logs" in result.output
    assert sleeps == [10, 5, 10, 10, 30, 5]


def test_logs_default_hits_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"orchestrator_logs": "orch-line-1\norch-line-2"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--raw"])

    assert result.exit_code == 0, result.output
    assert "orch-line-1" in _plain(result.output)
    assert "orch-line-2" in _plain(result.output)
    assert len(orch) == 1
    assert orch[0]["tail_lines"] == 1000
    assert env == []


def test_logs_env_flag_alone_infers_env_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing --env without -c should infer component=env-server."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"env_server_logs": "inferred line"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--env", "reverse-text", "--raw"])

    assert result.exit_code == 0, result.output
    assert "inferred line" in result.output
    assert env == []
    assert len(orch) == 1
    assert orch[0] == {
        "tail_lines": 1000,
        "component": "env-server",
        "env_name": "reverse-text",
    }


def test_logs_explicit_orchestrator_with_env_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit `-c orchestrator --env x` is a real conflict; must error."""

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise AssertionError(f"Should not be called: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app, ["rl", "logs", RUN_ID, "-c", "orchestrator", "--env", "reverse-text"]
    )
    assert result.exit_code != 0


# ---------- prime train logs -c env-server ----------


def test_logs_env_server_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {"env_server_logs": "ModuleNotFoundError: no module foo"}, orch, env, lst
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app,
        [
            "rl",
            "logs",
            RUN_ID,
            "-c",
            "env-server",
            "--env",
            "reverse-text",
            "--tail",
            "50",
            "--raw",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "ModuleNotFoundError" in result.output
    assert env == []
    assert len(orch) == 1
    assert orch[0] == {
        "tail_lines": 50,
        "component": "env-server",
        "env_name": "reverse-text",
    }


def test_logs_env_server_qualified_name_parses_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--env reverse-text/2` should split into env_name='reverse-text', env_index=2."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"env_server_logs": "line"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app,
        ["rl", "logs", RUN_ID, "-c", "env-server", "--env", "reverse-text/2", "--raw"],
    )

    assert result.exit_code == 0, result.output
    assert env[0]["env_name"] == "reverse-text"
    assert env[0]["env_index"] == 2


def test_logs_env_server_qualified_name_forwards_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy qualified env-server logs must preserve backend filters."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"env_server_logs": "line"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app,
        [
            "rl",
            "logs",
            RUN_ID,
            "-c",
            "env-server",
            "--env",
            "reverse-text/0",
            "--search",
            "error",
            "--regex",
            "--level",
            "error",
            "--since",
            "1h",
            "--raw",
        ],
    )

    assert result.exit_code == 0, result.output
    assert orch == []
    assert env[0] == {
        "env_name": "reverse-text",
        "env_index": 0,
        "tail_lines": 1000,
        "search": "error",
        "regex": True,
        "level": "ERROR",
        "since_seconds": 3600,
    }


def test_logs_env_server_owner_name_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--env primeintellect/reverse-text` is a valid name, not a qualifier.

    Only a trailing ``/<int>`` should be treated as a replica index.
    """
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"env_server_logs": "ok"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app,
        ["rl", "logs", RUN_ID, "--env", "primeintellect/reverse-text", "--raw"],
    )

    assert result.exit_code == 0, result.output
    assert env == []
    assert orch[0] == {
        "tail_lines": 1000,
        "component": "env-server",
        "env_name": "primeintellect/reverse-text",
    }


def test_logs_env_server_owner_name_with_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--env primeintellect/reverse-text/1` should split at the last slash."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"env_server_logs": "ok"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app,
        ["rl", "logs", RUN_ID, "--env", "primeintellect/reverse-text/1", "--raw"],
    )

    assert result.exit_code == 0, result.output
    assert env[0]["env_name"] == "primeintellect/reverse-text"
    assert env[0]["env_index"] == 1


def test_logs_env_server_requires_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise AssertionError(f"Should not be called: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "-c", "env-server"])
    assert result.exit_code != 0
    assert "--env" in result.output or "env" in result.output.lower()


def test_logs_invalid_component(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise AssertionError(f"Should not be called: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "-c", "bogus"])
    assert result.exit_code != 0


def test_logs_run_not_started(monkeypatch: pytest.MonkeyPatch) -> None:
    """Queued runs poll for the startup window before exiting with a friendly message."""
    assert (
        rl_commands.HOSTED_TRAINING_LOG_STARTUP_POLLS
        * rl_commands.HOSTED_TRAINING_LOG_STARTUP_POLL_SECONDS
        >= 180
    )
    calls = {"n": 0}

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        calls["n"] += 1
        raise APIError("Failed to get Hosted Training run logs: 404 run is queued")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "logs", RUN_ID, "--raw"])

    assert result.exit_code == 0, result.output
    assert calls["n"] == rl_commands.HOSTED_TRAINING_LOG_STARTUP_POLLS + 1
    assert "Hosted Training run is starting" in result.output
    assert "Hosted Training run has not started yet" in result.output
    assert "RL" not in result.output


def test_logs_startup_marker_without_404_exits_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        calls["n"] += 1
        raise APIError("Failed to get Hosted Training run logs: HTTP 403: run is pending")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "logs", RUN_ID, "--raw"])

    assert result.exit_code == 1
    assert calls["n"] == 1
    assert "Error:" in result.output
    assert "Hosted Training run has not started yet" not in result.output


def test_logs_queued_run_succeeds_when_logs_appear(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise APIError("Failed to get Hosted Training run logs: 404 run is queued")
        return {"logs": "orch-ready"}

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "logs", RUN_ID, "--raw"])

    assert result.exit_code == 0, result.output
    assert calls["n"] == 3
    assert "orch-ready" in result.output
    assert "Failed" not in result.output


def test_logs_env_server_follow_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Follow-mode polls and avoids reprinting overlapping tail lines."""
    call_count = {"n": 0}

    def mock_get_env_server_logs(
        self: Any,
        run_id: str,
        env_name: str,
        env_index: int = 0,
        tail_lines: int = 1000,
        **_kwargs: Any,
    ) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "line-1\nline-2"
        if call_count["n"] == 2:
            return "line-2\nline-3"
        raise KeyboardInterrupt

    monkeypatch.setattr(RLClient, "get_env_server_logs", mock_get_env_server_logs)

    result = CliRunner().invoke(
        app,
        [
            "rl",
            "logs",
            RUN_ID,
            "-c",
            "env-server",
            "--env",
            "reverse-text/0",
            "--follow",
            "--raw",
        ],
    )

    plain = _plain(result.output)
    assert result.exit_code == 0, result.output
    assert plain.count("line-1") == 1
    assert plain.count("line-2") == 1
    assert plain.count("line-3") == 1
    assert call_count["n"] >= 2


# ---------- prime train components ----------


def test_components_lists_orchestrator_and_env_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {
            "run_status": "RUNNING",
            "env_servers": [
                {
                    "env_name": "reverse-text",
                    "env_index": 0,
                    "pod_name": "rft-rl-run-123-env-reverse-text-0",
                    "status": "Running",
                },
                {
                    "env_name": "opencode-math",
                    "env_index": 0,
                    "pod_name": "rft-rl-run-123-env-opencode-math-0",
                    "status": "CrashLoopBackOff",
                },
            ],
        },
        orch,
        env,
        lst,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "components", RUN_ID])

    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "orchestrator" in out
    assert "RUNNING" in out
    assert "reverse-text" in out
    assert "opencode-math" in out
    assert "CrashLoopBackOff" in out
    # No qualifier when env names are unique.
    assert "reverse-text/" not in out
    assert "opencode-math/" not in out
    assert len(lst) == 1


def test_components_qualifies_duplicate_env_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the same env name appears twice, the listing must show 'name/N'."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {
            "run_status": "RUNNING",
            "env_servers": [
                {
                    "env_name": "reverse-text",
                    "env_index": 0,
                    "pod_name": "rft-x-env-reverse-text-0",
                    "status": "Running",
                },
                {
                    "env_name": "reverse-text",
                    "env_index": 1,
                    "pod_name": "rft-x-env-reverse-text-1",
                    "status": "Running",
                },
            ],
        },
        orch,
        env,
        lst,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "components", RUN_ID])

    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "reverse-text/0" in out
    assert "reverse-text/1" in out


def test_components_empty_env_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {"run_status": "QUEUED", "env_servers": []},
        orch,
        env,
        lst,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "components", RUN_ID])

    assert result.exit_code == 0, result.output
    assert "orchestrator" in result.output
    assert "QUEUED" in result.output


def test_components_lists_trainer_for_sft(monkeypatch: pytest.MonkeyPatch) -> None:
    """SFT runs are trainer-only: no orchestrator pod, so show the trainer."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {"run_status": "RUNNING", "run_loss": "sft", "env_servers": []},
        orch,
        env,
        lst,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "components", RUN_ID])

    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "trainer" in out
    assert "orchestrator" not in out


def test_components_rl_run_lists_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    """An RL (loss != 'sft') run keeps the orchestrator row."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {"run_status": "RUNNING", "run_loss": "rl", "env_servers": []},
        orch,
        env,
        lst,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "components", RUN_ID])

    assert result.exit_code == 0, result.output
    assert "orchestrator" in result.output


# ---------- --all / --tail / --since volume flags (ENG-6367) ----------


def test_logs_all_requests_api_maximum(monkeypatch: pytest.MonkeyPatch) -> None:
    """--all asks for the API maxima: 5000 lines over the last 24h."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"orchestrator_logs": "orch-line-1"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--all", "--raw"])

    assert result.exit_code == 0, result.output
    assert orch[0]["tail_lines"] == 5000
    assert orch[0]["since_seconds"] == 86400


def test_logs_all_with_search_keeps_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """--all composes with --search instead of overriding it."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"orchestrator_logs": "line"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app, ["rl", "logs", RUN_ID, "--all", "--search", "backpressure", "--raw"]
    )

    assert result.exit_code == 0, result.output
    assert orch[0] == {
        "tail_lines": 5000,
        "since_seconds": 86400,
        "search": "backpressure",
    }


def test_logs_all_with_tail_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise AssertionError(f"Should not be called: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--all", "--tail", "3000"])

    assert result.exit_code != 0
    assert "--tail" in _plain(result.output)


def test_logs_all_with_since_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise AssertionError(f"Should not be called: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--all", "--since", "2h"])

    assert result.exit_code != 0
    assert "--since" in _plain(result.output)


def test_logs_tail_above_api_cap_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise AssertionError(f"Should not be called: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--tail", "5001"])

    assert result.exit_code != 0
    assert "5000" in result.output


def test_logs_tail_zero_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise AssertionError(f"Should not be called: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "-n", "0"])

    assert result.exit_code != 0


def test_logs_large_tail_with_search_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--tail above 1000 must reach the API together with the search filter."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"orchestrator_logs": "line"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(
        app, ["rl", "logs", RUN_ID, "--tail", "3000", "--search", "error", "--raw"]
    )

    assert result.exit_code == 0, result.output
    assert orch[0] == {"tail_lines": 3000, "search": "error"}


def test_logs_since_passthrough_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    """--since alone widens the window without a search/level filter."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get({"orchestrator_logs": "line"}, orch, env, lst)
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "logs", RUN_ID, "--since", "90m", "--raw"])

    assert result.exit_code == 0, result.output
    assert orch[0] == {"tail_lines": 1000, "since_seconds": 5400}


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        ("15m", 900),
        ("1h", 3600),
        ("2h", 7200),
        ("90m", 5400),
        ("1d", 86_400),
        ("24h", 86_400),
        ("86400", 86_400),
        ("60", 60),
    ],
)
def test_parse_since_accepts_human_durations(value: str | None, expected: int | None) -> None:
    assert rl_commands._parse_since(value) == expected


@pytest.mark.parametrize(
    "value",
    ["abc", "1x", "30", "86401", "59s", "25h", "-5m"],
)
def test_parse_since_rejects_out_of_window_values(value: str) -> None:
    with pytest.raises(typer.BadParameter):
        rl_commands._parse_since(value)


def test_components_lists_eval_pod_and_inference_for_sft_online_evals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SFT WITH online evals (ENG-6565) is no longer trainer-only: the chart
    renders an eval pod under the `orchestrator` role plus an inference pool.
    Hiding them leaves `-c orchestrator` / `-c inference` undiscoverable even
    though both have logs."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {
            "run_status": "RUNNING",
            "run_loss": "sft",
            "run_config": {"eval": {"interval": 5}},
            "env_servers": [
                {
                    "env_name": "eval-reverse-text",
                    "env_index": 0,
                    "pod_name": "fft-sft-eval-smoke-env-0-0",
                    "status": "Running",
                }
            ],
        },
        orch,
        env,
        lst,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "components", RUN_ID])

    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    for expected in ("trainer", "orchestrator", "inference", "eval-reverse-text"):
        assert expected in out, f"{expected!r} missing from: {out}"


def test_components_trainer_only_sft_still_hides_the_rollout_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runConfig without [eval] keeps the pre-ENG-6565 behaviour."""
    orch: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    lst: List[Dict[str, Any]] = []
    mock_get = _make_mock_get(
        {
            "run_status": "RUNNING",
            "run_loss": "sft",
            "run_config": {"max_steps": 10},
            "env_servers": [],
        },
        orch,
        env,
        lst,
    )
    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "components", RUN_ID])

    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "trainer" in out
    assert "orchestrator" not in out
    assert "inference" not in out
