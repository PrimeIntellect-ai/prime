"""`prime inference trace-capture` against a fake of the platform's HTTP contract."""

import json

import httpx
import pytest
from prime_cli.client import APIClient
from prime_cli.commands import trace_capture
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()
ENV = {"NO_COLOR": "1", "PRIME_DISABLE_VERSION_CHECK": "1"}
MODEL = "frontier/dep-x/qwen3-32b"


def _route(**overrides):
    route = {
        "model": MODEL,
        "name": None,
        "run_id": f"inference/{MODEL}",
        "enabled": False,
        "active": False,
        "inactive_reason": None,
    }
    return {**route, **overrides}


@pytest.fixture
def platform(monkeypatch):
    """Answers like the platform route; tests set `status`, `routes`, or `team_id`."""
    state = {"status": 200, "routes": [_route()], "requests": [], "team_id": "team-1"}

    def respond(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        assert request.url.path == "/api/v1/inference/trace-capture"
        if state["team_id"]:
            assert request.url.params["teamId"] == state["team_id"]
        else:
            assert "teamId" not in request.url.params
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"detail": "nope"})
        if request.method == "GET":
            return httpx.Response(200, json={"routes": state["routes"]})
        body = json.loads(request.content)
        return httpx.Response(200, json=_route(enabled=body["enabled"], active=body["enabled"]))

    def api_client():
        client = APIClient(api_key="test")
        client.client = httpx.Client(transport=httpx.MockTransport(respond))
        return client

    class Config:
        def __init__(self):
            self.team_id = state["team_id"]

    monkeypatch.setattr(trace_capture, "APIClient", api_client)
    monkeypatch.setattr(trace_capture, "Config", Config)
    # Both dimensions: rich ignores a bare width on dumb or non-tty terminals
    # (80x25 short-circuit), and ANSI styling around styled cells would break
    # the row assertions below.
    monkeypatch.setattr(trace_capture.console, "size", (200, 50))
    return state


def _run(*args):
    return runner.invoke(app, ["inference", "trace-capture", *args], env=ENV)


def test_list_shows_enabled_active_and_the_inactive_reason(platform):
    platform["routes"] = [
        _route(enabled=True, active=True),
        _route(
            model="team/other",
            run_id="inference/team/other",
            enabled=True,
            inactive_reason="storage_limit",
        ),
        _route(model="team/off", run_id="inference/team/off"),
        _route(
            model="team/unflagged",
            run_id="inference/team/unflagged",
            enabled=True,
            inactive_reason="enabler_unflagged",
        ),
    ]

    result = _run("list")

    assert result.exit_code == 0, result.output
    rows = [
        " ".join(strip_ansi(line).replace("│", " ").split()) for line in result.output.splitlines()
    ]
    assert any(f"{MODEL} yes active inference/{MODEL}" in row for row in rows)
    assert any("team/other yes inactive: trace storage limit reached" in row for row in rows)
    assert any("team/off no off inference/team/off" in row for row in rows)
    assert any("team/unflagged yes inactive: enabler lost capture access" in row for row in rows)
    assert "Trace capture is active on 1 of 4 route(s)." in result.output


def test_without_a_team_the_commands_target_the_personal_account(platform):
    platform["team_id"] = None

    list_result = _run("list")
    enable_result = _run("enable", MODEL, "--yes")

    assert list_result.exit_code == 0, list_result.output
    assert "teamId" not in platform["requests"][0].url.params
    assert enable_result.exit_code == 0, enable_result.output
    assert "teamId" not in platform["requests"][1].url.params


def test_enable_sends_the_exact_two_field_write(platform):
    result = _run("enable", MODEL, "--yes")

    assert result.exit_code == 0, result.output
    request = platform["requests"][-1]
    assert request.method == "PUT"
    assert json.loads(request.content) == {"model": MODEL, "enabled": True}
    assert "Status: active" in result.output
    assert f"prime traces list --run-id inference/{MODEL}" in result.output


def test_adapter_routes_are_spelled_in_the_model_id(platform):
    adapter_route = f"{MODEL}:lora-1"
    platform["routes"] = [_route(model=adapter_route, run_id=f"inference/{adapter_route}")]

    result = _run("disable", adapter_route, "-o", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(platform["requests"][-1].content) == {
        "model": adapter_route,
        "enabled": False,
    }
    assert json.loads(result.stdout)["enabled"] is False


@pytest.mark.parametrize(
    ("status", "message"),
    [(404, "Not available"), (403, "Not allowed"), (503, "inference is unreachable")],
)
def test_errors_name_their_cause(platform, status, message):
    platform["status"] = status

    result = _run("enable", MODEL, "--yes")

    assert result.exit_code == 1
    assert message in " ".join(result.output.split())
