"""`prime inference trace-capture` against a fake of the platform's HTTP contract."""

import json

import httpx
import pytest
from prime_cli.client import APIClient
from prime_cli.commands import trace_capture
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()
ENV = {"NO_COLOR": "1", "PRIME_DISABLE_VERSION_CHECK": "1"}
MODEL = "frontier/dep-x/qwen3-32b"


def _route(**overrides):
    route = {
        "model": MODEL,
        "lora_id": "__base__",
        "name": None,
        "run_id": f"inference/{MODEL}",
        "enabled": False,
        "active": False,
        "inactive_reason": None,
    }
    return {**route, **overrides}


@pytest.fixture
def platform(monkeypatch):
    """Answers like the platform route; tests set `status` or `routes`."""
    state = {"status": 200, "routes": [_route()], "requests": []}

    def respond(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        assert request.url.path == "/api/v1/inference/trace-capture"
        assert request.url.params["teamId"] == "team-1"
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

    class TeamConfig:
        team_id = "team-1"

    monkeypatch.setattr(trace_capture, "APIClient", api_client)
    monkeypatch.setattr(trace_capture, "Config", TeamConfig)
    monkeypatch.setattr(trace_capture.console, "width", 200)
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
    ]

    result = _run("list")

    assert result.exit_code == 0, result.output
    rows = [" ".join(line.replace("│", " ").split()) for line in result.output.splitlines()]
    assert any(f"{MODEL} - yes active inference/{MODEL}" in row for row in rows)
    assert any("team/other - yes inactive: trace storage limit reached" in row for row in rows)
    assert any("team/off - no off inference/team/off" in row for row in rows)
    assert "Custom trace capture is active on 1 of 3 route(s)." in result.output


def test_enable_puts_the_route_and_reports_it_active(platform):
    result = _run("enable", MODEL, "--yes")

    assert result.exit_code == 0, result.output
    request = platform["requests"][-1]
    assert request.method == "PUT"
    assert json.loads(request.content) == {"model": MODEL, "lora_id": "__base__", "enabled": True}
    assert "Status: active" in result.output
    assert f"prime traces list --run-id inference/{MODEL}" in result.output


def test_disable_sends_the_adapter_and_returns_json(platform):
    result = _run("disable", MODEL, "--lora-id", "lora-1", "-o", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(platform["requests"][-1].content)["lora_id"] == "lora-1"
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
