import importlib

import pytest

from prime_mcp.tools import rl

mcp_module = importlib.import_module("prime_mcp.mcp")


@pytest.mark.asyncio
async def test_list_rl_runs_passes_team_filter(monkeypatch):
    captured = {}

    async def fake_make_prime_request(method, endpoint, params=None, json_data=None):
        captured.update(
            {
                "method": method,
                "endpoint": endpoint,
                "params": params,
                "json_data": json_data,
            }
        )
        return {"runs": []}

    monkeypatch.setattr(rl, "make_prime_request", fake_make_prime_request)

    result = await rl.list_rl_runs(team_id="team-123")

    assert result == {"runs": []}
    assert captured == {
        "method": "GET",
        "endpoint": "rft/runs",
        "params": {"team_id": "team-123"},
        "json_data": None,
    }


@pytest.mark.asyncio
async def test_get_rl_run_progress_uses_progress_endpoint(monkeypatch):
    captured = {}
    payload = {"latestStep": 42, "stepsWithSamples": [40, 41, 42]}

    async def fake_make_prime_request(method, endpoint, params=None, json_data=None):
        captured.update(
            {
                "method": method,
                "endpoint": endpoint,
                "params": params,
                "json_data": json_data,
            }
        )
        return payload

    monkeypatch.setattr(rl, "make_prime_request", fake_make_prime_request)

    result = await rl.get_rl_run_progress("run-123")

    assert result == payload
    assert captured == {
        "method": "GET",
        "endpoint": "rft/runs/run-123/progress",
        "params": None,
        "json_data": None,
    }


@pytest.mark.asyncio
async def test_get_rl_run_rollouts_passes_pagination(monkeypatch):
    captured = {}
    payload = {"samples": [{"prompt": "hi"}], "page": 2, "limit": 50}

    async def fake_make_prime_request(method, endpoint, params=None, json_data=None):
        captured.update(
            {
                "method": method,
                "endpoint": endpoint,
                "params": params,
                "json_data": json_data,
            }
        )
        return payload

    monkeypatch.setattr(rl, "make_prime_request", fake_make_prime_request)

    result = await rl.get_rl_run_rollouts("run-123", step=7, page=2, limit=50)

    assert result == payload
    assert captured == {
        "method": "GET",
        "endpoint": "rft/runs/run-123/samples",
        "params": {"step": 7, "page": 2, "limit": 50},
        "json_data": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"step": -1}, "step must be greater than or equal to 0"),
        ({"step": 1, "page": 0}, "page must be greater than 0"),
        ({"step": 1, "limit": 0}, "limit must be greater than 0"),
    ],
)
async def test_get_rl_run_rollouts_validates_pagination(monkeypatch, kwargs, message):
    async def fail_make_prime_request(*args, **kwargs):
        raise AssertionError("make_prime_request should not be called for invalid input")

    monkeypatch.setattr(rl, "make_prime_request", fail_make_prime_request)

    result = await rl.get_rl_run_rollouts("run-123", **kwargs)

    assert result == {"error": message}


@pytest.mark.asyncio
async def test_mcp_rollout_tools_delegate_to_rl_module(monkeypatch):
    async def fake_list(team_id):
        return {"team_id": team_id, "runs": []}

    async def fake_progress(run_id):
        return {"run_id": run_id, "latestStep": 9}

    async def fake_rollouts(run_id, step, page=1, limit=100):
        return {"run_id": run_id, "step": step, "page": page, "limit": limit}

    monkeypatch.setattr(mcp_module.rl, "list_rl_runs", fake_list)
    monkeypatch.setattr(mcp_module.rl, "get_rl_run_progress", fake_progress)
    monkeypatch.setattr(mcp_module.rl, "get_rl_run_rollouts", fake_rollouts)

    assert await mcp_module.list_rl_runs("team-123") == {"team_id": "team-123", "runs": []}
    assert await mcp_module.get_rl_run_progress("run-123") == {
        "run_id": "run-123",
        "latestStep": 9,
    }
    assert await mcp_module.get_rl_run_rollouts("run-123", step=3, page=2, limit=25) == {
        "run_id": "run-123",
        "step": 3,
        "page": 2,
        "limit": 25,
    }
