"""Filesystem checkpoint SDK requests and response parsing."""

from datetime import datetime, timezone

import pytest

from prime_sandboxes import APIClient, AsyncSandboxClient, SandboxCheckpoint, SandboxClient


def checkpoint_response(state: str = "PENDING") -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": "checkpoint-1",
        "sandbox_id": "sandbox-1",
        "parent_id": None,
        "team_id": None,
        "state": state,
        "depth": 1,
        "docker_image": "python:3.11-slim",
        "created_at": now,
        "updated_at": now,
    }


def test_checkpoint_sync_requests_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    calls = []

    def request(method: str, path: str) -> dict:
        calls.append((method, path))
        return checkpoint_response("PENDING" if method == "POST" else "DURABLE")

    monkeypatch.setattr(client.client, "request", request)
    created = client.checkpoint("sandbox-1")
    durable = client.get_checkpoint(created.id)

    assert isinstance(created, SandboxCheckpoint)
    assert created.state == "PENDING"
    assert durable.state == "DURABLE"
    assert calls == [
        ("POST", "/sandbox/sandbox-1/checkpoints"),
        ("GET", "/sandbox/checkpoints/checkpoint-1"),
    ]


@pytest.mark.asyncio
async def test_checkpoint_async_requests_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    async with AsyncSandboxClient(api_key="test-key") as client:
        calls = []

        async def request(method: str, path: str) -> dict:
            calls.append((method, path))
            return checkpoint_response("PENDING" if method == "POST" else "DURABLE")

        monkeypatch.setattr(client.client, "request", request)
        created = await client.checkpoint("sandbox-1")
        durable = await client.get_checkpoint(created.id)

    assert created.state == "PENDING"
    assert durable.state == "DURABLE"
    assert calls == [
        ("POST", "/sandbox/sandbox-1/checkpoints"),
        ("GET", "/sandbox/checkpoints/checkpoint-1"),
    ]
