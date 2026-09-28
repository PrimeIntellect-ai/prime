"""Filesystem checkpoint SDK requests and response parsing."""

from datetime import datetime, timezone

import pytest

from prime_sandboxes import (
    APIClient,
    AsyncSandboxClient,
    CreateSandboxRequest,
    SandboxCheckpoint,
    SandboxClient,
)


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


def sandbox_response() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": "restored-1",
        "name": "restored",
        "docker_image": "python:3.11-slim",
        "cpu_cores": 1.0,
        "memory_gb": 1.0,
        "disk_size_gb": 5.0,
        "disk_mount_path": "/sandbox-workspace",
        "gpu_count": 0,
        "status": "PENDING",
        "timeout_minutes": 60,
        "created_at": now,
        "updated_at": now,
    }


def test_restore_request_rejects_image_and_disk_size() -> None:
    with pytest.raises(ValueError, match="omit docker_image"):
        CreateSandboxRequest(
            name="restored", checkpoint_id="checkpoint-1", docker_image="python:3.11-slim"
        )
    with pytest.raises(ValueError, match="omit docker_image"):
        CreateSandboxRequest(name="restored", checkpoint_id="checkpoint-1", disk_size_gb=10)


def test_restore_create_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    calls = []

    def request(method: str, path: str, **kwargs: object) -> dict:
        calls.append((method, path, kwargs))
        return sandbox_response()

    monkeypatch.setattr(client.client, "request", request)
    restored = client.create(CreateSandboxRequest(name="restored", checkpoint_id="checkpoint-1"))

    assert restored.id == "restored-1"
    method, path, kwargs = calls[0]
    assert (method, path) == ("POST", "/sandbox")
    payload = kwargs["json"]
    assert payload["checkpoint_id"] == "checkpoint-1"
    assert payload["vm"] is True
    assert "docker_image" not in payload
    assert "disk_size_gb" not in payload


@pytest.mark.asyncio
async def test_async_restore_create_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    async with AsyncSandboxClient(api_key="test-key") as client:
        calls = []

        async def request(method: str, path: str, **kwargs: object) -> dict:
            calls.append((method, path, kwargs))
            return sandbox_response()

        monkeypatch.setattr(client.client, "request", request)
        restored = await client.create(
            CreateSandboxRequest(name="restored", checkpoint_id="checkpoint-1")
        )

    assert restored.id == "restored-1"
    payload = calls[0][2]["json"]
    assert payload["checkpoint_id"] == "checkpoint-1"
    assert "docker_image" not in payload
    assert "disk_size_gb" not in payload


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


def test_list_checkpoints_sync_and_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SandboxClient(APIClient(api_key="test-key"))
    calls = []

    def request(method: str, path: str, params: dict | None = None) -> dict:
        calls.append((method, path, params))
        return {"checkpoints": [checkpoint_response("DURABLE")]}

    monkeypatch.setattr(client.client, "request", request)
    listed = client.list_checkpoints("sandbox-1")
    filtered = client.list_checkpoints("sandbox-1", "checkpoint-1")

    assert [c.state for c in listed] == ["DURABLE"]
    assert filtered[0].id == "checkpoint-1"
    assert calls == [
        ("GET", "/sandbox/sandbox-1/checkpoints", None),
        ("GET", "/sandbox/sandbox-1/checkpoints", {"checkpoint_id": "checkpoint-1"}),
    ]


@pytest.mark.asyncio
async def test_list_checkpoints_async(monkeypatch: pytest.MonkeyPatch) -> None:
    async with AsyncSandboxClient(api_key="test-key") as client:
        calls = []

        async def request(method: str, path: str, params: dict | None = None) -> dict:
            calls.append((method, path, params))
            return {"checkpoints": []}

        monkeypatch.setattr(client.client, "request", request)
        listed = await client.list_checkpoints("sandbox-1", "checkpoint-1")

    assert listed == []
    assert calls == [
        ("GET", "/sandbox/sandbox-1/checkpoints", {"checkpoint_id": "checkpoint-1"}),
    ]
