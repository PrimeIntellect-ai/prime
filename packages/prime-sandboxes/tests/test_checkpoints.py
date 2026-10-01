"""Filesystem checkpoint SDK requests and response parsing."""

from datetime import datetime, timezone

import pytest
import pytest_asyncio

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


@pytest_asyncio.fixture(params=[False, True], ids=["sync", "async"])
async def checkpoint_client(request):
    if request.param:
        async with AsyncSandboxClient(api_key="test-key") as client:
            yield client
    else:
        client = SandboxClient(APIClient(api_key="test-key"))
        try:
            yield client
        finally:
            client.client.client.close()


async def wait_for_checkpoint(client, timeout_seconds=300):
    if isinstance(client, AsyncSandboxClient):
        return await client.wait_for_checkpoint("checkpoint-1", timeout_seconds)
    return client.wait_for_checkpoint("checkpoint-1", timeout_seconds)


@pytest.fixture
def checkpoint_clock(monkeypatch):
    from prime_sandboxes import sandbox as module

    sleeps = []
    now = [0.0]

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    async def async_sleep(delay):
        sleep(delay)

    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", sleep)
    monkeypatch.setattr(module.asyncio, "sleep", async_sleep)
    monkeypatch.setattr(module.random, "uniform", lambda low, high: 0.0)
    return sleeps


def mock_checkpoint_status(client, monkeypatch, states, error=None):
    calls = []
    responses = iter(states)

    def get(checkpoint_id):
        calls.append(checkpoint_id)
        return SandboxCheckpoint.model_validate(
            {**checkpoint_response(next(responses)), "error": error}
        )

    async def async_get(checkpoint_id):
        return get(checkpoint_id)

    monkeypatch.setattr(
        client._checkpoint_batcher,
        "get",
        async_get if isinstance(client, AsyncSandboxClient) else get,
    )
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("states", [["DURABLE"], ["PENDING", "FROZEN", "DURABLE"]])
async def test_wait_for_checkpoint_durable(
    checkpoint_client, checkpoint_clock, monkeypatch, states
):
    calls = mock_checkpoint_status(checkpoint_client, monkeypatch, states)
    durable = await wait_for_checkpoint(checkpoint_client)
    assert durable.state == "DURABLE"
    assert calls == ["checkpoint-1"] * len(states)
    assert checkpoint_clock == ([] if len(states) == 1 else [1.0, 1.5])


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["FAILED", "DELETING"])
async def test_wait_for_checkpoint_terminal(
    checkpoint_client, checkpoint_clock, monkeypatch, state
):
    mock_checkpoint_status(checkpoint_client, monkeypatch, [state], error="upload failed")
    with pytest.raises(RuntimeError, match=f"checkpoint-1.*{state}.*upload failed"):
        await wait_for_checkpoint(checkpoint_client)
    assert checkpoint_clock == []


@pytest.mark.asyncio
async def test_wait_for_checkpoint_timeout(checkpoint_client, checkpoint_clock, monkeypatch):
    calls = mock_checkpoint_status(checkpoint_client, monkeypatch, ["PENDING", "FROZEN"])
    with pytest.raises(TimeoutError, match="checkpoint-1.*2s"):
        await wait_for_checkpoint(checkpoint_client, timeout_seconds=2)
    assert calls == ["checkpoint-1", "checkpoint-1"]
    assert checkpoint_clock == [1.0, 1.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
async def test_wait_for_checkpoint_invalid_timeout(checkpoint_client, monkeypatch, timeout):
    calls = mock_checkpoint_status(checkpoint_client, monkeypatch, [])
    with pytest.raises(ValueError, match="timeout_seconds"):
        await wait_for_checkpoint(checkpoint_client, timeout_seconds=timeout)
    assert calls == []


@pytest.mark.asyncio
async def test_wait_for_checkpoint_api_error(checkpoint_client, monkeypatch):
    from prime_sandboxes import APIError

    def get(checkpoint_id):
        raise APIError("HTTP 404: checkpoint not found")

    async def async_get(checkpoint_id):
        return get(checkpoint_id)

    monkeypatch.setattr(
        checkpoint_client._checkpoint_batcher,
        "get",
        async_get if isinstance(checkpoint_client, AsyncSandboxClient) else get,
    )
    with pytest.raises(APIError, match="404"):
        await wait_for_checkpoint(checkpoint_client)


@pytest.mark.asyncio
async def test_wait_for_checkpoint_cancellation(monkeypatch):
    import asyncio

    async with AsyncSandboxClient(api_key="test-key") as client:
        mock_checkpoint_status(client, monkeypatch, ["PENDING"])
        sleeping = asyncio.Event()

        async def sleep(delay):
            sleeping.set()
            await asyncio.Event().wait()

        monkeypatch.setattr("prime_sandboxes.sandbox.asyncio.sleep", sleep)
        task = asyncio.create_task(client.wait_for_checkpoint("checkpoint-1"))
        await sleeping.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def concurrent_checkpoint_waits(client, checkpoint_ids):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    if isinstance(client, AsyncSandboxClient):
        return await asyncio.gather(
            *(client.wait_for_checkpoint(checkpoint_id) for checkpoint_id in checkpoint_ids),
            return_exceptions=True,
        )
    barrier = threading.Barrier(len(checkpoint_ids))

    def wait(checkpoint_id):
        barrier.wait(timeout=5)
        try:
            return client.wait_for_checkpoint(checkpoint_id)
        except Exception as error:
            return error

    with ThreadPoolExecutor(max_workers=len(checkpoint_ids)) as pool:
        return list(pool.map(wait, checkpoint_ids))


def mock_checkpoint_batch(client, monkeypatch, missing=None, unsupported=False):
    from prime_sandboxes import APIError

    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if method == "GET":
            return {**checkpoint_response("DURABLE"), "id": path.rsplit("/", 1)[-1]}
        assert path == "/sandbox/checkpoints:batchGet"
        assert kwargs["idempotent_post"] is True
        if unsupported:
            raise APIError("HTTP 404: Not Found")
        ids = kwargs["json"]["checkpoint_ids"]
        return {
            "checkpoints": [
                {
                    **checkpoint_response("DURABLE"),
                    "id": checkpoint_id,
                    "sandbox_id": f"sandbox-{checkpoint_id}",
                }
                for checkpoint_id in ids
                if checkpoint_id != missing
            ],
            "errors": (
                [{"checkpoint_id": missing, "code": "NOT_FOUND", "message": "Checkpoint not found"}]
                if missing in ids
                else []
            ),
        }

    async def async_request(method, path, **kwargs):
        return request(method, path, **kwargs)

    monkeypatch.setattr(
        client.client,
        "request",
        async_request if isinstance(client, AsyncSandboxClient) else request,
    )
    return calls


@pytest.mark.asyncio
async def test_concurrent_checkpoint_waits_share_batch(checkpoint_client, monkeypatch):
    calls = mock_checkpoint_batch(checkpoint_client, monkeypatch)
    results = await concurrent_checkpoint_waits(checkpoint_client, ["c1", "c2", "c1", "c3"])
    assert [result.id for result in results] == ["c1", "c2", "c1", "c3"]
    assert len(calls) == 1
    assert set(calls[0][2]["json"]["checkpoint_ids"]) == {"c1", "c2", "c3"}
    assert checkpoint_client._operation_leases._active == {}


@pytest.mark.asyncio
async def test_checkpoint_batch_lookup_error_is_isolated(checkpoint_client, monkeypatch):
    from prime_sandboxes import APIError

    calls = mock_checkpoint_batch(checkpoint_client, monkeypatch, missing="c2")
    results = await concurrent_checkpoint_waits(checkpoint_client, ["c1", "c2", "c3"])
    assert results[0].id == "c1" and results[2].id == "c3"
    assert isinstance(results[1], APIError)
    assert "c2: NOT_FOUND" in str(results[1])
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_checkpoint_batch_falls_back_once(checkpoint_client, monkeypatch):
    calls = mock_checkpoint_batch(checkpoint_client, monkeypatch, unsupported=True)
    results = await concurrent_checkpoint_waits(checkpoint_client, ["c1", "c2"])
    assert [result.id for result in results] == ["c1", "c2"]
    result = (
        await checkpoint_client.wait_for_checkpoint("c3")
        if isinstance(checkpoint_client, AsyncSandboxClient)
        else checkpoint_client.wait_for_checkpoint("c3")
    )
    assert result.id == "c3"
    assert [method for method, _, _ in calls].count("POST") == 1
    assert [method for method, _, _ in calls].count("GET") == 3


@pytest.mark.asyncio
async def test_checkpoint_batches_split_at_100(monkeypatch):
    async with AsyncSandboxClient(api_key="test-key") as client:
        calls = mock_checkpoint_batch(client, monkeypatch)
        ids = [f"c{index}" for index in range(101)]
        results = await concurrent_checkpoint_waits(client, ids)
        assert [result.id for result in results] == ids
        assert [len(kwargs["json"]["checkpoint_ids"]) for _, _, kwargs in calls] == [100, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [[], ["c"] * 2, [f"c{i}" for i in range(101)], [""], ["c" * 65]])
async def test_get_checkpoints_validates_ids(checkpoint_client, monkeypatch, ids):
    calls = mock_checkpoint_batch(checkpoint_client, monkeypatch)
    with pytest.raises(ValueError):
        if isinstance(checkpoint_client, AsyncSandboxClient):
            await checkpoint_client.get_checkpoints(ids)
        else:
            checkpoint_client.get_checkpoints(ids)
    assert calls == []


@pytest.mark.asyncio
async def test_checkpoint_batch_close_cancels_inflight_fetch(monkeypatch):
    import asyncio

    client = AsyncSandboxClient(api_key="test-key")
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def request(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(client.client, "request", request)
    waiter = asyncio.create_task(client.wait_for_checkpoint("c1"))
    await started.wait()
    await client.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await waiter
    assert cancelled.is_set()
    assert client._operation_leases._active == {}


@pytest.mark.asyncio
async def test_cancelling_one_checkpoint_waiter_keeps_shared_batch(monkeypatch):
    import asyncio

    async with AsyncSandboxClient(api_key="test-key") as client:
        started = asyncio.Event()
        finish = asyncio.Event()
        calls = []

        async def request(method, path, **kwargs):
            calls.append(kwargs["json"]["checkpoint_ids"])
            started.set()
            await finish.wait()
            return {"checkpoints": [checkpoint_response("DURABLE")], "errors": []}

        monkeypatch.setattr(client.client, "request", request)
        first = asyncio.create_task(client.wait_for_checkpoint("checkpoint-1"))
        second = asyncio.create_task(client.wait_for_checkpoint("checkpoint-1"))
        await started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        finish.set()
        assert (await second).state == "DURABLE"
        assert calls == [["checkpoint-1"]]
        assert client._operation_leases._active == {}
