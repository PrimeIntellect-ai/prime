"""Unit tests for bounded-memory gateway file transfers."""

from datetime import datetime, timezone

import httpx
import pytest
from prime_sandboxes import APIClient, AsyncSandboxClient, SandboxClient
from prime_sandboxes import sandbox as sandbox_module

AUTH = {
    "gateway_url": "https://gateway.example.com",
    "user_ns": "ns",
    "job_id": "job",
    "token": "tok",
}


class _SyncAuthCache:
    def get_or_refresh(self, _sandbox_id: str):
        return AUTH


class _AsyncAuthCache:
    async def get_or_refresh(self, _sandbox_id: str):
        return AUTH


class _ChunkedSyncStream(httpx.SyncByteStream):
    def __iter__(self):
        yield b"first chunk\n"
        yield b"second chunk\n"


class _ChunkedAsyncStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"first chunk\n"
        yield b"second chunk\n"


def _upload_response() -> httpx.Response:
    request = httpx.Request("POST", "https://gateway.example.com/upload")
    return httpx.Response(
        200,
        request=request,
        json={
            "success": True,
            "path": "/tmp/uploaded.txt",
            "size": 12,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def test_sync_upload_passes_file_object_without_reading_into_memory(tmp_path, monkeypatch):
    local_path = tmp_path / "upload.txt"
    local_path.write_bytes(b"upload contents")
    client = SandboxClient(APIClient(api_key="test-key"))
    client._auth_cache = _SyncAuthCache()
    captured = {}

    def fake_post(_url, **kwargs):
        file_object = kwargs["files"]["file"][1]
        captured["is_bytes"] = isinstance(file_object, bytes)
        captured["content"] = file_object.read()
        return _upload_response()

    monkeypatch.setattr(client, "_gateway_post", fake_post)

    result = client.upload_file("sandbox-1", "/tmp/uploaded.txt", str(local_path))

    assert result.success is True
    assert captured == {"is_bytes": False, "content": b"upload contents"}


@pytest.mark.asyncio
async def test_async_upload_passes_file_object_without_reading_into_memory(tmp_path, monkeypatch):
    local_path = tmp_path / "upload.txt"
    local_path.write_bytes(b"upload contents")
    client = AsyncSandboxClient(api_key="test-key")
    client._auth_cache = _AsyncAuthCache()
    captured = {}

    async def fake_post(_url, **kwargs):
        file_object = kwargs["files"]["file"][1]
        captured["is_bytes"] = isinstance(file_object, bytes)
        captured["content"] = file_object.read()
        return _upload_response()

    monkeypatch.setattr(client, "_gateway_post", fake_post)

    result = await client.upload_file("sandbox-1", "/tmp/uploaded.txt", str(local_path))

    assert result.success is True
    assert captured == {"is_bytes": False, "content": b"upload contents"}
    await client.aclose()


def test_sync_download_streams_and_replaces_destination_atomically(tmp_path, monkeypatch):
    destination = tmp_path / "nested" / "download.txt"
    destination.parent.mkdir()
    destination.write_bytes(b"old contents")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, request=request, stream=_ChunkedSyncStream())

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    def client_factory(*args, **kwargs):
        return original_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(sandbox_module.httpx, "Client", client_factory)
    client = SandboxClient(APIClient(api_key="test-key"))
    client._auth_cache = _SyncAuthCache()

    client.download_file("sandbox-1", "/tmp/remote.txt", str(destination))

    assert destination.read_bytes() == b"first chunk\nsecond chunk\n"
    assert len(requests) == 1
    assert list(tmp_path.rglob("*.tmp")) == []


@pytest.mark.asyncio
async def test_async_download_streams_and_replaces_destination_atomically(tmp_path):
    destination = tmp_path / "nested" / "download.txt"
    destination.parent.mkdir()
    destination.write_bytes(b"old contents")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, request=request, stream=_ChunkedAsyncStream())

    client = AsyncSandboxClient(api_key="test-key")
    client._auth_cache = _AsyncAuthCache()
    client._gateway_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await client.download_file("sandbox-1", "/tmp/remote.txt", str(destination))

    assert destination.read_bytes() == b"first chunk\nsecond chunk\n"
    assert len(requests) == 1
    assert list(tmp_path.rglob("*.tmp")) == []
    await client.aclose()


def test_sync_download_keeps_existing_destination_when_gateway_fails(tmp_path, monkeypatch):
    destination = tmp_path / "download.txt"
    destination.write_bytes(b"old contents")

    def handler(request):
        return httpx.Response(404, request=request, text="missing")

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    def client_factory(*args, **kwargs):
        return original_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(sandbox_module.httpx, "Client", client_factory)
    client = SandboxClient(APIClient(api_key="test-key"))
    client._auth_cache = _SyncAuthCache()

    with pytest.raises(sandbox_module.APIError, match="HTTP 404"):
        client.download_file("sandbox-1", "/tmp/missing.txt", str(destination))

    assert destination.read_bytes() == b"old contents"
    assert list(tmp_path.rglob("*.tmp")) == []
