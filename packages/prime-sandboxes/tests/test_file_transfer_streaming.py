"""Hermetic tests for streamed uploads and downloads.

The live file-transfer tests in ``test_file_operations.py`` cover gateway
semantics against a real sandbox. These cover the client-side contract that
matters locally: transfers move bytes in bounded pieces, a failed download
leaves the destination alone, and a retried upload resends the whole file.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from prime_sandboxes.core.client import APIError
from prime_sandboxes.sandbox import (
    TRANSFER_CHUNK_BYTES,
    AsyncSandboxClient,
    SandboxClient,
)

_AUTH = {
    "gateway_url": "https://gateway.example",
    "user_ns": "test-ns",
    "job_id": "job-123",
    "token": "test-token",
}

_UPLOAD_RESPONSE = {
    "success": True,
    "path": "/tmp/uploaded.bin",
    "size": 1,
    "timestamp": "2026-01-01T00:00:00Z",
}

# A streamed upload arrives as many small pieces. A body handed over as one
# buffer arrives as a handful of framing pieces plus a single piece the size of
# the file, so anything at or above our own chunk size means it was buffered.
_MAX_STREAMED_PIECE = TRANSFER_CHUNK_BYTES


class _SyncAuthCache:
    def get_or_refresh(self, _sandbox_id: str) -> dict[str, str]:
        return dict(_AUTH)


class _AsyncAuthCache:
    async def get_or_refresh(self, _sandbox_id: str) -> dict[str, str]:
        return dict(_AUTH)


def _sync_client() -> SandboxClient:
    client = SandboxClient.__new__(SandboxClient)
    client._auth_cache = _SyncAuthCache()  # type: ignore[assignment]
    return client


@asynccontextmanager
async def _async_client(transport: httpx.AsyncBaseTransport):
    client = AsyncSandboxClient.__new__(AsyncSandboxClient)
    client._auth_cache = _AsyncAuthCache()  # type: ignore[assignment]
    async with httpx.AsyncClient(transport=transport) as gateway:
        client._gateway_client = gateway
        yield client


@pytest.fixture
def sync_transport(monkeypatch):
    """Route the httpx.Client the sync gateway helpers create to a test transport."""
    real_client = httpx.Client

    def install(transport: httpx.BaseTransport) -> None:
        def factory(*args, **kwargs):
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        monkeypatch.setattr("prime_sandboxes.sandbox.httpx.Client", factory)

    return install


@pytest.fixture(autouse=True)
def no_transfer_retry_sleep(monkeypatch):
    """Drop the retry backoff so retry tests do not wait out the exponential delay."""

    async def async_no_sleep(_delay):
        return None

    for name in ("_gateway_upload_file", "_gateway_download_to_file"):
        monkeypatch.setattr(getattr(SandboxClient, name).retry, "sleep", lambda _delay: None)
        monkeypatch.setattr(getattr(AsyncSandboxClient, name).retry, "sleep", async_no_sleep)


class _ChunkedStream(httpx.SyncByteStream):
    """A response body that hands over one chunk at a time."""

    def __init__(self, chunks, on_chunk=None):
        self._chunks = chunks
        self._on_chunk = on_chunk

    def __iter__(self):
        for index, chunk in enumerate(self._chunks):
            if self._on_chunk is not None:
                self._on_chunk(index)
            yield chunk


class _AsyncChunkedStream(httpx.AsyncByteStream):
    """Async response body that hands over one chunk at a time."""

    def __init__(self, chunks, on_chunk=None):
        self._chunks = chunks
        self._on_chunk = on_chunk

    async def __aiter__(self):
        for index, chunk in enumerate(self._chunks):
            if self._on_chunk is not None:
                self._on_chunk(index)
            yield chunk


class _BrokenStream(httpx.SyncByteStream):
    """Streams one piece, then drops the connection mid-body."""

    def __iter__(self):
        yield b"partial"
        raise httpx.ReadError("connection dropped")


class _AsyncBrokenStream(httpx.AsyncByteStream):
    """Async body that streams one piece, then drops the connection."""

    async def __aiter__(self):
        yield b"partial"
        raise httpx.ReadError("connection dropped")


class _RecordingUploadTransport(httpx.BaseTransport):
    """Reads the request stream the way a socket would, piece by piece.

    Deliberately not a MockTransport: that calls ``request.read()`` first, which
    collapses the multipart body into a single buffer and hides whether the file
    was streamed.
    """

    def __init__(self, fail_times: int = 0):
        self._fail_times = fail_times
        self.attempts = 0
        self.pieces: list[list[int]] = []
        self.bodies: list[bytes] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        chunks = list(request.stream)
        self.pieces.append([len(chunk) for chunk in chunks])
        self.bodies.append(b"".join(chunks))
        if self.attempts <= self._fail_times:
            raise httpx.RemoteProtocolError("server disconnected", request=request)
        return httpx.Response(200, json=_UPLOAD_RESPONSE, request=request)


class _AsyncRecordingUploadTransport(httpx.AsyncBaseTransport):
    """Async twin of _RecordingUploadTransport."""

    def __init__(self, fail_times: int = 0):
        self._fail_times = fail_times
        self.attempts = 0
        self.pieces: list[list[int]] = []
        self.bodies: list[bytes] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        chunks = [chunk async for chunk in request.stream]
        self.pieces.append([len(chunk) for chunk in chunks])
        self.bodies.append(b"".join(chunks))
        if self.attempts <= self._fail_times:
            raise httpx.RemoteProtocolError("server disconnected", request=request)
        return httpx.Response(200, json=_UPLOAD_RESPONSE, request=request)


def _part_files(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.part"))


class TestSyncDownloadStreaming:
    def test_streams_the_body_into_a_part_file(self, tmp_path, sync_transport):
        """The in-flight file exists while the body is still arriving.

        A buffer-then-write download would consume every chunk before touching
        the filesystem, so no part file would exist at these points.
        """
        destination = tmp_path / "payload.bin"
        part_present_when_chunk_starts: list[bool] = []

        def on_chunk(_index):
            part_present_when_chunk_starts.append(any(tmp_path.glob("*.part")))

        chunks = [b"a" * 8, b"b" * 8, b"c" * 8]

        def handler(request):
            return httpx.Response(200, stream=_ChunkedStream(chunks, on_chunk), request=request)

        sync_transport(httpx.MockTransport(handler))
        _sync_client().download_file("sb-1", "/remote.bin", str(destination))

        assert destination.read_bytes() == b"".join(chunks)
        assert part_present_when_chunk_starts == [True, True, True]

    def test_does_not_read_the_body_into_memory(self, tmp_path, monkeypatch, sync_transport):
        """Reading the body (``response.read``/``.content``) would defeat the point."""
        destination = tmp_path / "payload.bin"

        def forbidden_read(_self):
            raise AssertionError("the download body must be streamed, not buffered")

        monkeypatch.setattr(httpx.Response, "read", forbidden_read)

        def handler(request):
            return httpx.Response(200, stream=_ChunkedStream([b"a" * 64]), request=request)

        sync_transport(httpx.MockTransport(handler))
        _sync_client().download_file("sb-1", "/remote.bin", str(destination))

        assert destination.read_bytes() == b"a" * 64

    def test_failed_transfer_leaves_destination_and_directory_clean(self, tmp_path, sync_transport):
        destination = tmp_path / "payload.bin"
        destination.write_bytes(b"previous contents")

        def handler(request):
            return httpx.Response(200, stream=_BrokenStream(), request=request)

        sync_transport(httpx.MockTransport(handler))

        with pytest.raises(APIError) as exc_info:
            _sync_client().download_file("sb-1", "/remote.bin", str(destination))

        assert "Download failed" in str(exc_info.value)
        assert destination.read_bytes() == b"previous contents"
        assert _part_files(tmp_path) == []

    def test_retries_a_body_read_error_before_giving_up(self, tmp_path, sync_transport):
        """A dropped connection mid-body is retried on the existing idempotent budget."""
        destination = tmp_path / "payload.bin"
        streams: list[_BrokenStream] = []

        def handler(request):
            stream = _BrokenStream()
            streams.append(stream)
            return httpx.Response(200, stream=stream, request=request)

        sync_transport(httpx.MockTransport(handler))

        with pytest.raises(APIError):
            _sync_client().download_file("sb-1", "/remote.bin", str(destination))

        assert len(streams) == 4
        assert not destination.exists()
        assert _part_files(tmp_path) == []

    def test_reports_the_gateway_error_body(self, tmp_path, sync_transport):
        """A streamed error body is still readable from the raised response."""
        destination = tmp_path / "payload.bin"

        def handler(request):
            return httpx.Response(404, stream=_ChunkedStream([b"no such file"]), request=request)

        sync_transport(httpx.MockTransport(handler))

        with pytest.raises(APIError) as exc_info:
            _sync_client().download_file("sb-1", "/remote.bin", str(destination))

        assert "no such file" in str(exc_info.value)
        assert not destination.exists()
        assert _part_files(tmp_path) == []


class TestSyncUploadStreaming:
    def test_sends_the_file_in_bounded_pieces(self, tmp_path, sync_transport):
        payload = tmp_path / "payload.bin"
        payload.write_bytes(b"x" * (2 * 1024 * 1024))
        transport = _RecordingUploadTransport()
        sync_transport(transport)

        _sync_client().upload_file("sb-1", "/remote.bin", str(payload))

        assert transport.attempts == 1
        pieces = transport.pieces[0]
        assert payload.read_bytes() in transport.bodies[0]
        # A body handed over as one buffer produces a handful of framing pieces
        # with a single piece the size of the file.
        assert len(pieces) > 8
        assert max(pieces) <= _MAX_STREAMED_PIECE

    def test_retry_resends_the_whole_file(self, tmp_path, sync_transport):
        payload = tmp_path / "payload.bin"
        payload.write_bytes(b"y" * (256 * 1024))
        transport = _RecordingUploadTransport(fail_times=1)
        sync_transport(transport)

        _sync_client().upload_file("sb-1", "/remote.bin", str(payload))

        assert transport.attempts == 2
        assert len(transport.bodies[0]) == len(transport.bodies[1])
        for body in transport.bodies:
            assert payload.read_bytes() in body


@pytest.mark.asyncio
class TestAsyncDownloadStreaming:
    async def test_streams_the_body_into_a_part_file(self, tmp_path):
        destination = tmp_path / "payload.bin"
        part_present_when_chunk_starts: list[bool] = []

        def on_chunk(_index):
            part_present_when_chunk_starts.append(any(tmp_path.glob("*.part")))

        chunks = [b"a" * 8, b"b" * 8, b"c" * 8]

        def handler(request):
            return httpx.Response(
                200, stream=_AsyncChunkedStream(chunks, on_chunk), request=request
            )

        async with _async_client(httpx.MockTransport(handler)) as client:
            await client.download_file("sb-1", "/remote.bin", str(destination))

        assert destination.read_bytes() == b"".join(chunks)
        assert part_present_when_chunk_starts == [True, True, True]

    async def test_does_not_read_the_body_into_memory(self, tmp_path, monkeypatch):
        destination = tmp_path / "payload.bin"

        async def forbidden_read(_self):
            raise AssertionError("the download body must be streamed, not buffered")

        monkeypatch.setattr(httpx.Response, "aread", forbidden_read)

        def handler(request):
            return httpx.Response(200, stream=_AsyncChunkedStream([b"a" * 64]), request=request)

        async with _async_client(httpx.MockTransport(handler)) as client:
            await client.download_file("sb-1", "/remote.bin", str(destination))

        assert destination.read_bytes() == b"a" * 64

    async def test_failed_transfer_leaves_destination_and_directory_clean(self, tmp_path):
        destination = tmp_path / "payload.bin"
        destination.write_bytes(b"previous contents")

        def handler(request):
            return httpx.Response(200, stream=_AsyncBrokenStream(), request=request)

        async with _async_client(httpx.MockTransport(handler)) as client:
            with pytest.raises(APIError) as exc_info:
                await client.download_file("sb-1", "/remote.bin", str(destination))

        assert "Download failed" in str(exc_info.value)
        assert destination.read_bytes() == b"previous contents"
        assert _part_files(tmp_path) == []

    async def test_retries_a_body_read_error_before_giving_up(self, tmp_path):
        destination = tmp_path / "payload.bin"
        streams: list[_AsyncBrokenStream] = []

        def handler(request):
            stream = _AsyncBrokenStream()
            streams.append(stream)
            return httpx.Response(200, stream=stream, request=request)

        async with _async_client(httpx.MockTransport(handler)) as client:
            with pytest.raises(APIError):
                await client.download_file("sb-1", "/remote.bin", str(destination))

        assert len(streams) == 4
        assert not destination.exists()
        assert _part_files(tmp_path) == []

    async def test_reports_the_gateway_error_body(self, tmp_path):
        destination = tmp_path / "payload.bin"

        def handler(request):
            return httpx.Response(
                404, stream=_AsyncChunkedStream([b"no such file"]), request=request
            )

        async with _async_client(httpx.MockTransport(handler)) as client:
            with pytest.raises(APIError) as exc_info:
                await client.download_file("sb-1", "/remote.bin", str(destination))

        assert "no such file" in str(exc_info.value)
        assert not destination.exists()
        assert _part_files(tmp_path) == []


@pytest.mark.asyncio
class TestAsyncUploadStreaming:
    async def test_sends_the_file_in_bounded_pieces(self, tmp_path):
        payload = tmp_path / "payload.bin"
        payload.write_bytes(b"x" * (2 * 1024 * 1024))
        transport = _AsyncRecordingUploadTransport()

        async with _async_client(transport) as client:
            await client.upload_file("sb-1", "/remote.bin", str(payload))

        assert transport.attempts == 1
        pieces = transport.pieces[0]
        assert payload.read_bytes() in transport.bodies[0]
        assert len(pieces) > 8
        assert max(pieces) <= _MAX_STREAMED_PIECE

    async def test_retry_resends_the_whole_file(self, tmp_path):
        payload = tmp_path / "payload.bin"
        payload.write_bytes(b"y" * (256 * 1024))
        transport = _AsyncRecordingUploadTransport(fail_times=1)

        async with _async_client(transport) as client:
            await client.upload_file("sb-1", "/remote.bin", str(payload))

        assert transport.attempts == 2
        assert len(transport.bodies[0]) == len(transport.bodies[1])
        for body in transport.bodies:
            assert payload.read_bytes() in body
