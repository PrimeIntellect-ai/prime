"""Async upload path.

Parity with the sync uploader is the point: the same lines must produce the
same batches, the same idempotency keys and the same wire bytes, whichever
client sent them. What is tested here beyond that is what only the async
client can do — accept an async producer, await an async ``on_batch`` hook,
and keep the event loop free while it reads and hashes.
"""

import asyncio
import gzip
import hashlib
import json
import threading

import httpx
import pytest
from test_upload_wire import parse_multipart

from prime_traces import (
    RetryableAPIError,
    TransportError,
    ValidationRejectedError,
)

RAW = b'{"id":"a"}\n{"id":"b"}\n'
LINES = [b'{"id":"a"}\n', b'{"id":"b"}\n']
COMMITTED = {"upload_id": "x" * 64, "status": "committed"}


class TestRequestShape:
    @pytest.mark.asyncio
    async def test_upload_lines_matches_the_sync_wire_contract(self, make_async_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            captured["content"] = request.content
            return httpx.Response(201, json=COMMITTED)

        client = make_async_client(handler)
        [receipt] = await client.upload_lines(
            LINES, context={"source": "hosted_eval"}, compress=False
        )

        request = captured["request"]
        assert request.url.path == "/api/v1/traces"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["Idempotency-Key"] == f"sha256:{hashlib.sha256(RAW).hexdigest()}"
        # Bare-trace format is the default and the header stays off the wire.
        assert "X-Prime-Line-Format" not in request.headers

        parts = parse_multipart(captured["content"], request.headers["content-type"])
        assert parts["traces"][1] == RAW
        assert json.loads(parts["metadata"][1]) == {
            "schema_version": 1,
            "context": {"source": "hosted_eval"},
        }
        assert receipt.status == "committed"

    @pytest.mark.asyncio
    async def test_compression_is_transport_only(self, make_async_client):
        """Gzipping happens in a worker thread; the digest and the decompressed
        bytes must still be the ones defined over the uncompressed input."""
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            captured["content"] = request.content
            return httpx.Response(201, json=COMMITTED)

        await make_async_client(handler).upload_lines(LINES)

        request = captured["request"]
        parts = parse_multipart(captured["content"], request.headers["content-type"])
        headers, body = parts["traces"]
        assert headers["content-encoding"] == "gzip"
        assert gzip.decompress(body) == RAW
        assert request.headers["Idempotency-Key"] == f"sha256:{hashlib.sha256(RAW).hexdigest()}"

    @pytest.mark.asyncio
    async def test_upload_file_reads_and_batches_like_the_sync_client(
        self, make_async_client, tmp_path
    ):
        path = tmp_path / "traces.jsonl"
        path.write_bytes(RAW)
        keys = []

        def handler(request: httpx.Request) -> httpx.Response:
            keys.append(request.headers["Idempotency-Key"])
            return httpx.Response(201, json=COMMITTED)

        receipts = await make_async_client(handler).upload_file(path, compress=False)
        assert len(receipts) == 1
        assert keys == [f"sha256:{hashlib.sha256(RAW).hexdigest()}"]


class TestAsyncProducers:
    @pytest.mark.asyncio
    async def test_async_records_upload_without_being_collected(self, make_async_client):
        captured = {}

        class RecordObject:
            def to_record(self):
                return {"id": "b", "nested": {"ok": True}}

        async def records():
            yield {"id": "a", "label": "café"}
            yield RecordObject()

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            captured["content"] = request.content
            return httpx.Response(201, json=COMMITTED)

        client = make_async_client(handler)
        [receipt] = await client.upload_records(records(), compress=False)

        # Byte-for-byte what the sync client produces for the same records.
        expected = b'{"id":"a","label":"caf\xc3\xa9"}\n{"id":"b","nested":{"ok":true}}\n'
        request = captured["request"]
        assert request.headers["Idempotency-Key"] == (
            f"sha256:{hashlib.sha256(expected).hexdigest()}"
        )
        parts = parse_multipart(captured["content"], request.headers["content-type"])
        assert parts["traces"][1] == expected
        assert receipt.status == "committed"

    @pytest.mark.asyncio
    async def test_async_record_serialization_runs_off_the_event_loop(self, make_async_client):
        loop_thread = threading.get_ident()
        serialization_thread = None

        class RecordObject:
            def to_record(self):
                nonlocal serialization_thread
                serialization_thread = threading.get_ident()
                return {"id": "a"}

        async def records():
            yield RecordObject()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json=COMMITTED)

        await make_async_client(handler).upload_records(records(), compress=False)

        assert serialization_thread is not None
        assert serialization_thread != loop_thread

    @pytest.mark.asyncio
    async def test_cancellation_waits_for_async_record_serialization(self, make_async_client):
        serialization_started = threading.Event()
        release_serialization = threading.Event()
        serialization_finished = threading.Event()
        producer_closed = False

        class SlowRecord:
            def to_record(self):
                serialization_started.set()
                release_serialization.wait()
                serialization_finished.set()
                return {"id": "a"}

        async def records():
            nonlocal producer_closed
            try:
                yield SlowRecord()
            finally:
                producer_closed = True

        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail(f"unexpected request: {request.url}")

        task = asyncio.create_task(
            make_async_client(handler).upload_records(records(), compress=False)
        )
        await asyncio.to_thread(serialization_started.wait)

        task.cancel()
        await asyncio.sleep(0.01)

        assert not task.done()
        assert not serialization_finished.is_set()
        assert not producer_closed

        release_serialization.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert serialization_finished.is_set()
        assert producer_closed

    @pytest.mark.asyncio
    async def test_repeated_cancellation_waits_for_async_record_producer_close(
        self, make_async_client
    ):
        serialization_started = threading.Event()
        release_serialization = threading.Event()
        close_started = asyncio.Event()
        release_close = asyncio.Event()
        producer_closed = False

        class SlowRecord:
            def to_record(self):
                serialization_started.set()
                release_serialization.wait()
                return {"id": "a"}

        class SlowClosingRecords:
            def __init__(self):
                self.first = True

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.first:
                    self.first = False
                    return SlowRecord()
                raise StopAsyncIteration

            async def aclose(self):
                nonlocal producer_closed
                close_started.set()
                await release_close.wait()
                producer_closed = True

        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail(f"unexpected request: {request.url}")

        task = asyncio.create_task(
            make_async_client(handler).upload_records(SlowClosingRecords(), compress=False)
        )
        await asyncio.to_thread(serialization_started.wait)

        task.cancel()
        release_serialization.set()
        await close_started.wait()

        # A second cancellation while producer cleanup is awaiting must not
        # detach the close task and return control with producer resources open.
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        assert not producer_closed

        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert producer_closed

    @pytest.mark.asyncio
    async def test_failed_upload_closes_the_async_record_producer(self, make_async_client):
        closed = False
        requests = []

        async def records():
            nonlocal closed
            try:
                for i in range(20):
                    yield {"id": i}
            finally:
                closed = True

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                400,
                json={"error": {"code": "invalid_trace", "message": "bad batch"}},
            )

        with pytest.raises(ValidationRejectedError):
            await make_async_client(handler).upload_records(
                records(), target_batch_bytes=40, compress=False
            )

        assert closed
        assert len(requests) == 1


class TestBatchCallback:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    async def test_on_batch_may_be_sync_or_a_coroutine_function(self, make_async_client, is_async):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json=COMMITTED)

        def sync_hook(batch, receipt):
            seen.append((batch.num_lines, receipt.status))

        async def async_hook(batch, receipt):
            await asyncio.sleep(0)
            seen.append((batch.num_lines, receipt.status))

        client = make_async_client(handler)
        await client.upload_lines(LINES, on_batch=async_hook if is_async else sync_hook)
        assert seen == [(2, "committed")]


class TestUploadRetries:
    @pytest.mark.asyncio
    async def test_retries_the_same_bytes_honoring_retry_after(self, make_async_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.headers["Idempotency-Key"])
            if len(attempts) == 1:
                return httpx.Response(
                    429,
                    headers={"Retry-After": "2"},
                    json={"error": {"code": "rate_limited", "message": "slow down"}},
                )
            return httpx.Response(201, json=COMMITTED)

        [receipt] = await make_async_client(handler).upload_lines(LINES)
        assert receipt.status == "committed"
        assert len(set(attempts)) == 1
        assert no_sleep == [2.0]

    @pytest.mark.asyncio
    async def test_retries_transport_failures_with_same_key(self, make_async_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.headers["Idempotency-Key"])
            if len(attempts) == 1:
                raise httpx.ConnectError("connection refused", request=request)
            if len(attempts) == 2:
                # Ambiguous: the server may have processed the request. Safe
                # only because the same key replays the committed receipt.
                raise httpx.ReadError("connection reset mid-response", request=request)
            return httpx.Response(201, json=COMMITTED)

        [receipt] = await make_async_client(handler).upload_lines(LINES)
        assert receipt.status == "committed"
        assert len(set(attempts)) == 1
        assert len(no_sleep) == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("failure", "expected"),
        [("response", RetryableAPIError), ("transport", TransportError)],
    )
    async def test_gives_up_after_max_attempts(
        self, make_async_client, no_sleep, failure, expected
    ):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request)
            if failure == "transport":
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(
                429,
                headers={"Retry-After": "0.1"},
                json={"error": {"code": "rate_limited", "message": "slow down"}},
            )

        with pytest.raises(expected) as exc_info:
            await make_async_client(handler).upload_lines(LINES, max_attempts=3)
        if failure == "response":
            assert exc_info.value.status_code == 429
            assert exc_info.value.code == "rate_limited"
        assert len(attempts) == 3
        assert len(no_sleep) == 2  # sleeps between attempts, not after the last

    @pytest.mark.asyncio
    async def test_non_positive_max_attempts_rejected(self, make_async_client):
        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail(f"unexpected request: {request.url}")

        with pytest.raises(ValueError, match="max_attempts"):
            await make_async_client(handler).upload_lines(LINES, max_attempts=0)
