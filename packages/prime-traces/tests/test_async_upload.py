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
    ErrorCode,
    LineFormat,
    RetryableAPIError,
    TransportError,
    ValidationRejectedError,
    iter_batches,
)

RAW = b'{"id":"a"}\n{"id":"b"}\n'
LINES = [b'{"id":"a"}\n', b'{"id":"b"}\n']
COMMITTED = {"upload_id": "x" * 64, "status": "committed"}


async def alines(values):
    for value in values:
        yield value


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
    async def test_episode_format_sets_the_header(self, make_async_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            return httpx.Response(201, json=COMMITTED)

        await make_async_client(handler).upload_lines(LINES, line_format=LineFormat.EPISODE)
        assert captured["request"].headers["X-Prime-Line-Format"] == "episode"

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
        assert keys == [batch.idempotency_key for batch in iter_batches([RAW])]


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
    async def test_async_lines_batch_identically_to_sync_lines(self, make_async_client):
        many = [b'{"i":%d}\n' % i for i in range(40)]
        keys = []

        def handler(request: httpx.Request) -> httpx.Response:
            keys.append(request.headers["Idempotency-Key"])
            return httpx.Response(201, json=COMMITTED)

        client = make_async_client(handler)
        receipts = await client.upload_lines(alines(many), target_batch_bytes=40, compress=False)

        expected = [batch.idempotency_key for batch in iter_batches(many, target_bytes=40)]
        assert len(expected) > 1  # the split is what makes this worth asserting
        assert keys == expected
        assert len(receipts) == len(expected)

    @pytest.mark.asyncio
    async def test_async_record_rejection_happens_before_any_request(self, make_async_client):
        async def records():
            yield {"id": "a"}
            yield object()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json=COMMITTED)

        with pytest.raises(TypeError, match="Record 2 must be a mapping"):
            await make_async_client(handler).upload_records(records())

    @pytest.mark.asyncio
    async def test_sync_source_is_consumed_off_the_event_loop(self, make_async_client):
        """A synchronous producer blocks its thread, not the loop.

        This is the whole reason to reach for the async client: reading and
        hashing a file must not stall the tasks running alongside it.
        """
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.001)

        def slow_lines():
            for i in range(3):
                # Not time.sleep: the no_sleep fixture patches that away.
                # Event.wait blocks whichever thread reaches it, which is
                # exactly the property under test.
                threading.Event().wait(0.05)
                yield b'{"i":%d}\n' % i

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json=COMMITTED)

        background = asyncio.create_task(ticker())
        try:
            await make_async_client(handler).upload_lines(slow_lines(), compress=False)
        finally:
            background.cancel()

        # ~150 ms of blocking work: on the loop the ticker would be frozen for
        # all of it. A loose floor keeps this from turning into a timing test.
        assert ticks > 10


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
    async def test_gives_up_after_max_attempts(self, make_async_client, no_sleep):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={"Retry-After": "0.1"},
                json={"error": {"code": "rate_limited", "message": "slow down"}},
            )

        with pytest.raises(RetryableAPIError) as exc_info:
            await make_async_client(handler).upload_lines(LINES, max_attempts=3)
        assert exc_info.value.status_code == 429
        assert exc_info.value.code == "rate_limited"
        assert len(no_sleep) == 2  # sleeps between attempts, not after the last

    @pytest.mark.asyncio
    async def test_transport_failure_exhausts_attempts(self, make_async_client, no_sleep):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with pytest.raises(TransportError):
            await make_async_client(handler).upload_lines(LINES, max_attempts=2)
        assert len(no_sleep) == 1

    @pytest.mark.asyncio
    async def test_non_positive_max_attempts_rejected(self, make_async_client):
        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail(f"unexpected request: {request.url}")

        with pytest.raises(ValueError, match="max_attempts"):
            await make_async_client(handler).upload_lines(LINES, max_attempts=0)

    @pytest.mark.asyncio
    async def test_durable_rejection_stops_the_upload(self, make_async_client):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                400,
                json={"error": {"code": "invalid_trace", "message": "line 7: not a trace"}},
            )

        many = [b'{"i":%d}\n' % i for i in range(40)]
        with pytest.raises(ValidationRejectedError) as exc_info:
            await make_async_client(handler).upload_lines(many, target_batch_bytes=40)
        assert ErrorCode(exc_info.value.code) is ErrorCode.INVALID_TRACE
        # The first rejection stops the run rather than pushing the rest.
        assert len(requests) == 1
