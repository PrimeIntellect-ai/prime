"""Async read, delete and episode paths.

Shared response samples live in ``_samples`` so the test modules do not import
one another.
"""

import asyncio
import threading

import httpx
import pytest
from _samples import (
    SUMMARY,
    UNAVAILABLE,
)

import prime_traces.async_traces as async_traces_module
from prime_traces import (
    AmbiguousDeleteError,
    AsyncTracesAPIClient,
    AsyncTracesClient,
    NotFoundError,
    RetryableAPIError,
    TransportError,
)


class TestList:
    @pytest.mark.asyncio
    async def test_filters_encode_as_documented_query_params(self, make_async_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["params"] = dict(request.url.params)
            return httpx.Response(200, json={"items": [SUMMARY], "next_cursor": None})

        page = await make_async_client(handler).list(
            run_id="run_9f3k2m",
            environment_id="terminal-bench-2",
            reward_min=0.5,
            has_error=False,
            context={"source": "hosted_eval"},
            limit=50,
        )

        assert captured["params"] == {
            "run_id": "run_9f3k2m",
            "environment_id": "terminal-bench-2",
            "reward_min": "0.5",
            "has_error": "false",
            "context.source": "hosted_eval",
            "limit": "50",
        }
        [summary] = page.items
        assert summary.trace_id == "8d3f1a2b"
        assert summary.score.reward == 0.85
        assert summary.model.id == "deepseek-v4-flash"

    @pytest.mark.asyncio
    async def test_iter_follows_cursor(self, make_async_client):
        pages = {
            None: {"items": [{**SUMMARY, "trace_id": "t1"}], "next_cursor": "c1"},
            "c1": {"items": [{**SUMMARY, "trace_id": "t2"}], "next_cursor": None},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=pages[request.url.params.get("cursor")])

        client = make_async_client(handler)
        ids = [summary.trace_id async for summary in client.iter(run_id="run_9f3k2m")]
        assert ids == ["t1", "t2"]

    @pytest.mark.asyncio
    async def test_iter_stops_reading_when_the_consumer_breaks(self, make_async_client):
        """Abandoning the generator must not keep paging in the background."""
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request.url.params.get("cursor"))
            return httpx.Response(
                200, json={"items": [SUMMARY], "next_cursor": f"c{len(requests)}"}
            )

        client = make_async_client(handler)
        pages = client.iter()
        try:
            async for _ in pages:
                break
        finally:
            await pages.aclose()
        assert len(requests) == 1


class TestPointReads:
    @pytest.mark.asyncio
    async def test_get_raw_streams_document(self, make_async_client):
        raw = b'{"version":4,"id":"8d3f1a2b","steps":[]}'

        def handler(request: httpx.Request) -> httpx.Response:
            assert dict(request.url.params) == {"raw": "true"}
            return httpx.Response(200, content=raw)

        assert await make_async_client(handler).get_raw("8d3f1a2b") == raw

    @pytest.mark.asyncio
    async def test_download_raw_writes_file(self, make_async_client, tmp_path):
        raw = b'{"version":4,"id":"8d3f1a2b"}'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=raw)

        dest = tmp_path / "trace.json"
        written = await make_async_client(handler).download_raw("8d3f1a2b", dest)
        assert written == len(raw)
        assert dest.read_bytes() == raw
        assert list(tmp_path.glob(".prime-traces-*")) == []

    @pytest.mark.asyncio
    async def test_download_raw_failure_preserves_existing_file(self, make_async_client, tmp_path):
        dest = tmp_path / "trace.json"
        dest.write_bytes(b"previous good document")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404, json={"error": {"code": "trace_not_found", "message": "gone"}}
            )

        with pytest.raises(NotFoundError):
            await make_async_client(handler).download_raw("8d3f1a2b", dest)
        assert dest.read_bytes() == b"previous good document"
        assert list(tmp_path.glob(".prime-traces-*")) == []

    @pytest.mark.asyncio
    async def test_midstream_failure_is_not_retried_and_cleans_up(
        self, make_async_client, no_sleep, tmp_path
    ):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request)
            # An async body: httpx.AsyncClient refuses to stream a sync one.
            async def body():
                yield b'{"version":4,'
                raise httpx.ReadError("connection reset")

            return httpx.Response(200, content=body())

        dest = tmp_path / "trace.json"
        with pytest.raises(TransportError):
            await make_async_client(handler).download_raw("8d3f1a2b", dest)
        # A mid-stream failure is never retried under the consumer, so the
        # prefix that did arrive is discarded rather than left as a document.
        assert len(attempts) == 1
        assert no_sleep == []
        assert not dest.exists()
        assert list(tmp_path.glob(".prime-traces-*")) == []

    @pytest.mark.asyncio
    async def test_local_write_failure_closes_response_stream(
        self, make_async_client, monkeypatch, tmp_path
    ):
        stream_closed = asyncio.Event()

        async def stream_bytes(*args, **kwargs):
            try:
                yield b'{"version":4,'
                yield b'"id":"8d3f1a2b"}'
            finally:
                stream_closed.set()

        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail(f"unexpected request: {request.url}")

        client = make_async_client(handler)
        monkeypatch.setattr(client.client, "stream_bytes", stream_bytes)
        original_to_thread = asyncio.to_thread

        async def fail_file_write(function, *args, **kwargs):
            if getattr(function, "__name__", None) == "write":
                raise OSError("disk full")
            return await original_to_thread(function, *args, **kwargs)

        monkeypatch.setattr(async_traces_module.asyncio, "to_thread", fail_file_write)

        dest = tmp_path / "trace.json"
        with pytest.raises(OSError, match="disk full"):
            await client.download_raw("8d3f1a2b", dest)

        assert stream_closed.is_set()
        assert not dest.exists()
        assert list(tmp_path.glob(".prime-traces-*")) == []

    def test_discard_partial_unlinks_even_when_close_fails(self, tmp_path):
        partial = tmp_path / ".prime-traces-controlled.partial"
        partial.touch()

        class FailingCloseHandle:
            def close(self):
                raise OSError("flush failed")

        with pytest.raises(OSError, match="flush failed"):
            async_traces_module._discard_partial_file(FailingCloseHandle(), partial)
        assert not partial.exists()

    @pytest.mark.asyncio
    async def test_cancellation_during_write_closes_response_stream(
        self, make_async_client, monkeypatch, tmp_path
    ):
        stream_closed = asyncio.Event()
        write_started = threading.Event()
        release_write = threading.Event()
        close_called = threading.Event()

        partial = tmp_path / ".prime-traces-controlled.partial"
        partial.touch()

        class SlowHandle:
            name = str(partial)

            def write(self, chunk):
                write_started.set()
                if not release_write.wait(timeout=5):
                    raise TimeoutError("test did not release file write")
                return len(chunk)

            def close(self):
                close_called.set()

        async def stream_bytes(*args, **kwargs):
            try:
                yield b'{"version":4,"id":"8d3f1a2b"}'
            finally:
                stream_closed.set()

        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail(f"unexpected request: {request.url}")

        client = make_async_client(handler)
        monkeypatch.setattr(client.client, "stream_bytes", stream_bytes)
        monkeypatch.setattr(async_traces_module, "_open_partial_file", lambda dest: SlowHandle())

        dest = tmp_path / "trace.json"
        task = asyncio.create_task(client.download_raw("8d3f1a2b", dest))
        await asyncio.to_thread(write_started.wait)
        task.cancel()
        # Give cancellation time to reach the shielded write. The event loop
        # must remain responsive while the real worker thread stays blocked.
        await asyncio.sleep(0.01)

        pending_while_write = not task.done()
        closed_before_write_finished = close_called.is_set()
        release_write.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert pending_while_write
        assert not closed_before_write_finished
        assert stream_closed.is_set()
        assert not dest.exists()
        assert list(tmp_path.glob(".prime-traces-*")) == []

    @pytest.mark.asyncio
    async def test_cancellation_during_partial_open_cleans_up_file(
        self, make_async_client, monkeypatch, tmp_path
    ):
        started = threading.Event()
        release = threading.Event()
        opened = threading.Event()
        original_open = async_traces_module._open_partial_file

        def delayed_open(dest):
            started.set()
            release.wait()
            handle = original_open(dest)
            opened.set()
            return handle

        monkeypatch.setattr(async_traces_module, "_open_partial_file", delayed_open)

        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail(f"unexpected request: {request.url}")

        dest = tmp_path / "trace.json"
        task = asyncio.create_task(make_async_client(handler).download_raw("8d3f1a2b", dest))
        await asyncio.to_thread(started.wait)

        task.cancel()
        await asyncio.sleep(0)
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.to_thread(opened.wait)
        assert not dest.exists()
        assert list(tmp_path.glob(".prime-traces-*")) == []

class TestReadRetries:
    @pytest.mark.asyncio
    async def test_get_retries_transient_503_honoring_retry_after(
        self, make_async_client, no_sleep
    ):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            if len(attempts) == 1:
                return httpx.Response(503, headers={"Retry-After": "3"}, json=UNAVAILABLE)
            return httpx.Response(200, json=SUMMARY)

        summary = await make_async_client(handler).get("8d3f1a2b")
        assert summary.trace_id == "8d3f1a2b"
        assert len(attempts) == 2
        assert no_sleep == [3.0]

    @pytest.mark.asyncio
    async def test_get_gives_up_after_bounded_attempts(self, make_async_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            return httpx.Response(503, json=UNAVAILABLE)

        with pytest.raises(RetryableAPIError):
            await make_async_client(handler).get("8d3f1a2b")
        assert len(attempts) == 3
        assert len(no_sleep) == 2

    @pytest.mark.asyncio
    async def test_get_does_not_retry_404(self, make_async_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            return httpx.Response(
                404, json={"error": {"code": "trace_not_found", "message": "gone"}}
            )

        with pytest.raises(NotFoundError):
            await make_async_client(handler).get("8d3f1a2b")
        assert len(attempts) == 1
        assert no_sleep == []

    @pytest.mark.asyncio
    async def test_get_retries_transport_failures(self, make_async_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            if len(attempts) == 1:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, json=SUMMARY)

        await make_async_client(handler).get("8d3f1a2b")
        assert len(attempts) == 2
        assert len(no_sleep) == 1

    @pytest.mark.asyncio
    async def test_stream_retries_before_first_byte(self, make_async_client, no_sleep, tmp_path):
        raw = b'{"version":4,"id":"8d3f1a2b"}'
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            if len(attempts) == 1:
                return httpx.Response(503, json=UNAVAILABLE)
            return httpx.Response(200, content=raw)

        dest = tmp_path / "trace.json"
        written = await make_async_client(handler).download_raw("8d3f1a2b", dest)
        assert written == len(raw)
        assert dest.read_bytes() == raw
        assert len(attempts) == 2

class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_trace_with_created_at_hint(self, make_async_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["path"] = request.url.path
            captured["params"] = dict(request.url.params)
            return httpx.Response(202)

        await make_async_client(handler).delete("8d3f1a2b", created_at="2026-07-20T18:02:11.482Z")
        assert captured["method"] == "DELETE"
        assert captured["path"] == "/api/v1/traces/8d3f1a2b"
        assert captured["params"] == {"created_at": "2026-07-20T18:02:11.482Z"}

    @pytest.mark.asyncio
    async def test_ambiguous_transport_failure_is_not_retried(self, make_async_client, no_sleep):
        """A replay could delete a trace written after the first attempt."""
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            raise httpx.ReadError("connection reset", request=request)

        with pytest.raises(AmbiguousDeleteError) as caught:
            await make_async_client(handler).delete("8d3f1a2b")
        assert caught.value.status_code is None
        assert len(attempts) == 1
        assert no_sleep == []

    @pytest.mark.parametrize("status", [502, 504])
    @pytest.mark.asyncio
    async def test_ambiguous_gateway_failure_is_not_retried(
        self, make_async_client, no_sleep, status
    ):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            return httpx.Response(status, text="upstream connect error")

        with pytest.raises(AmbiguousDeleteError) as caught:
            await make_async_client(handler).delete("8d3f1a2b")
        assert caught.value.status_code == status
        assert len(attempts) == 1
        assert no_sleep == []

    @pytest.mark.asyncio
    async def test_service_refusal_503_delete_is_retried(self, make_async_client, no_sleep):
        """A service code proves the service declined it — nothing was deleted."""
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            if len(attempts) == 1:
                return httpx.Response(503, json=UNAVAILABLE)
            return httpx.Response(202)

        await make_async_client(handler).delete("8d3f1a2b")
        assert len(attempts) == 2
        assert len(no_sleep) == 1


class TestEpisodes:
    @pytest.mark.asyncio
    async def test_list_episode_traces_forwards_backend_filters(self, make_async_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["params"] = dict(request.url.params)
            return httpx.Response(200, json={"items": [SUMMARY], "next_cursor": None})

        page = await make_async_client(handler).list_episode_traces(
            "ep-1", has_error=True, context={"source": "hosted_eval"}
        )
        assert captured["path"] == "/api/v1/episodes/ep-1/traces"
        assert captured["params"] == {"has_error": "true", "context.source": "hosted_eval"}
        assert page.items[0].trace_id == "8d3f1a2b"


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_async_context_manager_closes_the_transport(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=SUMMARY)

        api_client = AsyncTracesAPIClient(
            api_key="test-key",
            base_url="http://testserver",
            team_id="",
            transport=httpx.MockTransport(handler),
        )
        async with AsyncTracesClient(api_client=api_client) as client:
            await client.get("8d3f1a2b")
        assert api_client.client.is_closed

    @pytest.mark.asyncio
    async def test_repeated_cancellation_waits_for_transport_close(self):
        close_started = asyncio.Event()
        release_close = asyncio.Event()
        transport_closed = False

        class SlowClosingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json=SUMMARY)

            async def aclose(self) -> None:
                nonlocal transport_closed
                close_started.set()
                await release_close.wait()
                transport_closed = True

        api_client = AsyncTracesAPIClient(
            api_key="test-key",
            base_url="http://testserver",
            team_id="",
            transport=SlowClosingTransport(),
        )
        context_entered = asyncio.Event()

        async def use_client():
            async with api_client:
                context_entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(use_client())
        await context_entered.wait()

        # The first cancellation exits the context; the second arrives while
        # the transport is still releasing pooled resources.
        task.cancel()
        await close_started.wait()
        task.cancel()
        await asyncio.sleep(0.01)

        assert not task.done()
        assert not transport_closed

        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert transport_closed
