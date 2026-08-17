"""Async read, delete and episode paths.

The pinned response shapes and the reserved-ID cases are imported from
``test_reads`` rather than restated: one definition of the service contract,
asserted against both clients. Referencing the module (not the test classes)
keeps the sync suite from being collected twice.
"""

import asyncio
import threading

import httpx
import pytest
import test_reads

import prime_traces.async_traces as async_traces_module
from prime_traces import (
    AmbiguousDeleteError,
    APIError,
    AsyncTracesAPIClient,
    AsyncTracesClient,
    ForbiddenError,
    NotFoundError,
    RetryableAPIError,
    TransportError,
    UnauthorizedError,
)

SUMMARY = test_reads.SUMMARY
RESERVED_TRACE_ID = test_reads.RESERVED_TRACE_ID
ENCODED_TRACE_PATH = test_reads.ENCODED_TRACE_PATH
RESERVED_EPISODE_ID = test_reads.RESERVED_EPISODE_ID
ENCODED_EPISODE_PATH = test_reads.ENCODED_EPISODE_PATH
EPISODE = test_reads.TestEpisodes.EPISODE
EMPTY_AGGREGATE = test_reads.TestEpisodes.EMPTY_AGGREGATE

UNAVAILABLE = {"error": {"code": "storage_unavailable", "message": "try again"}}


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
    async def test_get_summary(self, make_async_client):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/traces/8d3f1a2b"
            return httpx.Response(200, json=SUMMARY)

        summary = await make_async_client(handler).get("8d3f1a2b")
        assert summary.task_id == "tb2-0187"

    @pytest.mark.asyncio
    async def test_get_encodes_trace_id_as_one_path_segment(self, make_async_client):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.raw_path == ENCODED_TRACE_PATH
            return httpx.Response(200, json=SUMMARY)

        await make_async_client(handler).get(RESERVED_TRACE_ID)

    @pytest.mark.asyncio
    async def test_get_rejects_trace_id_with_slash_before_request(self, make_async_client):
        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail(f"unexpected request: {request.url}")

        with pytest.raises(ValueError, match="cannot contain '/'"):
            await make_async_client(handler).get("trace/child")

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
    async def test_partial_is_cleaned_up_when_the_stream_breaks_midway(
        self, make_async_client, tmp_path
    ):
        def handler(request: httpx.Request) -> httpx.Response:
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

    @pytest.mark.asyncio
    async def test_stream_failure_unlinks_partial_when_close_also_fails(
        self, make_async_client, monkeypatch, tmp_path
    ):
        partial = tmp_path / ".prime-traces-controlled.partial"
        partial.touch()

        class FailingCloseHandle:
            name = str(partial)

            def write(self, chunk):
                return len(chunk)

            def close(self):
                raise OSError("flush failed")

        async def stream_bytes(*args, **kwargs):
            yield b'{"version":4,'
            raise RuntimeError("stream failed")

        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail(f"unexpected request: {request.url}")

        client = make_async_client(handler)
        monkeypatch.setattr(client.client, "stream_bytes", stream_bytes)
        monkeypatch.setattr(
            async_traces_module, "_open_partial_file", lambda dest: FailingCloseHandle()
        )

        dest = tmp_path / "trace.json"
        with pytest.raises(RuntimeError, match="stream failed"):
            await client.download_raw("8d3f1a2b", dest)

        assert not partial.exists()
        assert not dest.exists()

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
    async def test_repeated_cancellation_waits_for_http_response_close(
        self, make_async_client, monkeypatch, tmp_path
    ):
        write_started = threading.Event()
        release_write = threading.Event()
        close_started = asyncio.Event()
        release_close = asyncio.Event()
        stream_closed = asyncio.Event()

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
                pass

        class SlowClosingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"version":4,"id":"8d3f1a2b"}'

            async def aclose(self):
                close_started.set()
                await release_close.wait()
                stream_closed.set()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=SlowClosingStream())

        client = make_async_client(handler)
        monkeypatch.setattr(async_traces_module, "_open_partial_file", lambda dest: SlowHandle())

        dest = tmp_path / "trace.json"
        task = asyncio.create_task(client.download_raw("8d3f1a2b", dest))
        await asyncio.to_thread(write_started.wait)

        task.cancel()
        release_write.set()
        await close_started.wait()

        # A caller may repeat cancellation while cleanup is in progress. The
        # response close must finish before cancellation reaches the caller.
        task.cancel()
        await asyncio.sleep(0.01)
        pending_while_closing = not task.done()
        release_close.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert pending_while_closing
        assert stream_closed.is_set()
        assert not dest.exists()
        assert list(tmp_path.glob(".prime-traces-*")) == []

    @pytest.mark.asyncio
    async def test_cancellation_during_close_waits_for_worker(
        self, make_async_client, monkeypatch, tmp_path
    ):
        close_started = threading.Event()
        release_close = threading.Event()
        close_calls = 0

        partial = tmp_path / ".prime-traces-controlled.partial"
        partial.touch()

        class SlowCloseHandle:
            name = str(partial)

            def write(self, chunk):
                return len(chunk)

            def close(self):
                nonlocal close_calls
                close_calls += 1
                if close_calls == 1:
                    close_started.set()
                    if not release_close.wait(timeout=5):
                        raise TimeoutError("test did not release file close")

        async def stream_bytes(*args, **kwargs):
            yield b'{"version":4,"id":"8d3f1a2b"}'

        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail(f"unexpected request: {request.url}")

        client = make_async_client(handler)
        monkeypatch.setattr(client.client, "stream_bytes", stream_bytes)
        monkeypatch.setattr(
            async_traces_module, "_open_partial_file", lambda dest: SlowCloseHandle()
        )

        dest = tmp_path / "trace.json"
        task = asyncio.create_task(client.download_raw("8d3f1a2b", dest))
        await asyncio.to_thread(close_started.wait)
        task.cancel()
        await asyncio.sleep(0.01)

        pending_while_close = not task.done()
        release_close.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert pending_while_close
        assert close_calls == 2
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

    @pytest.mark.asyncio
    async def test_cancelled_partial_open_unlinks_file_when_close_fails(
        self, make_async_client, monkeypatch, tmp_path
    ):
        started = threading.Event()
        release = threading.Event()
        partial = tmp_path / ".prime-traces-controlled.partial"

        class FailingCloseHandle:
            name = str(partial)

            def close(self):
                raise OSError("flush failed")

        def delayed_open(dest):
            started.set()
            release.wait()
            partial.touch()
            return FailingCloseHandle()

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
        assert not partial.exists()
        assert not dest.exists()

    @pytest.mark.asyncio
    async def test_final_replace_is_an_uncancellable_commit(
        self, make_async_client, monkeypatch, tmp_path
    ):
        loop_thread = threading.get_ident()
        replace_thread = None
        original_replace = async_traces_module.Path.replace

        def observed_replace(path, target):
            nonlocal replace_thread
            replace_thread = threading.get_ident()
            return original_replace(path, target)

        monkeypatch.setattr(async_traces_module.Path, "replace", observed_replace)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"complete")

        dest = tmp_path / "trace.json"
        await make_async_client(handler).download_raw("8d3f1a2b", dest)

        assert replace_thread == loop_thread
        assert dest.read_bytes() == b"complete"

    @pytest.mark.parametrize(
        ("status", "code", "expected"),
        [
            (404, "trace_not_found", NotFoundError),
            (401, None, UnauthorizedError),
            (403, "service_not_enabled", ForbiddenError),
        ],
    )
    @pytest.mark.asyncio
    async def test_errors_are_typed(self, make_async_client, status, code, expected):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={"error": {"code": code, "message": "nope"}})

        with pytest.raises(expected) as caught:
            await make_async_client(handler).get("8d3f1a2b")
        assert caught.value.status_code == status
        assert caught.value.code == code


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

    @pytest.mark.asyncio
    async def test_stream_failure_mid_body_is_not_retried(
        self, make_async_client, no_sleep, tmp_path
    ):
        attempts = []

        async def broken_body():
            yield b'{"version":4,'
            raise httpx.ReadError("connection reset mid-body")

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            return httpx.Response(200, content=broken_body())

        with pytest.raises(TransportError):
            await make_async_client(handler).download_raw("8d3f1a2b", tmp_path / "trace.json")
        assert len(attempts) == 1
        assert no_sleep == []


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
    async def test_delete_run_sends_run_id_and_expects_no_body(self, make_async_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["params"] = dict(request.url.params)
            return httpx.Response(202)

        assert await make_async_client(handler).delete_run("run_9f3k2m") is None
        assert captured["path"] == "/api/v1/traces"
        assert captured["params"] == {"run_id": "run_9f3k2m"}

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
    async def test_unknown_code_503_delete_is_not_retried(self, make_async_client, no_sleep):
        """A codeless 503 may have come from an intermediary after the request
        was forwarded, so the deletion may already have landed."""
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            return httpx.Response(503, text="no healthy upstream")

        with pytest.raises(AmbiguousDeleteError):
            await make_async_client(handler).delete("8d3f1a2b")
        assert len(attempts) == 1

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
    async def test_list_episodes_filters_and_envelope(self, make_async_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["params"] = dict(request.url.params)
            return httpx.Response(200, json={"items": [EPISODE], "next_cursor": None})

        page = await make_async_client(handler).list_episodes(
            run_id="run_9f3k2m", outcome="done", has_error=False
        )
        assert captured["params"] == {
            "run_id": "run_9f3k2m",
            "outcome": "done",
            "has_error": "false",
        }
        [episode] = page.items
        assert episode.episode_id == "ep-1"
        assert episode.error.type is None

    @pytest.mark.asyncio
    async def test_get_episode_nests_member_aggregate(self, make_async_client):
        detail = {**EPISODE, "traces": {**EMPTY_AGGREGATE, "trace_count": 2}}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/episodes/ep-1"
            return httpx.Response(200, json=detail)

        episode = await make_async_client(handler).get_episode("ep-1")
        assert episode.traces.trace_count == 2

    @pytest.mark.asyncio
    async def test_get_episode_encodes_episode_id_as_one_path_segment(self, make_async_client):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.raw_path == ENCODED_EPISODE_PATH
            return httpx.Response(
                200, json={**EPISODE, "episode_id": RESERVED_EPISODE_ID, "traces": EMPTY_AGGREGATE}
            )

        await make_async_client(handler).get_episode(RESERVED_EPISODE_ID)

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
    async def test_team_header_is_sent_when_configured(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["team"] = request.headers.get("X-Prime-Team-ID")
            return httpx.Response(200, json=SUMMARY)

        api_client = AsyncTracesAPIClient(
            api_key="test-key",
            base_url="http://testserver",
            team_id="team_42",
            transport=httpx.MockTransport(handler),
        )
        await api_client.get_json("/traces/8d3f1a2b")
        assert captured["team"] == "team_42"
        await api_client.aclose()

    @pytest.mark.asyncio
    async def test_missing_api_key_fails_loudly_at_request_time(self):
        api_client = AsyncTracesAPIClient(api_key="", base_url="http://testserver", team_id="")
        with pytest.raises(APIError, match="No API key configured"):
            await api_client.get_json("/traces")
        await api_client.aclose()
