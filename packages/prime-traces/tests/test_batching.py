import asyncio
import hashlib
import threading

import pytest

from prime_traces import (
    MAX_BATCH_BYTES,
    TraceTooLargeError,
    aiter_batches,
    iter_batches,
    read_jsonl_lines,
)


def line(payload: str) -> bytes:
    return payload.encode() + b"\n"


async def alines(values):
    for value in values:
        yield value


async def collect(source, **kwargs):
    return [batch async for batch in aiter_batches(source, **kwargs)]


class TestBatchIdentity:
    def test_digest_is_sha256_of_exact_bytes_including_newlines(self):
        lines = [line('{"id":"a"}'), line('{"id":"b"}')]
        [batch] = iter_batches(lines)
        expected = hashlib.sha256(b'{"id":"a"}\n{"id":"b"}\n').hexdigest()
        assert batch.digest == expected
        assert batch.idempotency_key == f"sha256:{expected}"
        assert batch.data == b'{"id":"a"}\n{"id":"b"}\n'

    def test_batching_is_deterministic_for_same_input(self):
        lines = [line(f'{{"id":"{i}","pad":"{"x" * 40}"}}') for i in range(50)]
        first = [(b.digest, b.num_lines) for b in iter_batches(lines, target_bytes=200)]
        second = [(b.digest, b.num_lines) for b in iter_batches(lines, target_bytes=200)]
        assert first == second
        assert len(first) > 1

    def test_final_line_without_terminator_is_preserved(self):
        lines = [line('{"id":"a"}'), b'{"id":"b"}']
        [batch] = iter_batches(lines)
        assert batch.data.endswith(b'{"id":"b"}')
        assert not batch.data.endswith(b"\n")


class TestChunkClosing:
    def test_closes_at_target_bytes(self):
        # 8-byte lines with a 20-byte target: two fit, the third overflows.
        lines = [line('{"i":1}'), line('{"i":2}'), line('{"i":3}')]
        batches = list(iter_batches(lines, target_bytes=20))
        assert [b.num_lines for b in batches] == [2, 1]
        assert batches[0].first_line_number == 1
        assert batches[1].first_line_number == 3

    def test_single_line_larger_than_target_forms_own_batch(self):
        big = line('{"pad":"' + "x" * 100 + '"}')
        batches = list(iter_batches([line('{"i":1}'), big], target_bytes=20))
        assert [b.num_lines for b in batches] == [1, 1]
        assert batches[1].size > 20

    def test_target_above_request_cap_rejected(self):
        with pytest.raises(ValueError):
            list(iter_batches([line("{}")], target_bytes=MAX_BATCH_BYTES + 1))

    def test_closes_at_max_lines(self):
        # Mirrors the service's per-request row cap: a large byte target over
        # tiny lines must not build a batch the service would reject with
        # too_many_traces_in_upload.
        lines = [line(f'{{"i":{i}}}') for i in range(5)]
        batches = list(iter_batches(lines, target_bytes=1024, max_lines=2))
        assert [b.num_lines for b in batches] == [2, 2, 1]

    def test_non_positive_max_lines_rejected(self):
        with pytest.raises(ValueError):
            list(iter_batches([line("{}")], max_lines=0))


class TestLineValidation:
    class NoStripBytes(bytes):
        def strip(self, *args, **kwargs):
            raise AssertionError("batching must not copy a line with strip()")

        def rstrip(self, *args, **kwargs):
            raise AssertionError("batching must not copy a line with rstrip()")

    def test_oversized_line_rejected_locally(self):
        big = line('{"pad":"' + "x" * 64 + '"}')
        with pytest.raises(TraceTooLargeError) as exc_info:
            list(iter_batches([line('{"i":1}'), big], max_line_bytes=32))
        assert exc_info.value.line_number == 2
        assert exc_info.value.limit == 32

    def test_line_cap_measured_without_terminator(self):
        # Content is exactly at the cap; the trailing newline must not tip it.
        content = b'{"p":"' + b"x" * 24 + b'"}'
        assert len(content) == 32
        [batch] = iter_batches([content + b"\n"], max_line_bytes=32)
        assert batch.num_lines == 1

    def test_line_cap_counts_carriage_return_like_the_server(self):
        # The server splits on LF alone, so a CRLF line keeps its \r and the
        # \r counts against the cap; the local check must reject exactly what
        # the server would.
        content = b'{"p":"' + b"x" * 23 + b'"}'
        assert len(content) == 31
        [batch] = iter_batches([content + b"\r\n"], max_line_bytes=32)  # 31 + \r = 32
        assert batch.num_lines == 1
        with pytest.raises(TraceTooLargeError):
            list(iter_batches([content + b"x" + b"\r\n"], max_line_bytes=32))  # 33

    def test_blank_lines_skipped(self):
        lines = [line('{"id":"a"}'), b"\n", b"   \n", line('{"id":"b"}')]
        [batch] = iter_batches(lines)
        assert batch.num_lines == 2
        assert batch.data == b'{"id":"a"}\n{"id":"b"}\n'

    def test_line_checks_do_not_strip_or_copy_input(self):
        record = self.NoStripBytes(b'{"id":"a"}\n')
        blank = self.NoStripBytes(b" \t\r\n")

        [batch] = iter_batches([record, blank], max_line_bytes=len(record) - 1)

        assert batch.data == record


class TestFileReading:
    def test_round_trip_preserves_bytes(self, tmp_path):
        content = b'{"id":"a"}\n{"id":"b"}\n{"id":"c"}'
        path = tmp_path / "traces.jsonl"
        path.write_bytes(content)
        [batch] = iter_batches(read_jsonl_lines(path))
        assert batch.data == content
        assert batch.num_lines == 3


class TestAsyncBatching:
    """`aiter_batches` must be a scheduling change, not a batching change.

    A producer that switches to the async client — or resumes an interrupted
    upload from the other one — has to reproduce the same upload IDs, so every
    closing rule is asserted to agree with `iter_batches` rather than merely
    to look reasonable.
    """

    LINES = [line(f'{{"id":"{i}","pad":"{"x" * 40}"}}') for i in range(50)]

    def identity(self, batches):
        return [(b.digest, b.num_lines, b.first_line_number) for b in batches]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("target_bytes", [200, 4096])
    async def test_async_source_batches_identically(self, target_bytes):
        expected = self.identity(iter_batches(self.LINES, target_bytes=target_bytes))
        actual = self.identity(await collect(alines(self.LINES), target_bytes=target_bytes))
        assert actual == expected

    @pytest.mark.asyncio
    async def test_sync_source_batches_identically(self):
        expected = self.identity(iter_batches(self.LINES, target_bytes=200))
        actual = self.identity(await collect(self.LINES, target_bytes=200))
        assert actual == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("as_async", [False, True])
    async def test_blank_lines_skipped_and_line_numbers_follow_the_source(self, as_async):
        lines = [b"\n", line('{"id":"a"}'), b"   \n", line('{"id":"b"}')]
        source = alines(lines) if as_async else lines
        [batch] = await collect(source, target_bytes=1024)
        assert batch.data == b'{"id":"a"}\n{"id":"b"}\n'
        # Numbering counts the blank lines it skipped, as the sync path does.
        assert batch.first_line_number == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("as_async", [False, True])
    async def test_closes_at_max_lines(self, as_async):
        lines = [line(f'{{"i":{i}}}') for i in range(5)]
        source = alines(lines) if as_async else lines
        batches = await collect(source, target_bytes=1024, max_lines=2)
        assert [b.num_lines for b in batches] == [2, 2, 1]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("as_async", [False, True])
    async def test_oversized_line_rejected_locally(self, as_async):
        lines = [line('{"i":1}'), line('{"pad":"' + "x" * 64 + '"}')]
        source = alines(lines) if as_async else lines
        with pytest.raises(TraceTooLargeError) as exc_info:
            await collect(source, max_line_bytes=32)
        assert exc_info.value.line_number == 2

    @pytest.mark.asyncio
    async def test_async_line_checks_do_not_strip_or_copy_input(self):
        guard = TestLineValidation.NoStripBytes
        record = guard(b'{"id":"a"}\n')
        source = alines([record, guard(b" \t\r\n")])

        [batch] = await collect(source, max_line_bytes=len(record) - 1)

        assert batch.data == record

    @pytest.mark.asyncio
    @pytest.mark.parametrize("as_async", [False, True])
    async def test_target_above_request_cap_rejected(self, as_async):
        source = alines([line("{}")]) if as_async else [line("{}")]
        with pytest.raises(ValueError):
            await collect(source, target_bytes=MAX_BATCH_BYTES + 1)

    @pytest.mark.asyncio
    async def test_abandoning_the_iterator_stops_reading_the_source(self):
        """A failed upload must not drain the rest of the file into memory."""
        consumed = 0

        def counted():
            nonlocal consumed
            for value in self.LINES:
                consumed += 1
                yield value

        batches = aiter_batches(counted(), target_bytes=200)
        await batches.__anext__()
        await batches.aclose()

        # Enough for the first batch and its lookahead, nowhere near the file.
        assert consumed < len(self.LINES)

    @pytest.mark.asyncio
    async def test_abandoning_the_iterator_closes_an_async_source(self):
        closed = False

        async def source():
            nonlocal closed
            try:
                for value in self.LINES:
                    yield value
            finally:
                closed = True

        batches = aiter_batches(source(), target_bytes=200)
        await batches.__anext__()
        await batches.aclose()

        assert closed

    @pytest.mark.asyncio
    async def test_cancellation_waits_for_sync_source_before_closing_iterator(self):
        """Cancellation must not close a generator still running in a worker."""
        started = threading.Event()
        release = threading.Event()
        closed = threading.Event()

        def slow_lines():
            try:
                started.set()
                release.wait()
                yield line("{}")
            finally:
                closed.set()

        batches = aiter_batches(slow_lines())
        task = asyncio.create_task(batches.__anext__())
        await asyncio.to_thread(started.wait)

        task.cancel()
        # Let cancellation enter aiter_batches while next_batch still owns the
        # synchronous generator, reproducing the old close-while-running race.
        await asyncio.sleep(0)
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
