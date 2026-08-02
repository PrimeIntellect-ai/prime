import hashlib

import pytest

from prime_traces import (
    MAX_BATCH_BYTES,
    TraceTooLargeError,
    iter_batches,
    read_jsonl_lines,
)


def line(payload: str) -> bytes:
    return payload.encode() + b"\n"


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


class TestLineValidation:
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


class TestFileReading:
    def test_round_trip_preserves_bytes(self, tmp_path):
        content = b'{"id":"a"}\n{"id":"b"}\n{"id":"c"}'
        path = tmp_path / "traces.jsonl"
        path.write_bytes(content)
        [batch] = iter_batches(read_jsonl_lines(path))
        assert batch.data == content
        assert batch.num_lines == 3
