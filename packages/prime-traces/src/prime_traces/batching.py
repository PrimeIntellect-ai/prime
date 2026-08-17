"""Client-side batching for content-addressed uploads.

The contract this implements (see the Prime Traces design docs):

- The SDK reads a completed JSONL file line by line; lines are opaque bytes.
  It never parses trace JSON, so it needs no dependency on verifiers.
- A request chunk closes at its uncompressed-byte threshold. Boundaries are
  measured in uncompressed bytes because compression ratios vary widely.
- A line larger than the single-line limit is rejected locally rather than
  split: under bare-trace format the line is one trace, under episode format
  one complete episode, and neither may span requests.
- upload_id = SHA256(exact uncompressed JSONL bytes, including newlines),
  sent as ``Idempotency-Key: sha256:<64 lowercase hex>``. There is no client
  upload ID or checkpoint: a crashed producer re-reads the file, rebuilds the
  same bytes, and regenerates the same keys — so batching must be
  deterministic for a given input.
"""

import asyncio
import hashlib
import threading
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Generator, Iterable, Iterator, Union

from .exceptions import TraceTooLargeError

MIB = 1024 * 1024

#: Server-side cap on uncompressed bytes per request (-> upload_too_large).
MAX_BATCH_BYTES = 256 * MIB
#: Server-side cap on one JSONL line (-> trace_too_large).
MAX_LINE_BYTES = 64 * MIB
#: Server-side cap on staged rows per request (-> too_many_traces_in_upload).
#: Exact for bare-trace uploads, where one line is one row. Episode lines
#: stage one episode row plus each nested trace, which opaque line bytes
#: cannot count — there the byte caps are the practical bound.
MAX_BATCH_LINES = 1_000_000
#: Default chunk-close threshold. Held under Cloud Run's 32 MiB request cap so
#: a batch fits the deployed transport even with compression disabled; the
#: 256 MiB service contract above is not reachable through that cap today. A
#: batch may exceed this only when a single line does — such a batch (or any
#: caller-raised target) still risks transport-level rejection before the
#: service sees it.
DEFAULT_TARGET_BATCH_BYTES = 30 * MIB


@dataclass(frozen=True)
class Batch:
    """One upload request's exact bytes and content-addressed identity."""

    data: bytes
    digest: str  # 64 lowercase hex chars, SHA-256 of ``data``
    num_lines: int
    first_line_number: int  # 1-based line number in the source, for reporting

    @property
    def idempotency_key(self) -> str:
        return f"sha256:{self.digest}"

    @property
    def size(self) -> int:
        return len(self.data)


def read_jsonl_lines(path: Union[str, Path]) -> Iterator[bytes]:
    """Yield raw lines from a JSONL file with their terminators preserved.

    Bytes are never rewritten — "bare-trace lines stay byte-identical" — so
    the same file always reproduces the same batches and digests.
    """
    with open(path, "rb") as f:
        yield from f


class _BatchBuilder:
    """Accumulation state behind ``iter_batches`` and ``aiter_batches``.

    Both entry points must produce identical batches — and therefore identical
    idempotency keys — for the same lines, so the closing rules live here once
    rather than being restated per entry point.
    """

    def __init__(self, *, target_bytes: int, max_line_bytes: int, max_lines: int) -> None:
        if target_bytes <= 0:
            raise ValueError("target_bytes must be positive")
        if target_bytes > MAX_BATCH_BYTES:
            raise ValueError(
                f"target_bytes ({target_bytes}) exceeds the {MAX_BATCH_BYTES} byte request cap"
            )
        if max_lines <= 0:
            raise ValueError("max_lines must be positive")
        self._target_bytes = target_bytes
        self._max_line_bytes = max_line_bytes
        self._max_lines = max_lines
        self._buffer: list[bytes] = []
        self._buffered_size = 0
        self._batch_start_line = 0

    @property
    def pending(self) -> bool:
        """Whether any line is buffered and awaiting a final ``close()``."""
        return bool(self._buffer)

    def closes_before(self, line_number: int, line: bytes) -> bool:
        """Validate ``line``, then report whether it must start a new batch."""
        content_size = len(line.rstrip(b"\n"))
        if content_size > self._max_line_bytes:
            raise TraceTooLargeError(line_number, content_size, self._max_line_bytes)
        return bool(self._buffer) and (
            self._buffered_size + len(line) > self._target_bytes
            or len(self._buffer) >= self._max_lines
        )

    def add(self, line_number: int, line: bytes) -> None:
        if not self._buffer:
            self._batch_start_line = line_number
        self._buffer.append(line)
        self._buffered_size += len(line)

    def close(self) -> Batch:
        """Seal the buffered lines into a batch and reset.

        This is where the joining and hashing happen, so it is the one call
        worth handing to a worker thread on the async path.
        """
        data = b"".join(self._buffer)
        batch = Batch(
            data=data,
            digest=hashlib.sha256(data).hexdigest(),
            num_lines=len(self._buffer),
            first_line_number=self._batch_start_line,
        )
        self._buffer = []
        self._buffered_size = 0
        return batch


def iter_batches(
    lines: Iterable[bytes],
    *,
    target_bytes: int = DEFAULT_TARGET_BATCH_BYTES,
    max_line_bytes: int = MAX_LINE_BYTES,
    max_lines: int = MAX_BATCH_LINES,
) -> Generator[Batch, None, None]:
    """Group JSONL lines into content-addressed request batches.

    Whitespace-only lines are skipped. Each kept line contributes its exact
    bytes (terminator included; a final line without one stays without one).
    The line-size cap is checked on the line content excluding the trailing
    newline but including any carriage return before it — the service splits
    on LF alone, so these are exactly the bytes it measures.

    A batch also closes at ``max_lines``, mirroring the service's per-request
    row cap: without it, a caller-raised ``target_bytes`` over a file of tiny
    lines could build a batch the service rejects with
    ``too_many_traces_in_upload`` — a 400 the retry semantics treat as
    "correct the file" when the file is fine and only the chunking is not.
    """
    builder = _BatchBuilder(
        target_bytes=target_bytes, max_line_bytes=max_line_bytes, max_lines=max_lines
    )
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if builder.closes_before(line_number, line):
            yield builder.close()
        builder.add(line_number, line)

    if builder.pending:
        yield builder.close()


async def aiter_batches(
    lines: Union[Iterable[bytes], AsyncIterable[bytes]],
    *,
    target_bytes: int = DEFAULT_TARGET_BATCH_BYTES,
    max_line_bytes: int = MAX_LINE_BYTES,
    max_lines: int = MAX_BATCH_LINES,
) -> AsyncIterator[Batch]:
    """Async counterpart of ``iter_batches``, accepting either kind of input.

    Batching is CPU- and disk-bound, not I/O-concurrent, so the work is moved
    off the event loop rather than made concurrent: a synchronous input — a
    file's lines, or records serialized on demand — is consumed entirely in a
    worker thread, and for an async input only the joining and hashing in
    ``close()`` is handed off. Batch boundaries and digests are identical to
    ``iter_batches`` either way, which is what keeps a resumed upload
    replaying the receipts of the batches it already committed.
    """
    if not isinstance(lines, AsyncIterable):
        iterator = iter_batches(
            lines, target_bytes=target_bytes, max_line_bytes=max_line_bytes, max_lines=max_lines
        )
        iterator_lock = threading.Lock()

        def next_batch() -> Union[Batch, None]:
            # Cancelling ``to_thread`` does not stop the worker. Serialize
            # ``next`` and ``close`` so cleanup cannot close a generator that
            # the worker is still executing.
            with iterator_lock:
                return next(iterator, None)

        def close_iterator() -> None:
            with iterator_lock:
                iterator.close()

        try:
            while True:
                # `next` re-enters the generator in a worker thread, one batch
                # at a time, so an aborted upload stops reading the source
                # instead of draining it into memory.
                batch = await asyncio.to_thread(next_batch)
                if batch is None:
                    return
                yield batch
        finally:
            await asyncio.to_thread(close_iterator)
    else:
        builder = _BatchBuilder(
            target_bytes=target_bytes, max_line_bytes=max_line_bytes, max_lines=max_lines
        )
        line_number = 0
        async for line in lines:
            line_number += 1
            if not line.strip():
                continue
            if builder.closes_before(line_number, line):
                yield await asyncio.to_thread(builder.close)
            builder.add(line_number, line)

        if builder.pending:
            yield await asyncio.to_thread(builder.close)
