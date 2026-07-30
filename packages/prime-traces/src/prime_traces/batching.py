"""Client-side batching for content-addressed uploads.

The contract this implements (see the Prime Traces design docs):

- The SDK reads a completed JSONL file line by line; lines are opaque bytes.
  It never parses trace JSON, so it needs no dependency on verifiers.
- A request chunk closes at its uncompressed-byte threshold. Boundaries are
  measured in uncompressed bytes because compression ratios vary widely.
- A line larger than the single-line limit is rejected locally rather than
  split: under bare-trace format the line is one trace, under episode format
  one complete episode, and neither may span requests.
- batch_id = SHA256(exact uncompressed JSONL bytes, including newlines),
  sent as ``Idempotency-Key: sha256:<64 lowercase hex>``. There is no client
  upload ID or checkpoint: a crashed producer re-reads the file, rebuilds the
  same bytes, and regenerates the same keys — so batching must be
  deterministic for a given input.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Union

from .exceptions import TraceTooLargeError

MIB = 1024 * 1024

#: Server-side cap on uncompressed bytes per request (-> batch_too_large).
MAX_BATCH_BYTES = 256 * MIB
#: Server-side cap on one JSONL line (-> trace_too_large).
MAX_LINE_BYTES = 64 * MIB
#: Default chunk-close threshold; a batch may exceed this only when a single
#: line does, and never exceeds MAX_BATCH_BYTES.
DEFAULT_TARGET_BATCH_BYTES = 128 * MIB


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


def iter_batches(
    lines: Iterable[bytes],
    *,
    target_bytes: int = DEFAULT_TARGET_BATCH_BYTES,
    max_line_bytes: int = MAX_LINE_BYTES,
) -> Iterator[Batch]:
    """Group JSONL lines into content-addressed request batches.

    Whitespace-only lines are skipped. Each kept line contributes its exact
    bytes (terminator included; a final line without one stays without one).
    The line-size cap is checked on the line content, excluding the
    terminator, matching how the service stores lines.
    """
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    if target_bytes > MAX_BATCH_BYTES:
        raise ValueError(
            f"target_bytes ({target_bytes}) exceeds the {MAX_BATCH_BYTES} byte request cap"
        )

    buffer: list[bytes] = []
    buffered_size = 0
    batch_start_line = 0

    def close() -> Batch:
        nonlocal buffer, buffered_size
        data = b"".join(buffer)
        batch = Batch(
            data=data,
            digest=hashlib.sha256(data).hexdigest(),
            num_lines=len(buffer),
            first_line_number=batch_start_line,
        )
        buffer = []
        buffered_size = 0
        return batch

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        content_size = len(line.rstrip(b"\r\n"))
        if content_size > max_line_bytes:
            raise TraceTooLargeError(line_number, content_size, max_line_bytes)

        if buffer and buffered_size + len(line) > target_bytes:
            yield close()
        if not buffer:
            batch_start_line = line_number
        buffer.append(line)
        buffered_size += len(line)

    if buffer:
        yield close()
