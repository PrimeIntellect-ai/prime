"""Client-side batching for content-addressed uploads."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Union

from .exceptions import TraceTooLargeError

MIB = 1024 * 1024

# Server-side cap on uncompressed bytes per request (-> upload_too_large).
MAX_BATCH_BYTES = 256 * MIB
# Server-side cap on one JSONL line (-> trace_too_large).
MAX_LINE_BYTES = 64 * MIB
# Server-side cap on staged rows per request (-> too_many_traces_in_upload).
# Exact for bare-trace uploads, where one line is one row. Episode lines
# stage one episode row plus each nested trace, which opaque line bytes
# cannot count — there the byte caps are the practical bound.
MAX_BATCH_LINES = 1_000_000
# Default chunk-close threshold. Held under Cloud Run's 32 MiB request cap so
# a batch fits the deployed transport even with compression disabled; the
# 256 MiB service contract above is not reachable through that cap today. A
# batch may exceed this only when a single line does — such a batch (or any
# caller-raised target) still risks transport-level rejection before the
# service sees it.
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


def iter_batches(
    lines: Iterable[bytes],
    *,
    target_bytes: int = DEFAULT_TARGET_BATCH_BYTES,
    max_line_bytes: int = MAX_LINE_BYTES,
    max_lines: int = MAX_BATCH_LINES,
) -> Iterator[Batch]:
    """Group JSONL lines into content-addressed request batches."""
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    if target_bytes > MAX_BATCH_BYTES:
        raise ValueError(
            f"target_bytes ({target_bytes}) exceeds the {MAX_BATCH_BYTES} byte request cap"
        )
    if max_lines <= 0:
        raise ValueError("max_lines must be positive")

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
        content_size = len(line.rstrip(b"\n"))
        if content_size > max_line_bytes:
            raise TraceTooLargeError(line_number, content_size, max_line_bytes)

        if buffer and (buffered_size + len(line) > target_bytes or len(buffer) >= max_lines):
            yield close()
        if not buffer:
            batch_start_line = line_number
        buffer.append(line)
        buffered_size += len(line)

    if buffer:
        yield close()
