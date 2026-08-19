"""Local JSONL sink, written in the wire format Prime Traces accepts.

Deliberately not a debug dump: the files this writes are valid trace/episode
JSONL, so ``prime_traces.TracesClient.upload_file`` can send them later
untouched. That is what makes an offline run a *deferred* run rather than a
different one — the run ID stamped into the records was issued at ``init()``
and does not change on sync.
"""

import json
import logging
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional, Sequence, Union

from .. import _fork
from .base import Sink, to_mapping

logger = logging.getLogger(__name__)


class OfflineSink(Sink):
    """Appends records to ``<dir>/<run_id>/records/<line_format>.jsonl``."""

    name = "offline"

    def __init__(self, directory: Union[str, Path], *, stamp_run: bool = True) -> None:
        self.enabled = True
        self.directory = Path(directory)
        self._stamp_run = stamp_run
        self._run_id: Optional[str] = None
        self._run_kind: Optional[str] = None
        self._handles: dict[str, BinaryIO] = {}
        self.records_written = 0
        _fork.register(self)

    def reset_after_fork(self) -> None:
        """Abandon inherited file handles; ``_handle`` reopens on next write.

        Dropping a buffered file object would not be enough on its own: on
        CPython the last reference going away closes it, and ``close()``
        *flushes* — writing out the parent's copied buffer and duplicating every
        record in it. The handles are unbuffered (see ``_handle``) precisely so
        that there is never anything in that buffer to duplicate.
        """
        self._handles = {}

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        self._run_id = run_id
        self._run_kind = context.get("run_kind")
        self._records_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _records_dir(self) -> Path:
        return self.directory / (self._run_id or "unknown") / "records"

    def write(
        self,
        records: Sequence[Any],
        *,
        line_format: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        if not self.enabled or not records:
            return
        name = str(getattr(line_format, "value", line_format) or _infer_format(records[0]))
        handle = self._handle(name)
        for record in records:
            mapping = dict(to_mapping(record))
            if self._stamp_run and not mapping.get("run") and self._run_id:
                run: dict[str, Any] = {"id": self._run_id}
                if self._run_kind:
                    run["type"] = self._run_kind
                mapping["run"] = run
            line = json.dumps(mapping, ensure_ascii=False, separators=(",", ":"), default=str)
            _write_all(handle, (line + "\n").encode("utf-8"))
            self.records_written += 1

    def _handle(self, name: str) -> BinaryIO:
        """An unbuffered append-mode handle for one line format.

        Unbuffered on purpose. A buffered writer keeps records in process memory
        until it decides to flush, and a fork copies that buffer — after which
        both processes eventually write it, putting every record in the file
        twice. Writing straight through means the only copy of a record lives in
        the file, and ``O_APPEND`` keeps concurrent writers from interleaving.
        """
        handle = self._handles.get(name)
        if handle is None:
            self._records_dir.mkdir(parents=True, exist_ok=True)
            handle = open(self._records_dir / f"{name}.jsonl", "ab", buffering=0)
            self._handles[name] = handle
        return handle

    def flush(self) -> None:
        """Nothing is held back — every write already went to the file."""

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except OSError as exc:  # pragma: no cover - teardown must not raise
                logger.debug("Error closing an offline record file: %s", exc)
        self._handles.clear()


def _write_all(handle: BinaryIO, data: bytes) -> None:
    """Write every byte. A raw handle may report a short write."""
    while data:
        written = handle.write(data)
        if not written:  # pragma: no cover - only on a non-blocking handle
            raise OSError("offline record write made no progress")
        data = data[written:]


def _infer_format(record: Any) -> str:
    if isinstance(record, Mapping):
        return "episode" if "traces" in record else "trace"
    return "episode" if hasattr(record, "traces") else "trace"
