"""Local JSONL sink, written in the wire format Prime Traces accepts, so the
files can later be sent by ``prime_traces.TracesClient.upload_file`` untouched."""

import logging
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional, Sequence, Union

from .. import _fork
from .._http import encode_json
from .base import Sink, is_episode, stamp_run, to_mapping

logger = logging.getLogger(__name__)


class OfflineSink(Sink):
    """Appends records to ``<dir>/<run_id>/records/{trace,episode}.jsonl``."""

    name = "offline"

    def __init__(self, directory: Union[str, Path]) -> None:
        self.enabled = True
        self.directory = Path(directory)
        self._run_id: Optional[str] = None
        self._handles: dict[str, BinaryIO] = {}
        self.records_written = 0
        _fork.register(self)

    def reset_after_fork(self) -> None:
        """Abandon inherited file handles. They are unbuffered (see ``_handle``),
        so dropping them cannot flush a copy of the parent's buffer."""
        self._handles = {}

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        self._run_id = run_id
        self._records_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _records_dir(self) -> Path:
        return self.directory / (self._run_id or "unknown") / "records"

    def write(self, records: Sequence[Any]) -> None:
        if not self.enabled or not records:
            return
        handle = self._handle("episode" if is_episode(records[0]) else "trace")
        for record in records:
            mapping = to_mapping(record)
            if self._run_id:
                mapping = stamp_run(mapping, self._run_id)
            # Same strict encoder as the online path: an archive that holds
            # NaN cannot later be uploaded.
            _write_all(handle, encode_json(mapping) + b"\n")
            self.records_written += 1

    def _handle(self, name: str) -> BinaryIO:
        """An unbuffered append-mode handle per line format. Unbuffered so a
        fork never copies pending records; ``O_APPEND`` keeps writers whole."""
        handle = self._handles.get(name)
        if handle is None:
            self._records_dir.mkdir(parents=True, exist_ok=True)
            handle = open(self._records_dir / f"{name}.jsonl", "ab", buffering=0)
            self._handles[name] = handle
        return handle

    def flush(self) -> None:
        """Every write already went to the file."""

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
