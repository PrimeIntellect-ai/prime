"""Training metrics sink: one ``POST /rft/metrics`` per logged step.

The RFT metrics endpoint takes a single step's metrics per request, so a
coalesced batch costs one request per record. The platform allows 60 a minute
per token; a 429 is retried with the server's ``Retry-After`` before it counts
as a strike.

Requests are replayed after an ambiguous failure (a 502/504, a read timeout):
the platform keeps one row per step, so a duplicate collapses on its side,
while a lost row is a hole in the training curves for good.
"""

import logging
from typing import Any, Mapping, Optional, Sequence

from .._http import PlatformClient
from ..exceptions import is_record_rejection, is_transient
from .base import Sink, SinkWriteError

logger = logging.getLogger(__name__)


class RftMetricsSink(Sink):
    """Posts per-step metrics dicts to the RFT metrics endpoint."""

    name = "rft_metrics"

    def __init__(self, client: PlatformClient) -> None:
        self.enabled = True
        self._client = client
        self._run_id: Optional[str] = None
        self.steps_written = 0

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        self._run_id = run_id

    def write(self, records: Sequence[Any]) -> None:
        if not self.enabled or not records:
            return
        if self._run_id is None:
            raise RuntimeError("RftMetricsSink.write called before start()")

        failed = 0
        reported: Optional[Exception] = None
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                failed += 1
                if reported is None:
                    reported = TypeError(f"metrics must be a mapping, got {type(record).__name__}")
                continue
            try:
                # Replayable: the platform merges rows by step, so a retry after
                # a lost response cannot duplicate anything visible.
                self._client.post(
                    "/rft/metrics",
                    json_body={"run_id": self._run_id, "metrics": dict(record)},
                    idempotent=True,
                )
            except Exception as exc:  # noqa: BLE001 - classified below
                failed += 1
                if not (is_record_rejection(exc) or is_transient(exc)):
                    # Sink-wide: the rest would fail the same way.
                    failed += len(records) - index - 1
                    reported = exc
                    break
                # Prefer a transient error for the worker's strike accounting.
                if reported is None or is_transient(exc):
                    reported = exc
            else:
                self.steps_written += 1

        if reported is not None:
            raise SinkWriteError(reported, failed_records=failed) from reported

    def flush(self) -> None:
        """Writes are synchronous; the uploader thread owns the asynchrony."""

    def close(self) -> None:
        """The client is shared with the backend, which closes it."""
