"""Streaming sink over Prime Traces — the primary record transport.

Uploads are content-addressed, so a retry after a lost response replays rather
than duplicates, and episode-aware, so a multi-trace rollout keeps its grouping.

The join key is ``run.id`` inside the trace document, which the ingestion
service indexes; ``context`` is upload-scoped provenance only.
"""

import logging
from typing import Any, Dict, Mapping, Optional, Sequence

from prime_traces import ErrorCode, LineFormat, TracesClient

from .. import _fork
from ..exceptions import ForbiddenError
from .base import Sink, is_episode, stamp_run, to_mapping

logger = logging.getLogger(__name__)

DEFAULT_RECEIPT_HISTORY_SIZE = 100


class TracesSink(Sink):
    """Uploads records through the Prime Traces service."""

    name = "traces"

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        api_key: Optional[str] = None,
        team_id: Optional[str] = None,
        compress: bool = True,
        receipt_history_size: int = DEFAULT_RECEIPT_HISTORY_SIZE,
    ) -> None:
        if receipt_history_size < 0:
            raise ValueError("receipt_history_size must be non-negative")
        self.enabled = True
        self._client = client
        self._injected_client = client is not None
        # The traces service has its own URL (PRIME_TRACES_URL / `traces_url`),
        # which prime-traces resolves itself.
        self._client_kwargs: Dict[str, Any] = {}
        if api_key is not None:
            self._client_kwargs["api_key"] = api_key
        if team_id is not None:
            self._client_kwargs["team_id"] = team_id
        self._compress = compress
        self._run_id: Optional[str] = None
        self._context: Dict[str, str] = {}
        self.receipts: list = []
        self.receipts_received = 0
        self._receipt_history_size = receipt_history_size
        _fork.register(self)

    # ------------------------------------------------------------------ setup

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        self._run_id = run_id
        self._context = {key: str(value) for key, value in context.items() if value is not None}
        self._ensure_client()

    def _ensure_client(self) -> bool:
        """Build the traces client lazily, so a fork reset is repaired on next write."""
        if self._client is not None:
            return True
        if self._injected_client:
            exc = RuntimeError("an injected traces client cannot be reused after a fork")
            self._disable(str(exc))
            raise exc
        try:
            self._client = TracesClient(**self._client_kwargs)
        except Exception as exc:  # noqa: BLE001 - the run applies its error policy
            self._disable(f"could not construct the traces client ({exc})")
            raise
        return True

    def reset_after_fork(self) -> None:
        """Drop (not close) the inherited client; its socket is the parent's."""
        self._client = None

    # ------------------------------------------------------------------ write

    def write(self, records: Sequence[Any]) -> None:
        if not self.enabled or not records or not self._ensure_client():
            return

        # The same bytes under a different format are rejected as a conflict,
        # so infer from the first record, which is stable within a batch.
        line_format = LineFormat.EPISODE if is_episode(records[0]) else LineFormat.TRACE
        payload = [self._prepare(record) for record in records]
        try:
            receipts = list(
                self._client.upload_records(
                    payload,
                    line_format=line_format,
                    context=dict(self._context) or None,
                    compress=self._compress,
                )
            )
        except ForbiddenError as exc:
            # No runtime action fixes a 403, so the sink retires either way.
            # What differs is whether the batch counts as lost.
            if _is_not_enabled(exc):
                # Outside the beta: there was never anywhere for these records
                # to go, so nothing was lost. Not a warning, not a failure.
                self._retire_quietly(
                    f"Prime Traces is not enabled for this account ({exc}); "
                    "continuing with the remaining sinks"
                )
                return
            # A credential without the traces scope is something the caller
            # can fix; raised so loss accounting and strict callers see it.
            self._disable(f"this credential cannot write traces ({exc})")
            raise
        self.receipts_received += len(receipts)
        if self._receipt_history_size:
            self.receipts.extend(receipts)
            del self.receipts[: -self._receipt_history_size]

    def _prepare(self, record: Any) -> Any:
        """The wire mapping for a record, carrying the run on the envelope and
        on every member trace. Producer objects go through their own
        ``to_record()`` here rather than inside the transport — same bytes,
        but the members are reachable for stamping."""
        if self._run_id is None:
            return record
        return stamp_run(to_mapping(record), self._run_id)

    def flush(self) -> None:
        """Uploads are synchronous; nothing is held back here."""

    def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                logger.debug("Error closing the traces client: %s", exc)

    # ----------------------------------------------------------- degradation

    def _disable(self, reason: str) -> None:
        if self.enabled:
            logger.warning("Traces sink disabled: %s", reason)
        self.enabled = False

    def _retire_quietly(self, reason: str) -> None:
        if self.enabled:
            logger.info("Traces sink off: %s", reason)
        self.enabled = False


def _is_not_enabled(exc: ForbiddenError) -> bool:
    """``service_not_enabled``: the account is outside the private beta. The
    other 403, ``forbidden``, means the token lacks the ``traces`` scope."""
    return exc.code == ErrorCode.SERVICE_NOT_ENABLED.value
