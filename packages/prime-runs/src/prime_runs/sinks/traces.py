"""Streaming sink over Prime Traces — the primary record transport.

Uploads are content-addressed, so a retry after a lost response replays rather
than duplicates, and episode-aware, so a multi-trace rollout keeps its grouping.

The join key is ``run.id`` inside the trace document, which the ingestion
service indexes; ``context`` is upload-scoped provenance only.
"""

import logging
from typing import Any, Dict, Mapping, Optional, Sequence

from .. import _fork
from .base import Sink, is_episode, stamp_run

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
            from prime_traces import TracesClient
        except ImportError as exc:  # pragma: no cover - dependency is declared
            self._disable(f"prime-traces is not installed ({exc})")
            raise
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

        from prime_traces import LineFormat

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
        except Exception as exc:
            if self._is_gated(exc):
                self._disable(
                    f"Prime Traces is not enabled for this account ({exc}); "
                    "falling back to the remaining sinks"
                )
            # Re-raised either way so the worker's loss accounting and strict
            # callers see the failed batch.
            raise
        self.receipts_received += len(receipts)
        if self._receipt_history_size:
            self.receipts.extend(receipts)
            del self.receipts[: -self._receipt_history_size]

    def _prepare(self, record: Any) -> Any:
        """Stamp the run onto bare mappings; producer objects pass through."""
        if not isinstance(record, Mapping) or self._run_id is None:
            return record
        return stamp_run(record, self._run_id)

    def flush(self) -> None:
        """Uploads are synchronous; nothing is held back here."""

    def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                logger.debug("Error closing the traces client: %s", exc)

    # ----------------------------------------------------------- degradation

    def _disable(self, reason: str) -> None:
        if self.enabled:
            logger.warning("Traces sink disabled: %s", reason)
        self.enabled = False

    @staticmethod
    def _is_gated(exc: Exception) -> bool:
        """A 403 (``service_not_enabled``, or a write-only token) is not
        fixable at runtime, so the sink turns itself off instead of retrying."""
        try:
            from prime_traces.exceptions import ForbiddenError
        except ImportError:  # pragma: no cover - dependency is declared
            return False
        return isinstance(exc, ForbiddenError)
