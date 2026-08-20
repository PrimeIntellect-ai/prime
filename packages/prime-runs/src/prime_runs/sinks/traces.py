"""Streaming sink over Prime Traces — the primary sample transport.

Records go out as they are produced, in content-addressed JSONL batches. Two
properties of that transport are why the run handle can promise what it does:
uploads are idempotent (the same bytes resolve to the same upload ID, so a
retry after a lost response replays rather than duplicates), and they are
episode-aware, so a multi-trace rollout keeps its grouping instead of being
flattened into one summary row.

**The join key is ``run.id`` inside the trace document, not an upload context
key.** The ingestion service extracts ``run.id`` into an indexed column with a
delete-by-run path; ``context`` is an upload-scoped map that answers a
different question. Producers already stamp the run onto their traces, and
``init()`` returns the ID they stamp — so this sink adds nothing to the join
and uses ``context`` only for provenance.
"""

import logging
from typing import Any, Dict, Mapping, Optional, Sequence

from .. import _fork
from .base import Sink

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
        traces_url: Optional[str] = None,
        team_id: Optional[str] = None,
        stamp_run: bool = True,
        compress: bool = True,
        receipt_history_size: int = DEFAULT_RECEIPT_HISTORY_SIZE,
    ) -> None:
        if receipt_history_size < 0:
            raise ValueError("receipt_history_size must be non-negative")
        self.enabled = True
        self._client = client
        self._injected_client = client is not None
        # Left unset, prime-traces resolves its own endpoint. That matters:
        # the service has its own URL (PRIME_TRACES_URL / config `traces_url`)
        # which is not necessarily the platform API's, and passing the
        # platform base URL through here would quietly override it.
        self._client_kwargs: Dict[str, Any] = {}
        if api_key is not None:
            self._client_kwargs["api_key"] = api_key
        if traces_url is not None:
            self._client_kwargs["base_url"] = traces_url
        if team_id is not None:
            self._client_kwargs["team_id"] = team_id
        self._stamp_run = stamp_run
        self._compress = compress
        self._run_id: Optional[str] = None
        self._run_kind: Optional[str] = None
        self._context: Dict[str, str] = {}
        self.receipts: list = []
        self.receipts_received = 0
        self._receipt_history_size = receipt_history_size
        _fork.register(self)

    # ------------------------------------------------------------------ setup

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        self._run_id = run_id
        self._run_kind = context.get("run_kind")
        self._context = {key: str(value) for key, value in context.items() if value is not None}
        self._ensure_client()

    def _ensure_client(self) -> bool:
        """Build the traces client if we do not have one. Lazy so that a fork
        reset — which drops the inherited client — is repaired on next write."""
        if self._client is not None:
            return True
        if self._injected_client:
            # The caller handed us a client and a fork took it away. Rebuilding
            # would silently swap their transport for a default one.
            return False
        try:
            from prime_traces import TracesClient
        except ImportError as exc:  # pragma: no cover - dependency is declared
            self._disable(f"prime-traces is not installed ({exc})")
            return False
        try:
            self._client = TracesClient(**self._client_kwargs)
        except Exception as exc:  # noqa: BLE001 - construction must not kill a run
            self._disable(f"could not construct the traces client ({exc})")
            return False
        return True

    def reset_after_fork(self) -> None:
        """Drop the inherited traces client; the next write builds a fresh one.

        Not closed: the child's copy of the socket is the parent's connection,
        and shutting it down here would cut the parent off mid-upload.
        """
        self._client = None

    # ------------------------------------------------------------------ write

    def write(
        self,
        records: Sequence[Any],
        *,
        line_format: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        if not self.enabled or not records or not self._ensure_client():
            return

        from prime_traces import LineFormat

        resolved = _resolve_line_format(line_format, records, LineFormat)
        context = dict(self._context)
        if step is not None:
            context["step"] = str(step)

        payload = [self._prepare(record) for record in records]
        try:
            receipts = list(
                self._client.upload_records(
                    payload,
                    line_format=resolved,
                    context=context or None,
                    compress=self._compress,
                )
            )
        except Exception as exc:  # noqa: BLE001 - classified below
            if self._is_gated(exc):
                self._disable(
                    f"Prime Traces is not enabled for this account ({exc}); "
                    "falling back to the remaining sinks"
                )
                # The worker contains this error in the default warn mode and
                # continues with the remaining sinks. Raising is still required
                # so strict callers see the failed batch and loss accounting is
                # updated instead of reporting a successful traces-only run.
                raise
            raise
        self.receipts_received += len(receipts)
        if self._receipt_history_size:
            self.receipts.extend(receipts)
            del self.receipts[: -self._receipt_history_size]

    def _prepare(self, record: Any) -> Any:
        """Stamp the run onto plain mappings that do not already carry one.

        Producer objects are passed through untouched — verifiers and prime-rl
        both stamp the run themselves at rollout time, and rewriting a caller's
        object to add something it already has is how two sources of truth for
        the run ID appear. A bare dict has no such convention, so filling in
        the indexed field is the difference between a queryable run and an
        orphaned upload.
        """
        if not self._stamp_run or not isinstance(record, Mapping):
            return record
        if record.get("run"):
            return record
        run: Dict[str, Any] = {"id": self._run_id}
        if self._run_kind:
            run["type"] = self._run_kind
        return {**record, "run": run}

    def flush(self) -> None:
        """Uploads are synchronous, so nothing is held back here.

        Batching happens inside ``upload_records``; the asynchrony a producer
        cares about lives one level up, in the uploader thread.
        """

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
        """Whether this failure means "not allowed", not "try again".

        Prime Traces is in closed beta: a non-allowlisted account gets 403
        ``service_not_enabled``, and a write-only hosted-eval token gets 403
        ``forbidden`` on anything it may not do. Neither is fixable at runtime,
        so the sink turns itself off instead of retrying for the rest of the run.
        """
        try:
            from prime_traces.exceptions import ForbiddenError
        except ImportError:  # pragma: no cover - dependency is declared
            return False
        return isinstance(exc, ForbiddenError)


def _resolve_line_format(line_format: Optional[str], records: Sequence[Any], enum: Any) -> Any:
    """Pick the wire format, preferring what the caller said.

    The default is inferred from the records themselves: anything carrying
    ``traces`` is an episode. Guessing wrong is not cosmetic — the same bytes
    submitted under a different format are rejected as a conflict — so the
    inference only ever looks at the first record's shape, which is stable
    within a batch a producer handed over as a unit.
    """
    if line_format is not None:
        return enum(line_format) if not isinstance(line_format, enum) else line_format
    first = records[0]
    mapping = _try_mapping(first)
    if mapping is not None:
        return enum.EPISODE if "traces" in mapping else enum.TRACE
    return enum.EPISODE if hasattr(first, "traces") else enum.TRACE


def _try_mapping(record: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(record, Mapping):
        return record
    return None
