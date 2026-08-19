"""The contract a run backend implements.

A backend owns one thing: the *lifecycle* of a run — bringing it into
existence, updating what is known about it, and closing it out with a terminal
status. It does not move samples; that is a sink's job (see
:mod:`prime_runs.sinks`). Keeping the two axes independent is what lets the
eval and training APIs — which agree on almost nothing at the wire level —
share a single ``Run`` handle, and what lets the sample transport change
underneath without touching either.
"""

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from ..models import RunHandle, RunSpec, RunStatus


@runtime_checkable
class Backend(Protocol):
    """Lifecycle operations for one family of runs."""

    kind: str
    """The ``RunKind`` this backend serves."""

    supports_step_metrics: bool
    """Whether ``log_metrics`` records a point per step.

    ``False`` means the API has no time series and the run keeps a last-value
    summary instead. The ``Run`` handle reads this to decide whether
    ``log(..., step=)`` is a real write or a summary merge, so producers get
    the same call either way.
    """

    def create(self, spec: RunSpec) -> RunHandle:
        """Open a new run and return its platform identity."""
        ...

    def attach(self, run_id: str) -> RunHandle:
        """Re-acquire an existing run, for resume and for non-primary ranks."""
        ...

    def update(
        self,
        run_id: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist config (inputs) and/or summary (outputs) mid-run.

        ``config`` is the run's *whole* config, not a patch. The evaluations API
        stores metadata with a document-level ``$set``, so a partial write
        replaces whatever was there — every caller must send the full picture.
        """
        ...

    def log_metrics(self, run_id: str, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Append one point to the run's time series. No-op when unsupported."""
        ...

    def finalize(
        self,
        run_id: str,
        *,
        status: RunStatus,
        summary: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Close the run out. Called exactly once per run.

        ``config`` is passed so a backend that has to record the terminal state
        *inside* metadata can merge it into the full config rather than
        replacing the document with one key.
        """
        ...

    def close(self) -> None:
        """Release transport resources."""
        ...
