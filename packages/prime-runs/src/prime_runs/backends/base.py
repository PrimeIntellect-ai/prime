"""The contract a run backend implements.

A backend owns the *lifecycle* of a run — creating it, updating what is known
about it, closing it out with a terminal status. It does not move records;
that is a sink's job (see :mod:`prime_runs.sinks`).
"""

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from ..models import RunHandle, RunSpec, RunStatus


@runtime_checkable
class Backend(Protocol):
    def create(self, spec: RunSpec) -> RunHandle:
        """Open a new run and return its identity."""
        ...

    def update(
        self,
        run_id: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist config (inputs) and/or summary (outputs).

        ``config`` is the run's *whole* config, not a patch: the evaluations API
        replaces the stored metadata document.
        """
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
        """Close the run out. Called exactly once per run. ``config`` is passed
        so a backend recording terminal state inside metadata can merge it."""
        ...

    def close(self) -> None:
        """Release transport resources."""
        ...
