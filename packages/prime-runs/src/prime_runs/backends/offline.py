"""Offline runs: a local directory that looks exactly like a platform run.

This is not a debugging affordance, it is the reason producers can delete their
``--no-push`` branching. A run that never reaches the network still has an ID,
a status, a config, a summary and a metrics stream, so the call sites above it
are identical whether or not anyone is logged in. The locally issued ID is used
as the run ID everywhere — including inside the trace documents — so a later
sync attaches the archive to a platform run without rewriting a single record.

Layout, one directory per run::

    <dir>/<run_id>/run.json       spec + status + timestamps
    <dir>/<run_id>/metrics.jsonl  one JSON object per log() call
    <dir>/<run_id>/records/       whatever the offline sink wrote
"""

import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..models import RunHandle, RunSpec, RunStatus

logger = logging.getLogger(__name__)

DEFAULT_DIR_ENV = "PRIME_RUNS_DIR"
DEFAULT_DIR = "prime-runs"


def default_dir() -> Path:
    return Path(os.getenv(DEFAULT_DIR_ENV) or DEFAULT_DIR)


def new_run_id() -> str:
    """A locally issued run ID, visibly distinct from a platform one."""
    return f"offline-{uuid.uuid4().hex[:16]}"


class OfflineBackend:
    """Run lifecycle recorded on the local filesystem."""

    kind = "offline"
    supports_step_metrics = True

    def __init__(self, directory: Union[str, Path, None] = None) -> None:
        self.directory = Path(directory) if directory is not None else default_dir()

    def run_dir(self, run_id: str) -> Path:
        return self.directory / run_id

    def create(self, spec: RunSpec) -> RunHandle:
        run_id = new_run_id()
        path = self.run_dir(run_id)
        path.mkdir(parents=True, exist_ok=True)
        run_name: str = spec.name or run_id
        state: Dict[str, Any] = {
            "id": run_id,
            "name": run_name,
            "kind": spec.kind,
            "status": RunStatus.RUNNING.value,
            "created_at": _now(),
            "spec": _spec_to_json(spec),
        }
        self._write_state(run_id, state)
        return RunHandle(id=run_id, name=run_name, url=str(path.resolve()), raw=state)

    def attach(self, run_id: str) -> RunHandle:
        path = self.run_dir(run_id)
        path.mkdir(parents=True, exist_ok=True)
        state = self._read_state(run_id)
        return RunHandle(
            id=run_id,
            name=str(state.get("name") or run_id),
            url=str(path.resolve()),
            raw=state,
        )

    def update(
        self,
        run_id: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = self._read_state(run_id)
        if config:
            state.setdefault("config", {}).update(config)
        if summary:
            state.setdefault("summary", {}).update(summary)
        state["updated_at"] = _now()
        self._write_state(run_id, state)

    def log_metrics(self, run_id: str, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        line = {"step": step, "timestamp": _now(), **metrics}
        path = self.run_dir(run_id) / "metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

    def finalize(
        self,
        run_id: str,
        *,
        status: RunStatus,
        summary: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = self._read_state(run_id)
        if config:
            state.setdefault("config", {}).update(config)
        state["status"] = status.value
        state["finished_at"] = _now()
        if error:
            state["error"] = error
        if summary:
            state.setdefault("summary", {}).update(summary)
        self._write_state(run_id, state)

    def close(self) -> None:
        """Nothing to release — every write is already flushed to disk."""

    # ------------------------------------------------------------------ state

    def _state_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.json"

    def _read_state(self, run_id: str) -> Dict[str, Any]:
        path = self._state_path(run_id)
        if not path.exists():
            return {"id": run_id}
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.warning("Could not read %s (%s); starting a fresh record", path, exc)
            return {"id": run_id}
        return state if isinstance(state, dict) else {"id": run_id}

    def _write_state(self, run_id: str, state: Dict[str, Any]) -> None:
        path = self._state_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a crash mid-write must not leave the run's own
        # record truncated, since it is the only description of what ran.
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), "utf-8")
        temp.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _spec_to_json(spec: RunSpec) -> Dict[str, Any]:
    data = asdict(spec)
    data["environments"] = [
        {key: value for key, value in env.items() if value is not None}
        for env in data.get("environments", [])
    ]
    return data
