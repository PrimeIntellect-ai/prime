"""Eval runs over ``/api/v1/evaluations/*``.

Wraps the endpoints verifiers previously called inline, plus environment
resolution through the hub's get-or-create so a local run uploads without a
prior ``prime env push``.

One gap is worth stating plainly, because it shapes the code below: **the eval
API has no producer-facing way to mark a run failed.** ``finalize`` moves a run
PROCESSING -> COMPLETED, ``UpdateEvaluationRequest`` carries no ``status``, and
FAILED is written only when an internal Cloud Task trigger fails. So a crashed
run stays RUNNING forever. ``_report_failure`` calls the status endpoint this
SDK needs, treats its absence as expected, and falls back to recording the
terminal state in ``metadata`` so the failure is at least visible and machine
readable. When the endpoint lands, the fallback stops firing on its own.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .._http import PlatformClient
from ..exceptions import (
    ConfigurationError,
    EnvironmentResolutionError,
    NotFoundError,
    RunAPIError,
)
from ..models import EnvironmentRef, RunHandle, RunSpec, RunStatus

logger = logging.getLogger(__name__)

# Statuses the platform's EvaluationStatus enum uses, keyed by ours.
_PLATFORM_STATUS = {
    RunStatus.COMPLETED: "COMPLETED",
    RunStatus.FAILED: "FAILED",
    RunStatus.CRASHED: "FAILED",
}


class EvalsBackend:
    """Lifecycle for evaluation runs."""

    kind = "eval"
    # The evaluations API stores a single metrics blob, not a time series.
    supports_step_metrics = False

    def __init__(
        self,
        client: PlatformClient,
        *,
        frontend_url: str,
        team_id: Optional[str] = None,
    ) -> None:
        self._client = client
        self._frontend_url = frontend_url.rstrip("/")
        self._team_id = team_id
        self._status_endpoint_missing = False

    # ------------------------------------------------------------------ create

    def create(self, spec: RunSpec) -> RunHandle:
        environment_ids = self._resolve_environments(spec.environments)
        if not environment_ids:
            raise ConfigurationError(
                "An eval run needs at least one environment. Pass "
                'environments=["my-env"] to init().'
            )

        run_name: str = spec.name or _default_name(spec)
        payload: Dict[str, Any] = {
            "name": run_name,
            "environments": [{"id": environment_id} for environment_id in environment_ids],
            "tags": list(spec.tags),
        }
        _set_if(payload, "model_name", spec.model)
        _set_if(payload, "dataset", spec.dataset or _first_environment_name(spec))
        _set_if(payload, "framework", spec.framework)
        _set_if(payload, "description", spec.description)
        _set_if(payload, "metadata", spec.config or None)
        _set_if(payload, "metrics", spec.summary or None)
        _set_if(payload, "team_id", spec.team_id or self._team_id)

        response = self._client.post("/evaluations/", json_body=payload)
        run_id = response.get("evaluation_id")
        if not run_id:
            raise RunAPIError(
                f"POST /evaluations/ returned no evaluation_id (keys: {sorted(response)})"
            )
        return RunHandle(
            id=run_id,
            name=str(response.get("name") or run_name),
            url=response.get("viewer_url") or self.url_for(run_id),
            raw=response,
        )

    def attach(self, run_id: str) -> RunHandle:
        try:
            response = self._client.get(f"/evaluations/{run_id}")
        except NotFoundError:
            raise
        except RunAPIError as exc:
            # Attach is a convenience — a resume or a non-primary rank joining.
            # Losing the run's name to a transient read is not worth failing on;
            # the ID is what everything downstream actually needs.
            logger.debug("Could not read evaluation %s on attach: %s", run_id, exc)
            return RunHandle(id=run_id, url=self.url_for(run_id))
        return RunHandle(
            id=run_id,
            name=response.get("name"),
            url=response.get("viewer_url") or self.url_for(run_id),
            raw=response,
        )

    def url_for(self, run_id: str) -> str:
        return f"{self._frontend_url}/dashboard/evaluations/{run_id}"

    # ------------------------------------------------------------------ update

    def update(
        self,
        run_id: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {}
        _set_if(payload, "metadata", config or None)
        _set_if(payload, "metrics", summary or None)
        if not payload:
            return
        self._client.put(f"/evaluations/{run_id}", json_body=payload)

    def log_metrics(self, run_id: str, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """No-op: see ``supports_step_metrics``.

        The run keeps a last-value summary and flushes it through ``update``,
        which is the only shape this API can store.
        """

    # ---------------------------------------------------------------- finalize

    def finalize(
        self,
        run_id: str,
        *,
        status: RunStatus,
        summary: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        if status is RunStatus.COMPLETED:
            body: Dict[str, Any] = {}
            _set_if(body, "metrics", summary or None)
            self._client.post(f"/evaluations/{run_id}/finalize", json_body=body or {"metrics": {}})
            return
        self._report_failure(run_id, status=status, summary=summary, error=error)

    def _report_failure(
        self,
        run_id: str,
        *,
        status: RunStatus,
        summary: Optional[Dict[str, Any]],
        error: Optional[str],
    ) -> None:
        """Mark a run failed, or record why we could not."""
        terminal = {
            "status": status.value,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            terminal["error"] = error

        if not self._status_endpoint_missing:
            try:
                self._client.post(
                    f"/evaluations/{run_id}/status",
                    json_body={"status": _PLATFORM_STATUS[status], "error": error},
                    max_attempts=1,
                )
                return
            except NotFoundError:
                # Expected until the status endpoint ships. Latch so a run that
                # fails repeatedly does not pay for the probe every time.
                self._status_endpoint_missing = True
                logger.debug(
                    "Platform has no /evaluations/{id}/status endpoint; "
                    "recording terminal state in metadata instead"
                )
            except RunAPIError as exc:
                if exc.status_code not in (405, 422):
                    raise
                self._status_endpoint_missing = True
                logger.debug("Status endpoint rejected the request (%s); using metadata", exc)

        # Fallback: the run cannot be moved out of RUNNING, but the failure is
        # at least recorded where an operator and the dashboard can both read it.
        self.update(run_id, config={"prime_runs": terminal}, summary=summary)
        logger.warning(
            "Run %s %s, but the platform has no way to mark an evaluation failed; "
            "it will keep showing as running. Recorded the failure in metadata.prime_runs.",
            run_id,
            status.value,
        )

    def close(self) -> None:
        self._client.close()

    # ----------------------------------------------------------- environments

    def _resolve_environments(self, refs: List[EnvironmentRef]) -> List[str]:
        """Environment IDs for the hub references a producer named.

        Unlike the old client, a reference that cannot be resolved raises
        instead of being skipped: dropping one silently produces a run attached
        to the wrong environments, which looks like a successful upload and is
        found much later.
        """
        resolved: List[str] = []
        for ref in refs:
            if ref.id:
                resolved.append(ref.id)
                continue
            body: Dict[str, Any] = {"name": ref.name}
            _set_if(body, "team_id", self._team_id)
            try:
                response = self._client.post("/environmentshub/resolve", json_body=body)
            except RunAPIError as exc:
                raise EnvironmentResolutionError(
                    f"Could not resolve environment {ref.name!r}: {exc}"
                ) from exc
            environment_id = (response.get("data") or {}).get("id")
            if not environment_id:
                raise EnvironmentResolutionError(f"Hub returned no id for environment {ref.name!r}")
            resolved.append(environment_id)
        return resolved


def _set_if(payload: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def _first_environment_name(spec: RunSpec) -> Optional[str]:
    for ref in spec.environments:
        if ref.name:
            return ref.name
    return None


def _default_name(spec: RunSpec) -> str:
    """A name for producers that did not pick one.

    The API requires a name, so the alternative to generating one is a 422 at
    the worst possible moment. Leads with the environment so runs sort together
    in the dashboard list.
    """
    stem = _first_environment_name(spec) or spec.framework or spec.kind
    return f"{stem}-{uuid.uuid4().hex[:8]}"
