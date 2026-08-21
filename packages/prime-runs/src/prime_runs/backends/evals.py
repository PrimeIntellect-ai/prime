"""Eval runs over ``/api/v1/evaluations/*``, plus environment resolution
through the hub's get-or-create so a local run uploads without ``prime env push``.

The eval API has no producer-facing way to mark a run failed: ``finalize``
moves a run to COMPLETED and ``UpdateEvaluationRequest`` carries no status. A
failed or crashed run therefore keeps showing as running; the terminal state is
recorded in ``metadata.prime_runs`` so it is at least visible.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .._http import PlatformClient
from ..exceptions import APIError, ConfigurationError, EnvironmentResolutionError
from ..models import EnvironmentRef, RunHandle, RunSpec, RunStatus

logger = logging.getLogger(__name__)


class EvalsBackend:
    """Lifecycle for evaluation runs."""

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

    # ------------------------------------------------------------------ create

    def create(self, spec: RunSpec) -> RunHandle:
        environments = self._resolve_environments(spec.environments)
        if not environments:
            raise ConfigurationError(
                "An eval run needs at least one environment. Pass "
                'environments=["my-env"] to init().'
            )

        run_name: str = spec.name or _default_name(spec)
        payload: Dict[str, Any] = {
            "name": run_name,
            "environments": environments,
            "tags": list(spec.tags),
        }
        _set_if(payload, "model_name", spec.model)
        # The API's `dataset` column is always the environment under another name.
        _set_if(payload, "dataset", _first_environment_name(spec))
        _set_if(payload, "framework", spec.framework)
        _set_if(payload, "description", spec.description)
        _set_if(payload, "metadata", spec.config or None)
        _set_if(payload, "team_id", spec.team_id or self._team_id)

        # Not replayable (the POST default): a retry after a lost response would
        # create a second run and only the second would be tracked.
        response = self._client.post("/evaluations/", json_body=payload)
        run_id = response.get("evaluation_id")
        if not run_id:
            raise APIError(
                f"POST /evaluations/ returned no evaluation_id (keys: {sorted(response)})"
            )
        return RunHandle(
            id=run_id,
            name=str(response.get("name") or run_name),
            url=response.get("viewer_url") or self.url_for(run_id),
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
        """``config`` must be the whole config: the service writes metadata with
        a document-level ``$set``."""
        payload: Dict[str, Any] = {}
        _set_if(payload, "metadata", config or None)
        _set_if(payload, "metrics", summary or None)
        if not payload:
            return
        self._client.put(f"/evaluations/{run_id}", json_body=payload)

    # ---------------------------------------------------------------- finalize

    def finalize(
        self,
        run_id: str,
        *,
        status: RunStatus,
        summary: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if status is RunStatus.COMPLETED:
            body: Dict[str, Any] = {}
            _set_if(body, "metrics", summary or None)
            # Finalization enqueues the platform's statistics task; replaying an
            # ambiguous failure could enqueue it twice, so this stays non-idempotent.
            self._client.post(
                f"/evaluations/{run_id}/finalize",
                json_body=body or {"metrics": {}},
            )
            return

        # No status endpoint exists yet. Record the terminal state in metadata,
        # merged into the full config because this PUT replaces the document.
        terminal = {
            "status": status.value,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            terminal["error"] = error
        self.update(run_id, config={**(config or {}), "prime_runs": terminal}, summary=summary)
        logger.warning(
            "Run %s %s, but the evaluations API cannot record that; it will keep "
            "showing as running. Recorded the failure in metadata.prime_runs.",
            run_id,
            status.value,
        )

    def close(self) -> None:
        self._client.close()

    # ----------------------------------------------------------- environments

    def _resolve_environments(self, refs: List[EnvironmentRef]) -> List[Dict[str, Any]]:
        """Hub references as the API's ``EnvironmentReference`` objects, carrying
        a pinned ``version_id`` through. A reference that cannot be resolved
        raises rather than being skipped: a run silently attached to the wrong
        environments looks like a successful upload."""
        resolved: List[Dict[str, Any]] = []
        for ref in refs:
            entry: Dict[str, Any] = {"id": ref.id or self._lookup_environment(ref)}
            _set_if(entry, "version_id", ref.version_id)
            resolved.append(entry)
        return resolved

    def _lookup_environment(self, ref: EnvironmentRef) -> str:
        if ref.slug:
            owner_slug, name = ref.slug.split("/", 1)
            try:
                response = self._client.get(f"/environmentshub/{owner_slug}/{name}/@latest")
            except APIError as exc:
                raise EnvironmentResolutionError(
                    f"Could not resolve environment {ref.slug!r}: {exc}"
                ) from exc
            details = response.get("data") or response
            environment_id = details.get("id")
            if not environment_id:
                raise EnvironmentResolutionError(f"Hub returned no id for environment {ref.slug!r}")
            return str(environment_id)

        body: Dict[str, Any] = {"name": ref.name}
        _set_if(body, "team_id", self._team_id)
        try:
            # Get-or-create: a replay returns the same environment.
            response = self._client.post(
                "/environmentshub/resolve", json_body=body, idempotent=True
            )
        except APIError as exc:
            raise EnvironmentResolutionError(
                f"Could not resolve environment {ref.name!r}: {exc}"
            ) from exc
        environment_id = (response.get("data") or {}).get("id")
        if not environment_id:
            raise EnvironmentResolutionError(f"Hub returned no id for environment {ref.name!r}")
        return str(environment_id)


def _set_if(payload: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def _first_environment_name(spec: RunSpec) -> Optional[str]:
    for ref in spec.environments:
        if ref.name:
            return ref.name
        if ref.slug:
            return ref.slug.split("/", 1)[1]
    return None


def _default_name(spec: RunSpec) -> str:
    """The API requires a name; lead with the environment so runs sort together."""
    stem = _first_environment_name(spec) or spec.framework or "eval"
    return f"{stem}-{uuid.uuid4().hex[:8]}"
