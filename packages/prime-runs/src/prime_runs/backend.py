"""Run backends: the contract, the evaluations backend, and the disabled no-op.

A backend owns the *lifecycle* of a run — creating it, updating what is known
about it, closing it out with a terminal status. It does not move records;
that is a sink's job (see :mod:`prime_runs.sinks`).

:class:`EvalsBackend` works over ``/api/v1/evaluations/*``, resolving
environments through the hub's get-or-create so a local run uploads without
``prime env push``. The eval API has no producer-facing way to mark a run
failed: ``finalize`` moves a run to COMPLETED and ``UpdateEvaluationRequest``
carries no status. A failed or crashed run therefore keeps showing as running;
the terminal state is recorded in ``metadata.prime_runs`` so it is at least
visible.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

from ._http import PlatformClient
from .exceptions import APIError, ConfigurationError, EnvironmentResolutionError
from .models import EnvironmentRef, RunHandle, RunSpec, RunStatus, TrainingSpec

logger = logging.getLogger(__name__)


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


# ------------------------------------------------------------------- evals


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


# ---------------------------------------------------------------- disabled


def disabled_run_id() -> str:
    """A locally issued ID, visibly distinct from a platform one."""
    return f"disabled-{uuid.uuid4().hex[:16]}"


class DisabledBackend:
    """No-op lifecycle, so ``mode="disabled"`` needs no branching upstream."""

    def create(self, spec: RunSpec) -> RunHandle:
        return RunHandle(id=disabled_run_id())

    def update(self, run_id: str, **kwargs: Any) -> None:
        return None

    def finalize(self, run_id: str, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


# ------------------------------------------------------------------- rft

#: The RFT API's status vocabulary is ``completed | failed``; the SDK's finer
#: distinctions travel in ``error_message``.
RFT_TERMINAL_STATUS = {
    RunStatus.COMPLETED: "completed",
    RunStatus.FAILED: "failed",
    RunStatus.CANCELLED: "failed",
    RunStatus.CRASHED: "failed",
}


class RftBackend:
    """Lifecycle for external training runs over ``/api/v1/rft/*``.

    What prime-rl's ``TrainRun`` did, in the SDK: register a run (or attach to
    one a managed launch created), close it out. External runs are created
    *for a team* and the platform enables them per team, so ``team_id`` is
    required and a team outside the allowlist gets a 403 from :meth:`create`;
    the producer decides whether that means "run locally" (the way verifiers
    treats a missing key) or "stop".

    The API has no config/summary update — ``run_config`` travels with
    :meth:`create` and :meth:`update` is a no-op — and its status vocabulary
    is ``completed | failed``: ``cancelled`` and ``crashed`` are reported as
    ``failed`` with the reason in ``error_message``.
    """

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
        if not spec.model:
            raise ConfigurationError("A training run needs model= — the base model being trained.")
        team_id = spec.team_id or self._team_id
        if not team_id:
            raise ConfigurationError(
                "A training run needs a team: pass team_id= or set PRIME_TEAM_ID. "
                "External training runs are created for a team, and the platform "
                "enables them per team."
            )
        training = spec.training or TrainingSpec()
        payload: Dict[str, Any] = {
            "base_model": spec.model,
            "max_steps": int(training.max_steps),
            # Environment ids are passed through: training environments are
            # named by hub id, and the RFT API does its own resolution.
            "environments": [_training_environment(ref) for ref in spec.environments],
            "team_id": team_id,
        }
        _set_if(payload, "name", spec.name)
        _set_if(payload, "batch_size", training.batch_size)
        _set_if(payload, "rollouts_per_example", training.rollouts_per_example)
        _set_if(payload, "seq_len", training.seq_len)
        _set_if(payload, "run_config", spec.config or None)
        _set_if(payload, "wandb_project", training.wandb_project)
        _set_if(payload, "wandb_entity", training.wandb_entity)
        _set_if(payload, "wandb_run_name", training.wandb_run_name)

        # Not replayable: a retry after a lost response would register a
        # second run and only the second would be tracked.
        response = self._client.post("/rft/external-runs", json_body=payload)
        run = response.get("run") or {}
        run_id = run.get("id")
        if not run_id:
            raise APIError(f"POST /rft/external-runs returned no run id (keys: {sorted(response)})")
        return RunHandle(
            id=str(run_id),
            name=str(run.get("name") or spec.name or "") or None,
            url=self.url_for(str(run_id)),
        )

    def attach(self, run_id: str) -> RunHandle:
        """A handle for a run the platform already created — a managed launch
        that injected its id. No request: the first write proves access."""
        return RunHandle(id=run_id, url=self.url_for(run_id))

    def url_for(self, run_id: str) -> str:
        return f"{self._frontend_url}/dashboard/training/{run_id}"

    # ------------------------------------------------------------------ update

    def update(
        self,
        run_id: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """No-op: an RFT run takes its config at creation and has no summary
        document. Run-level outputs go through ``log_metrics``."""
        if config or summary:
            logger.debug("Run %s: the RFT API has no config/summary update; nothing sent", run_id)

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
            # Compare-and-set on the server, and "already finalized" is a
            # success, so a lost response is safe to replay.
            try:
                self._client.post(
                    "/rft/finalize",
                    json_body={"run_id": run_id, "exit_code": 0},
                    idempotent=True,
                )
            except APIError as exc:
                # Preserve TrainRun's fallback: finalization may be unavailable
                # even though the status endpoint can still close the run out.
                logger.warning(
                    "Run %s could not be finalized (%s); falling back to a status update",
                    run_id,
                    exc,
                )
                self._set_terminal_status(run_id, status=status)
            return

        message = error or status.value
        if status is not RunStatus.FAILED:
            message = f"{status.value}: {message}"
        self._set_terminal_status(run_id, status=status, error_message=message)

    def _set_terminal_status(
        self,
        run_id: str,
        *,
        status: RunStatus,
        error_message: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {"status": RFT_TERMINAL_STATUS[status]}
        _set_if(payload, "error_message", error_message)
        try:
            self._client.put(
                f"/rft/external-runs/{run_id}/status",
                json_body=payload,
            )
        except APIError as exc:
            if exc.status_code == 409:
                # Already closed out (a managed launcher, or an earlier attempt
                # whose response was lost). The outcome is recorded either way.
                logger.info(
                    "Run %s is already closed on the platform; not marking it %s",
                    run_id,
                    status.value,
                )
                return
            raise

    def log_metrics(self, run_id: str, metrics: Dict[str, Any]) -> None:
        """One step's metrics, synchronously. The run handle batches these
        through its uploader; this is the direct call."""
        self._client.post("/rft/metrics", json_body={"run_id": run_id, "metrics": metrics})

    def close(self) -> None:
        self._client.close()


def _training_environment(ref: EnvironmentRef) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"id": ref.id or ref.slug or ref.name}
    _set_if(entry, "version_id", ref.version_id)
    return entry
