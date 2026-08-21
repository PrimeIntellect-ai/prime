"""Eval run lifecycle against ``/api/v1/evaluations/*``."""

import httpx
import pytest
from conftest import RecordingHandler

from prime_runs.backends import EvalsBackend
from prime_runs.exceptions import (
    ConfigurationError,
    EnvironmentResolutionError,
    RetryableAPIError,
)
from prime_runs.models import EnvironmentRef, RunSpec, RunStatus


def make_backend(make_platform_client, routes, **kwargs):
    handler = RecordingHandler(routes)
    client = make_platform_client(handler)
    backend = EvalsBackend(client, frontend_url="https://app.example", **kwargs)
    return backend, handler


def test_create_resolves_environment_names_through_the_hub(make_platform_client, eval_routes):
    backend, handler = make_backend(make_platform_client, eval_routes)
    spec = RunSpec(
        name="test-run",
        environments=[EnvironmentRef(name="gsm8k")],
        model="Qwen/Qwen3-8B",
        framework="verifiers",
    )

    handle = backend.create(spec)

    assert handler.paths()[0] == "POST /api/v1/environmentshub/resolve"
    created = handler.bodies_for("/api/v1/evaluations/")[0]
    assert created["environments"] == [{"id": "env-123"}]
    assert created["model_name"] == "Qwen/Qwen3-8B"
    assert created["framework"] == "verifiers"
    # The environment name doubles as the dataset, as the old uploader did.
    assert created["dataset"] == "gsm8k"
    assert handle.id == "eval-abc"
    assert handle.url == "https://app.example/dashboard/evaluations/eval-abc"


def test_an_explicit_environment_id_skips_the_hub(make_platform_client, eval_routes):
    backend, handler = make_backend(make_platform_client, eval_routes)

    backend.create(RunSpec(name="r", environments=[EnvironmentRef(id="env-999")]))

    assert "POST /api/v1/environmentshub/resolve" not in handler.paths()
    assert handler.bodies_for("/api/v1/evaluations/")[0]["environments"] == [{"id": "env-999"}]


@pytest.mark.parametrize("environment", ["alice/gsm8k", {"slug": "alice/gsm8k"}])
def test_a_published_environment_slug_uses_owner_aware_lookup(
    make_platform_client, eval_routes, environment
):
    routes = dict(eval_routes)
    routes["GET /api/v1/environmentshub/alice/gsm8k/@latest"] = {"data": {"id": "env-published"}}
    backend, handler = make_backend(make_platform_client, routes)

    backend.create(RunSpec(name="r", environments=[EnvironmentRef.coerce(environment)]))

    assert handler.paths()[0] == "GET /api/v1/environmentshub/alice/gsm8k/@latest"
    assert "POST /api/v1/environmentshub/resolve" not in handler.paths()
    assert handler.bodies_for("/api/v1/evaluations/")[0]["environments"] == [
        {"id": "env-published"}
    ]


def test_a_published_environment_slug_supplies_dataset_and_default_name(
    make_platform_client, eval_routes
):
    routes = dict(eval_routes)
    routes["GET /api/v1/environmentshub/alice/gsm8k/@latest"] = {"data": {"id": "env-published"}}
    backend, handler = make_backend(make_platform_client, routes)

    backend.create(RunSpec(environments=[EnvironmentRef.coerce("alice/gsm8k")]))

    created = handler.bodies_for("/api/v1/evaluations/")[0]
    assert created["dataset"] == "gsm8k"
    assert created["name"].startswith("gsm8k-")


def test_an_unresolvable_environment_fails_the_run_rather_than_being_skipped(
    make_platform_client, eval_routes
):
    """Silently dropping it produces a run attached to the wrong environments —
    an upload that looks successful and is discovered wrong much later."""
    routes = dict(eval_routes)
    routes["POST /api/v1/environmentshub/resolve"] = lambda request: httpx.Response(
        404, json={"detail": "no such environment"}
    )
    backend, _ = make_backend(make_platform_client, routes)

    with pytest.raises(EnvironmentResolutionError, match="gsm8k"):
        backend.create(RunSpec(name="r", environments=[EnvironmentRef(name="gsm8k")]))


def test_a_run_with_no_environments_is_rejected_before_any_request(
    make_platform_client, eval_routes
):
    backend, handler = make_backend(make_platform_client, eval_routes)

    with pytest.raises(ConfigurationError, match="at least one environment"):
        backend.create(RunSpec(name="r"))
    assert handler.requests == []


def test_a_run_without_a_name_gets_one(make_platform_client, eval_routes):
    """The API requires a name; the alternative to generating one is a 422 at
    the worst possible moment."""
    backend, handler = make_backend(make_platform_client, eval_routes)

    backend.create(RunSpec(environments=[EnvironmentRef(name="gsm8k")]))

    assert handler.bodies_for("/api/v1/evaluations/")[0]["name"].startswith("gsm8k-")


def test_team_id_is_forwarded_to_the_hub_and_the_run(make_platform_client, eval_routes):
    backend, handler = make_backend(make_platform_client, eval_routes, team_id="team-7")

    backend.create(RunSpec(name="r", environments=[EnvironmentRef(name="gsm8k")]))

    assert handler.bodies_for("/api/v1/environmentshub/resolve")[0]["team_id"] == "team-7"
    assert handler.bodies_for("/api/v1/evaluations/")[0]["team_id"] == "team-7"


def test_finalizing_a_completed_run_posts_its_metrics(make_platform_client, eval_routes):
    backend, handler = make_backend(make_platform_client, eval_routes)

    backend.finalize("eval-abc", status=RunStatus.COMPLETED, summary={"avg_reward": 0.75})

    assert handler.bodies_for("/api/v1/evaluations/eval-abc/finalize")[0] == {
        "metrics": {"avg_reward": 0.75}
    }


def test_an_ambiguous_finalize_failure_is_not_replayed(make_platform_client, eval_routes):
    """Finalization enqueues asynchronous processing, so a retry can enqueue it twice."""
    routes = dict(eval_routes)
    routes["POST /api/v1/evaluations/eval-abc/finalize"] = lambda request: httpx.Response(502)
    backend, handler = make_backend(make_platform_client, routes)

    with pytest.raises(RetryableAPIError):
        backend.finalize("eval-abc", status=RunStatus.COMPLETED)

    assert handler.paths().count("POST /api/v1/evaluations/eval-abc/finalize") == 1


def test_a_failed_run_is_recorded_in_metadata(make_platform_client, eval_routes, caplog):
    """The platform has no producer-facing way to fail an evaluation. The run
    cannot leave RUNNING, but the failure is recorded where an operator and the
    dashboard can read it, and the SDK says the run will keep showing as running."""
    backend, handler = make_backend(make_platform_client, eval_routes)

    with caplog.at_level("WARNING"):
        backend.finalize("eval-abc", status=RunStatus.FAILED, error="boom")

    assert "POST /api/v1/evaluations/eval-abc/finalize" not in handler.paths()
    terminal = handler.bodies_for("/api/v1/evaluations/eval-abc")[0]["metadata"]["prime_runs"]
    assert terminal["status"] == "failed"
    assert terminal["error"] == "boom"
    assert "keep showing as running" in caplog.text


def test_update_sends_nothing_when_there_is_nothing_to_send(make_platform_client, eval_routes):
    backend, handler = make_backend(make_platform_client, eval_routes)

    backend.update("eval-abc")

    assert handler.requests == []


def test_a_pinned_environment_version_reaches_the_api(make_platform_client, eval_routes):
    """The API's EnvironmentReference carries version_id. Dropping it attaches
    the run to whatever version the hub resolves today — the difference between
    a reproducible eval and one that quietly moved."""
    backend, handler = make_backend(make_platform_client, eval_routes)

    backend.create(RunSpec(name="r", environments=[EnvironmentRef(id="env-1", version_id="v-7")]))

    assert handler.bodies_for("/api/v1/evaluations/")[0]["environments"] == [
        {"id": "env-1", "version_id": "v-7"}
    ]


def test_a_version_pin_survives_hub_resolution(make_platform_client, eval_routes):
    backend, handler = make_backend(make_platform_client, eval_routes)

    backend.create(RunSpec(name="r", environments=[EnvironmentRef(name="gsm8k", version_id="v-7")]))

    assert handler.bodies_for("/api/v1/evaluations/")[0]["environments"] == [
        {"id": "env-123", "version_id": "v-7"}
    ]


def test_the_failure_fallback_preserves_the_run_config(make_platform_client, eval_routes):
    """The service writes metadata with a document-level $set, so a PUT carrying
    only the terminal block would erase everything finish() just wrote."""
    backend, handler = make_backend(make_platform_client, eval_routes)

    backend.finalize(
        "eval-abc",
        status=RunStatus.FAILED,
        error="boom",
        config={"num_rollouts": 4, "model": "Qwen3-8B"},
    )

    metadata = handler.bodies_for("/api/v1/evaluations/eval-abc")[0]["metadata"]
    assert metadata["num_rollouts"] == 4
    assert metadata["model"] == "Qwen3-8B"
    assert metadata["prime_runs"]["status"] == "failed"
