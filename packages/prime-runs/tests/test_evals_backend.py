"""Eval run lifecycle against ``/api/v1/evaluations/*``."""

import httpx
import pytest
from conftest import RecordingHandler

from prime_runs.backends import EvalsBackend
from prime_runs.exceptions import ConfigurationError, EnvironmentResolutionError
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


def test_a_failed_run_falls_back_to_metadata_when_the_status_endpoint_is_missing(
    make_platform_client, eval_routes, caplog
):
    """The platform has no producer-facing way to fail an evaluation yet.

    Until it does, the run cannot leave RUNNING — but the failure must still be
    recorded somewhere an operator and the dashboard can both read it, and the
    SDK must say plainly that the run will keep showing as running.
    """
    backend, handler = make_backend(make_platform_client, eval_routes)

    with caplog.at_level("WARNING"):
        backend.finalize("eval-abc", status=RunStatus.FAILED, error="boom")

    assert "POST /api/v1/evaluations/eval-abc/status" in handler.paths()
    terminal = handler.bodies_for("/api/v1/evaluations/eval-abc")[0]["metadata"]["prime_runs"]
    assert terminal["status"] == "failed"
    assert terminal["error"] == "boom"
    assert "keep showing as running" in caplog.text


def test_the_missing_status_endpoint_is_probed_once_per_backend(make_platform_client, eval_routes):
    backend, handler = make_backend(make_platform_client, eval_routes)

    backend.finalize("eval-abc", status=RunStatus.FAILED, error="one")
    backend.finalize("eval-abc", status=RunStatus.CRASHED, error="two")

    assert handler.paths().count("POST /api/v1/evaluations/eval-abc/status") == 1


def test_a_status_endpoint_that_exists_is_used_instead_of_the_fallback(
    make_platform_client, eval_routes
):
    routes = dict(eval_routes)
    routes["POST /api/v1/evaluations/eval-abc/status"] = {"evaluation_id": "eval-abc"}
    backend, handler = make_backend(make_platform_client, routes)

    backend.finalize("eval-abc", status=RunStatus.FAILED, error="boom")

    assert handler.bodies_for("/api/v1/evaluations/eval-abc/status")[0] == {
        "status": "FAILED",
        "error": "boom",
    }
    assert "PUT /api/v1/evaluations/eval-abc" not in handler.paths()


def test_update_sends_nothing_when_there_is_nothing_to_send(make_platform_client, eval_routes):
    backend, handler = make_backend(make_platform_client, eval_routes)

    backend.update("eval-abc")

    assert handler.requests == []


def test_attach_survives_a_read_failure(make_platform_client, eval_routes):
    """Losing a run's name to a transient read is not worth failing a resume on."""
    routes = dict(eval_routes)
    routes["GET /api/v1/evaluations/eval-abc"] = lambda request: httpx.Response(
        500, json={"detail": "nope"}
    )
    backend, _ = make_backend(make_platform_client, routes)

    handle = backend.attach("eval-abc")

    assert handle.id == "eval-abc"
    assert handle.url == "https://app.example/dashboard/evaluations/eval-abc"
