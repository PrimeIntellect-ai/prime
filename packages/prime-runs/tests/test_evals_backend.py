"""Eval run lifecycle against ``/api/v1/evaluations/*``."""

import httpx
import pytest
from conftest import RecordingHandler

from prime_runs.backend import EvalsBackend
from prime_runs.exceptions import (
    APIError,
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


def test_attach_makes_no_request(make_platform_client, eval_routes):
    backend, handler = make_backend(make_platform_client, eval_routes)

    handle = backend.attach("eval-hosted")

    assert handle.id == "eval-hosted"
    assert handle.url == "https://app.example/dashboard/evaluations/eval-hosted"
    assert handler.paths() == []


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
    routes = dict(eval_routes)
    routes["POST /api/v1/evaluations/eval-abc/finalize"] = lambda request: httpx.Response(502)
    backend, handler = make_backend(make_platform_client, routes)

    with pytest.raises(RetryableAPIError):
        backend.finalize("eval-abc", status=RunStatus.COMPLETED)

    assert handler.paths().count("POST /api/v1/evaluations/eval-abc/finalize") == 1


def test_a_failed_run_is_closed_through_the_status_update(
    make_platform_client, eval_routes, caplog
):
    backend, handler = make_backend(make_platform_client, eval_routes)

    with caplog.at_level("WARNING"):
        backend.finalize("eval-abc", status=RunStatus.FAILED, error="boom")

    assert "POST /api/v1/evaluations/eval-abc/finalize" not in handler.paths()
    assert handler.bodies_for("/api/v1/evaluations/eval-abc") == [
        {"status": "FAILED", "error_message": "boom"}
    ]
    assert caplog.text == ""


def test_a_cancelled_run_sends_no_error_message_unless_given(make_platform_client, eval_routes):
    backend, handler = make_backend(make_platform_client, eval_routes)

    backend.finalize("eval-abc", status=RunStatus.CANCELLED)

    assert handler.bodies_for("/api/v1/evaluations/eval-abc") == [{"status": "CANCELLED"}]


def test_a_crashed_run_arrives_as_failed_with_the_reason(make_platform_client, eval_routes):
    backend, handler = make_backend(make_platform_client, eval_routes)

    backend.finalize("eval-abc", status=RunStatus.CRASHED, error="exited without finishing")

    assert handler.bodies_for("/api/v1/evaluations/eval-abc") == [
        {"status": "FAILED", "error_message": "crashed: exited without finishing"}
    ]


def test_the_error_message_is_capped_at_the_api_limit(make_platform_client, eval_routes):
    backend, handler = make_backend(make_platform_client, eval_routes)

    backend.finalize("eval-abc", status=RunStatus.FAILED, error="x" * 5000)

    sent = handler.bodies_for("/api/v1/evaluations/eval-abc")[0]["error_message"]
    assert len(sent) == 4096


def test_a_run_the_platform_already_closed_is_left_alone(make_platform_client, eval_routes, caplog):
    routes = dict(eval_routes)
    routes["PUT /api/v1/evaluations/eval-abc"] = lambda request: httpx.Response(
        409, json={"detail": "Evaluation is COMPLETED; only PENDING or RUNNING ..."}
    )
    backend, handler = make_backend(make_platform_client, routes)

    with caplog.at_level("INFO"):
        backend.finalize("eval-abc", status=RunStatus.FAILED, error="boom")

    assert handler.paths().count("PUT /api/v1/evaluations/eval-abc") == 1
    assert "already closed" in caplog.text


def test_a_hosted_run_rejection_is_raised(make_platform_client, eval_routes):
    routes = dict(eval_routes)
    routes["PUT /api/v1/evaluations/eval-abc"] = lambda request: httpx.Response(
        400, json={"detail": "Hosted evaluations cannot be closed through this endpoint"}
    )
    backend, _ = make_backend(make_platform_client, routes)

    with pytest.raises(APIError):
        backend.finalize("eval-abc", status=RunStatus.FAILED, error="boom")


def test_a_pinned_environment_version_reaches_the_api(make_platform_client, eval_routes):
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


def test_a_failed_run_still_lands_its_config_and_summary(make_platform_client, eval_routes):
    backend, handler = make_backend(make_platform_client, eval_routes)

    backend.finalize(
        "eval-abc",
        status=RunStatus.FAILED,
        error="boom",
        summary={"avg_reward": 0.1},
        config={"num_rollouts": 4, "model": "Qwen3-8B"},
    )

    # The config/summary go up first, on their own: a status guard that rejects
    # the update (409) must not take the run's config with it.
    bodies = handler.bodies_for("/api/v1/evaluations/eval-abc")
    assert bodies == [
        {"metadata": {"num_rollouts": 4, "model": "Qwen3-8B"}, "metrics": {"avg_reward": 0.1}},
        {"status": "FAILED", "error_message": "boom"},
    ]
    assert "prime_runs" not in bodies[0]["metadata"]
