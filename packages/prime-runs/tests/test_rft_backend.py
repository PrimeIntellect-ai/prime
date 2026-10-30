"""Training run lifecycle against ``/api/v1/rft/*``."""

import httpx
import pytest
from conftest import RecordingHandler

from prime_runs._http import PlatformClient
from prime_runs.backend import RftBackend
from prime_runs.exceptions import ConfigurationError, ForbiddenError, RetryableAPIError
from prime_runs.models import EnvironmentRef, RunSpec, RunStatus, TrainingSpec


def make_backend(make_platform_client, routes, **kwargs):
    handler = RecordingHandler(routes)
    client = make_platform_client(handler)
    kwargs.setdefault("team_id", "team-1")
    backend = RftBackend(client, frontend_url="https://app.example", **kwargs)
    return backend, handler


def train_spec(**overrides) -> RunSpec:
    values = dict(
        name="test-run",
        kind="train",
        model="Qwen/Qwen3-8B",
        environments=[EnvironmentRef.coerce("primeintellect/vf-math")],
        training=TrainingSpec(max_steps=100, batch_size=64, rollouts_per_example=8, seq_len=4096),
        config={"trainer": {"lr": 1e-6}},
    )
    values.update(overrides)
    return RunSpec(**values)


def test_create_registers_an_external_run_for_the_team(make_platform_client, rft_routes):
    backend, handler = make_backend(make_platform_client, rft_routes)

    handle = backend.create(train_spec())

    assert handler.paths() == ["POST /api/v1/rft/external-runs"]
    body = handler.bodies_for("/api/v1/rft/external-runs")[0]
    assert body["base_model"] == "Qwen/Qwen3-8B"
    assert body["max_steps"] == 100
    assert body["batch_size"] == 64
    assert body["rollouts_per_example"] == 8
    assert body["seq_len"] == 4096
    assert body["run_config"] == {"trainer": {"lr": 1e-6}}
    assert body["team_id"] == "team-1"
    assert body["name"] == "test-run"
    assert handle.id == "run-abc"
    assert handle.url == "https://app.example/dashboard/training/run-abc"


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ("primeintellect/vf-math", {"id": "primeintellect/vf-math"}),
        ("agentic-judge+gsm8k", {"id": "agentic-judge+gsm8k"}),
        ({"id": "env-123", "version_id": "v7"}, {"id": "env-123", "version_id": "v7"}),
    ],
    ids=["slug", "name", "id"],
)
def test_training_environments_are_passed_through_not_resolved(
    make_platform_client, rft_routes, environment, expected
):
    backend, handler = make_backend(make_platform_client, rft_routes)

    backend.create(train_spec(environments=[EnvironmentRef.coerce(environment)]))

    assert handler.bodies_for("/api/v1/rft/external-runs")[0]["environments"] == [expected]
    assert "POST /api/v1/environmentshub/resolve" not in handler.paths()


def test_the_spec_team_wins_over_the_backend_default(make_platform_client, rft_routes):
    backend, handler = make_backend(make_platform_client, rft_routes, team_id="team-default")

    backend.create(train_spec(team_id="team-explicit"))

    assert handler.bodies_for("/api/v1/rft/external-runs")[0]["team_id"] == "team-explicit"


def test_create_needs_a_base_model(make_platform_client, rft_routes):
    backend, handler = make_backend(make_platform_client, rft_routes)

    with pytest.raises(ConfigurationError, match="model="):
        backend.create(train_spec(model=None))
    assert handler.paths() == []


def test_create_needs_a_team(make_platform_client, rft_routes):
    backend, handler = make_backend(make_platform_client, rft_routes, team_id=None)

    with pytest.raises(ConfigurationError, match="team"):
        backend.create(train_spec())
    assert handler.paths() == []


def test_a_team_outside_the_allowlist_is_a_forbidden_error(make_platform_client, rft_routes):
    routes = dict(rft_routes)
    routes["POST /api/v1/rft/external-runs"] = lambda request: httpx.Response(
        403, json={"detail": "External training runs are not enabled for this team"}
    )
    backend, handler = make_backend(make_platform_client, routes)

    with pytest.raises(ForbiddenError, match="not enabled for this team"):
        backend.create(train_spec())
    assert len(handler.paths()) == 1


def test_create_is_not_replayed_after_an_ambiguous_failure(make_platform_client, rft_routes):
    routes = dict(rft_routes)
    routes["POST /api/v1/rft/external-runs"] = lambda request: httpx.Response(502)
    backend, handler = make_backend(make_platform_client, routes)

    with pytest.raises(RetryableAPIError):
        backend.create(train_spec())
    assert handler.paths() == ["POST /api/v1/rft/external-runs"]


def test_attach_makes_no_request(make_platform_client, rft_routes):
    backend, handler = make_backend(make_platform_client, rft_routes)

    handle = backend.attach("run-managed")

    assert handle.id == "run-managed"
    assert handle.url == "https://app.example/dashboard/training/run-managed"
    assert handler.paths() == []


def test_update_is_a_no_op(make_platform_client, rft_routes):
    backend, handler = make_backend(make_platform_client, rft_routes)

    backend.update("run-abc", config={"a": 1}, summary={"loss": 0.1})

    assert handler.paths() == []


def test_a_completed_run_is_finalized_with_exit_code_zero(make_platform_client, rft_routes):
    backend, handler = make_backend(make_platform_client, rft_routes)

    backend.finalize("run-abc", status=RunStatus.COMPLETED, summary={"loss": 0.1})

    assert handler.paths() == ["POST /api/v1/rft/finalize"]
    assert handler.bodies_for("/api/v1/rft/finalize")[0] == {"run_id": "run-abc", "exit_code": 0}


def test_finalize_is_replayed_after_an_ambiguous_failure(make_platform_client, rft_routes):
    attempts = []

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(504)
        return httpx.Response(200, json={"data": {"status": "success"}})

    routes = dict(rft_routes)
    routes["POST /api/v1/rft/finalize"] = flaky
    backend, handler = make_backend(make_platform_client, routes)

    backend.finalize("run-abc", status=RunStatus.COMPLETED)

    assert len(attempts) == 2


def test_finalize_falls_back_to_a_completed_status_update(make_platform_client, rft_routes):
    routes = dict(rft_routes)
    routes["POST /api/v1/rft/finalize"] = lambda request: httpx.Response(
        400, json={"detail": "finalize unavailable"}
    )
    backend, handler = make_backend(make_platform_client, routes)

    backend.finalize("run-abc", status=RunStatus.COMPLETED)

    assert handler.paths() == [
        "POST /api/v1/rft/finalize",
        "PUT /api/v1/rft/external-runs/run-abc/status",
    ]
    assert handler.bodies_for("/api/v1/rft/external-runs/run-abc/status") == [
        {"status": "completed"}
    ]


@pytest.mark.parametrize(
    ("status", "error", "expected_message"),
    [
        (RunStatus.FAILED, "ValueError: boom", "ValueError: boom"),
        (RunStatus.CANCELLED, "interrupted", "cancelled: interrupted"),
        (RunStatus.CRASHED, "process exited without finishing the run", "crashed: process exited"),
        (RunStatus.FAILED, None, "failed"),
    ],
    ids=["failed", "cancelled", "crashed", "failed-no-reason"],
)
def test_other_terminal_states_are_reported_as_failed_with_the_reason(
    make_platform_client, rft_routes, status, error, expected_message
):
    backend, handler = make_backend(make_platform_client, rft_routes)

    backend.finalize("run-abc", status=status, error=error)

    assert handler.paths() == ["PUT /api/v1/rft/external-runs/run-abc/status"]
    body = handler.bodies_for("/api/v1/rft/external-runs/run-abc/status")[0]
    assert body["status"] == "failed"
    assert body["error_message"].startswith(expected_message)
    assert "POST /api/v1/rft/finalize" not in handler.paths()


def test_marking_an_already_closed_run_failed_is_not_an_error(make_platform_client, rft_routes):
    routes = dict(rft_routes)
    routes["PUT /api/v1/rft/external-runs/run-abc/status"] = lambda request: httpx.Response(
        409, json={"detail": "Cannot update status of run in 'COMPLETED' state."}
    )
    backend, handler = make_backend(make_platform_client, routes)

    backend.finalize("run-abc", status=RunStatus.FAILED, error="boom")

    assert handler.paths() == ["PUT /api/v1/rft/external-runs/run-abc/status"]


def test_a_hosted_run_is_finalized_through_the_internal_router():
    """Attached with the launcher's ``$PRIME_API_BASE``: the same call reaches
    ``/api/internal/rft/finalize`` carrying the run's token in ``x-api-key``."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": {"status": "success"}})

    client = PlatformClient(
        api_key="run-token",
        base_url="http://testserver/api/internal/rft",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    backend = RftBackend(client, frontend_url="https://app.example")

    handle = backend.attach("run-managed")
    backend.finalize(handle.id, status=RunStatus.COMPLETED)

    assert [str(r.url) for r in seen] == ["http://testserver/api/internal/rft/finalize"]
    assert seen[0].headers["x-api-key"] == "run-token"
    assert seen[0].headers["Authorization"] == "Bearer run-token"
