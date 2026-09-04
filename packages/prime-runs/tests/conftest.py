"""Shared fixtures. Every test is hermetic: no network, no real ~/.prime."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import httpx
import pytest

from prime_runs._http import PlatformClient

_PRIME_ENV_VARS = (
    "PRIME_API_KEY",
    "PRIME_TEAM_ID",
    "PRIME_API_BASE_URL",
    "PRIME_BASE_URL",
    "PRIME_TRACES_URL",
    "PRIME_FRONTEND_URL",
    "PRIME_RUNS_MODE",
)


@pytest.fixture(autouse=True)
def isolated_prime_config(monkeypatch, tmp_path):
    """Never read the developer's real ~/.prime or env vars."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in _PRIME_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Record retry backoff instead of waiting it out."""
    sleeps: List[float] = []
    monkeypatch.setattr("prime_runs._http.time.sleep", sleeps.append)
    return sleeps


class RecordingHandler:
    """A MockTransport handler that records requests and replies from a route map."""

    def __init__(self, routes: Dict[str, Any]) -> None:
        self.routes = routes
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"
        route = self.routes.get(key)
        if route is None:
            return httpx.Response(404, json={"detail": f"no route for {key}"})
        if callable(route):
            return route(request)
        return httpx.Response(200, json=route)

    def paths(self) -> List[str]:
        return [f"{r.method} {r.url.path}" for r in self.requests]

    def bodies_for(self, path: str) -> List[Any]:
        import json

        return [json.loads(r.content) for r in self.requests if r.url.path == path and r.content]


@pytest.fixture
def make_platform_client() -> Callable[..., PlatformClient]:
    def _make(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> PlatformClient:
        client = httpx.Client(
            base_url="http://testserver",
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer test-key"},
        )
        return PlatformClient(
            api_key="test-key", base_url="http://testserver", client=client, **kwargs
        )

    return _make


@pytest.fixture
def eval_routes() -> Dict[str, Any]:
    """The happy path for an eval run: resolve env, create, samples, finalize."""
    return {
        "POST /api/v1/environmentshub/resolve": {"data": {"id": "env-123"}},
        "POST /api/v1/evaluations/": lambda request: httpx.Response(
            201,
            json={
                "evaluation_id": "eval-abc",
                "name": "test-run",
                "status": "RUNNING",
                "eval_type": "environment",
                "viewer_url": "https://app.example/dashboard/evaluations/eval-abc",
                "created_at": "2026-08-19T00:00:00Z",
            },
        ),
        "POST /api/v1/evaluations/eval-abc/samples": {
            "evaluation_id": "eval-abc",
            "samples_pushed": 1,
            "status": "RUNNING",
        },
        "POST /api/v1/evaluations/eval-abc/finalize": {
            "evaluation_id": "eval-abc",
            "status": "PROCESSING",
        },
        "PUT /api/v1/evaluations/eval-abc": {
            "evaluation_id": "eval-abc",
            "name": "test-run",
            "status": "RUNNING",
            "updated_at": "2026-08-19T00:00:00Z",
        },
    }


class FakeSink:
    """A sink that records what it was given, and can be told to fail."""

    def __init__(self, name: str = "fake", fail_on_write: bool = False) -> None:
        self.name = name
        self.enabled = True
        self.fail_on_write = fail_on_write
        self.started: List[Any] = []
        self.batches: List[Any] = []
        self.flushes = 0
        self.closed = False

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        self.started.append((run_id, dict(context)))

    def write(self, records: Sequence[Any]) -> None:
        if self.fail_on_write:
            raise RuntimeError("sink is broken")
        self.batches.append(list(records))

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_sink() -> Callable[..., FakeSink]:
    return FakeSink


@pytest.fixture
def rft_routes() -> Dict[str, Any]:
    """The happy path for a training run: register, metrics, samples, finalize."""

    def presign(request: httpx.Request) -> httpx.Response:
        import json

        step = json.loads(request.content)["step"]
        return httpx.Response(
            200,
            json={
                "data": {
                    "presignedUrl": f"http://storage.test/run-abc/step-{step}.parquet",
                    "s3Key": f"runs/run-abc/step-{step}.parquet",
                    "expiresIn": 900,
                }
            },
        )

    return {
        "POST /api/v1/rft/external-runs": lambda request: httpx.Response(
            201,
            json={"run": {"id": "run-abc", "name": "test-run", "status": "RUNNING"}},
        ),
        "POST /api/v1/rft/metrics": {"data": {"status": "success"}},
        "POST /api/v1/rft/samples/presign": presign,
        "POST /api/v1/rft/samples/confirm": {"data": {"status": "success"}},
        "POST /api/v1/rft/finalize": {
            "data": {"status": "success", "message": "Run finalized successfully"}
        },
        "PUT /api/v1/rft/external-runs/run-abc/status": {
            "run": {"id": "run-abc", "status": "FAILED"}
        },
    }


class StorageHandler:
    """Object storage for presigned PUTs: records every upload, answers 200."""

    def __init__(self, status_codes: Sequence[int] = ()) -> None:
        self.uploads: List[httpx.Request] = []
        self._status_codes = list(status_codes)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.uploads.append(request)
        code = self._status_codes.pop(0) if self._status_codes else 200
        return httpx.Response(code)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))
