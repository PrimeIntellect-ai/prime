"""Tests for `prime inference catalog|deploy|deploy-checkpoint|deployments get`."""

import itertools
import json
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest
from prime_cli.commands import inference_deploy
from prime_cli.core import APIError
from prime_cli.main import app
from prime_cli.utils.formatters import strip_ansi
from typer.testing import CliRunner

TEST_ENV = {
    "PRIME_API_KEY": "dummy",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "COLUMNS": "200",
    "TERM": "xterm",
}


def _deployment(status: str, **overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": "dep_1",
        "deploymentName": "my-endpoint",
        "status": status,
        "statusPhase": None,
        "errorMessage": None,
        "servedModelAlias": "qwen3-8b-my-endpoint",
        "endpointUrl": None,
        "checkpoint": None,
        "isOnDemand": True,
        "createdAt": "2026-11-10T00:00:00Z",
        "dashboardUrl": "https://app.primeintellect.ai/dashboard/inference/dep_1",
    }
    if status == "RUNNING":
        payload["endpointUrl"] = "https://dep-1.inference.primeintellect.ai/v1"
    payload.update(overrides)
    return payload


class FakeAPI:
    def __init__(self, gets: Optional[List[Dict[str, Any]]] = None, post: Any = None) -> None:
        self.gets = list(gets or [])
        self.post_response = post
        self.calls: List[Tuple[str, str, Any]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def get(_self: Any, endpoint: str, params: Any = None, timeout: Any = None) -> Any:
            self.calls.append(("GET", endpoint, params))
            nxt = self.gets.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

        def post(_self: Any, endpoint: str, json: Any = None, timeout: Any = None) -> Any:
            self.calls.append(("POST", endpoint, json))
            if isinstance(self.post_response, Exception):
                raise self.post_response
            return self.post_response

        monkeypatch.setattr("prime_cli.core.APIClient.get", get)
        monkeypatch.setattr("prime_cli.core.APIClient.post", post)
        monkeypatch.setattr(inference_deploy.time, "sleep", lambda _s: None)
        monkeypatch.setattr("prime_cli.core.Config.team_id", None, raising=False)


def _run(*args: str):
    result = CliRunner().invoke(app, list(args), env=TEST_ENV)
    return result, strip_ansi(result.output)


def _catalog_model(**overrides: Any) -> Dict[str, Any]:
    """A catalog row exactly as `GET /inference/catalog` serializes one."""
    model: Dict[str, Any] = {
        "id": "cat_qwen",
        "slug": "qwen3-8b",
        "repo": "Qwen/Qwen3-8B",
        "revision": "main",
        "displayName": "Qwen3 8B",
        "quant": "fp8",
        "maxSeqLen": 32768,
        "engines": [
            {"gpuProduct": "NVIDIA-H100-80GB-HBM3", "gpus": 2},
            {"gpuProduct": "NVIDIA-H200", "gpus": 1},
        ],
        "gpuProduct": "NVIDIA-H100-80GB-HBM3",
        "minGpus": 2,
        "enabled": True,
        "metadata": {},
    }
    model.update(overrides)
    return model


def _catalog_cluster(**overrides: Any) -> Dict[str, Any]:
    cluster: Dict[str, Any] = {
        "primeClusterId": "pc_1",
        "name": "team-h200",
        "displayName": "Team H200",
        "inferenceEnabled": True,
        "dynamoProfile": None,
        "gpuProducts": ["NVIDIA-H200"],
        "gatewayReady": True,
        "modelGpus": {"cat_qwen": 1},
    }
    cluster.update(overrides)
    return cluster


def test_catalog_renders_models_and_clusters(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeAPI(
        gets=[
            {
                "models": [
                    _catalog_model(),
                    _catalog_model(id="cat_llama", gpuProduct=None, minGpus=1, quant=None),
                ],
                "clusters": [
                    _catalog_cluster(),
                    _catalog_cluster(
                        primeClusterId="pc_2",
                        displayName=None,
                        name="raw-name",
                        gatewayReady=False,
                    ),
                ],
            }
        ]
    )
    api.install(monkeypatch)

    result, out = _run("inference", "catalog", "--team", "team_1")

    assert result.exit_code == 0, out
    assert api.calls == [("GET", "/inference/catalog", {"teamId": "team_1"})]
    qwen = next(ln for ln in out.splitlines() if "cat_qwen" in ln)
    for cell in ("Qwen3 8B", "Qwen/Qwen3-8B", "fp8", "32768"):
        assert cell in qwen
    assert "yes" in next(ln for ln in out.splitlines() if "pc_1" in ln)
    # displayName falls back to name; gatewayReady false means not ready.
    pc_2 = next(ln for ln in out.splitlines() if "pc_2" in ln)
    assert "raw-name" in pc_2 and "no" in pc_2


def test_catalog_reads_readiness_from_a_pre_rollout_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend still sending the address must not read as not-ready."""
    legacy = _catalog_cluster()
    del legacy["gatewayReady"]
    legacy["gatewayEndpoint"] = "http://100.64.0.7:8000"
    FakeAPI(gets=[{"models": [_catalog_model()], "clusters": [legacy]}]).install(monkeypatch)

    result, out = _run("inference", "catalog", "--team", "team_1")

    assert result.exit_code == 0, out
    assert "yes" in next(ln for ln in out.splitlines() if "pc_1" in ln)
    # The address is still never rendered.
    assert "100.64.0.7" not in out


def test_catalog_gpus_come_from_model_gpus_not_the_preferred_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller's own cluster is the authority on what a model costs there.

    `cat_qwen` declares 2x H100 as its preferred engine but the team's only
    cluster runs it on 1 GPU, and `cat_llama` is a model that cluster cannot
    run at all.
    """
    FakeAPI(
        gets=[
            {
                "models": [_catalog_model(), _catalog_model(id="cat_llama")],
                "clusters": [_catalog_cluster()],
            }
        ]
    ).install(monkeypatch)

    result, out = _run("inference", "catalog", "--team", "team_1")

    assert result.exit_code == 0, out
    assert "1x on your clusters" in next(ln for ln in out.splitlines() if "cat_qwen" in ln)
    # No cluster runs cat_llama; it reports what the on-demand pool would give,
    # never a claim that the model is unavailable.
    llama = next(ln for ln in out.splitlines() if "cat_llama" in ln)
    assert "2x NVIDIA-H100-80GB-HBM3" in llama
    assert "not runnable" not in out


def test_catalog_without_a_capability_map_falls_back_rather_than_claiming_nothing_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clusters carrying no modelGpus say nothing about what runs.

    A backend predating the field looks exactly like this; reading its silence
    as incapability would mark every model unrunnable while deploy still works.
    """
    blind = _catalog_cluster(modelGpus={})
    FakeAPI(gets=[{"models": [_catalog_model()], "clusters": [blind]}]).install(monkeypatch)

    result, out = _run("inference", "catalog", "--team", "team_1")

    assert result.exit_code == 0, out
    qwen = next(ln for ln in out.splitlines() if "cat_qwen" in ln)
    assert "2x NVIDIA-H100-80GB-HBM3" in qwen
    assert "not runnable" not in out


def test_catalog_without_team_clusters_falls_back_to_the_preferred_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeAPI(
        gets=[
            {
                "models": [
                    _catalog_model(),
                    _catalog_model(id="cat_llama", gpuProduct=None, minGpus=1),
                ],
                "clusters": [],
            }
        ]
    ).install(monkeypatch)

    result, out = _run("inference", "catalog")

    assert result.exit_code == 0, out
    assert "2x NVIDIA-H100-80GB-HBM3" in next(ln for ln in out.splitlines() if "cat_qwen" in ln)
    assert "1x any" in next(ln for ln in out.splitlines() if "cat_llama" in ln)


def test_catalog_accepts_a_model_with_no_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`revision` is nullable on the wire; a null must not sink the table."""
    FakeAPI(gets=[{"models": [_catalog_model(revision=None)], "clusters": []}]).install(monkeypatch)

    result, out = _run("inference", "catalog")

    assert result.exit_code == 0, out
    assert "cat_qwen" in out


def test_catalog_shows_team_clusters_even_with_no_deployable_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clusters table is the only CLI source of IDs for --cluster.

    A team can own clusters while the catalog offers it no models (every prod
    team today), and returning early there leaves --cluster undiscoverable.
    """
    FakeAPI(gets=[{"models": [], "clusters": [_catalog_cluster()]}]).install(monkeypatch)

    result, out = _run("inference", "catalog", "--team", "team_1")

    assert result.exit_code == 0, out
    assert "No deployable models" in out
    assert "Team Clusters" in out
    assert "pc_1" in out


def test_catalog_json_is_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"models": [_catalog_model()], "clusters": [_catalog_cluster()]}
    FakeAPI(gets=[payload]).install(monkeypatch)

    result, out = _run("inference", "catalog", "-o", "json")

    assert result.exit_code == 0, out
    assert json.loads(result.stdout) == payload


def test_deploy_sends_body_and_waits_until_running(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeAPI(
        post=_deployment("PENDING"),
        gets=[_deployment("PENDING", statusPhase="PULLING"), _deployment("RUNNING")],
    )
    api.install(monkeypatch)

    result, out = _run(
        "inference",
        "deploy",
        "cat_qwen",
        "--team",
        "team_1",
        "--cluster",
        "pc_1",
        "--name",
        "my-endpoint",
        "--replicas",
        "2",
        "--wait",
    )

    assert result.exit_code == 0, out
    assert api.calls[0] == (
        "POST",
        "/inference/deploy",
        {
            "modelCatalogId": "cat_qwen",
            "replicas": 2,
            "teamId": "team_1",
            "deploymentName": "my-endpoint",
            "primeClusterId": "pc_1",
        },
    )
    assert [c[1] for c in api.calls[1:]] == ["/inference/deployments/dep_1"] * 2
    assert "https://dep-1.inference.primeintellect.ai/v1" in out
    assert "qwen3-8b-my-endpoint" in out
    assert "dashboard/inference/dep_1" in out
    assert "curl https://dep-1.inference.primeintellect.ai/v1/chat/completions" in out


def test_deploy_rejects_invalid_name_without_calling_api(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeAPI(post=_deployment("PENDING"))
    api.install(monkeypatch)

    result, out = _run("inference", "deploy", "cat_qwen", "--name", "Bad_Name")

    assert result.exit_code == 1
    assert "--name must be lowercase" in out
    assert api.calls == []


def test_wait_exits_nonzero_on_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAPI(
        post=_deployment("PENDING"),
        gets=[_deployment("FAILED", errorMessage="out of capacity")],
    ).install(monkeypatch)

    result, out = _run("inference", "deploy", "cat_qwen", "--wait", "-o", "json")

    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "FAILED"


@pytest.mark.parametrize(
    "status", ["FAILED", "STOPPED", "TERMINATING", "TOMBSTONED", "COMPLETED", "UNKNOWN"]
)
def test_wait_stops_on_every_terminal_status(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    """A deployment that will never reach RUNNING must end the wait at once.

    Anything missing here is polled until the 1800s default timeout — delete a
    deployment mid-wait (TERMINATING) and the CLI would sit for half an hour.
    """
    api = FakeAPI(post=_deployment("PENDING"), gets=[_deployment(status)])
    api.install(monkeypatch)

    result, out = _run("inference", "deploy", "cat_qwen", "--wait")

    assert result.exit_code == 1, out
    # Exactly one poll: it stopped on the first read, never looped.
    assert len([c for c in api.calls if c[0] == "GET"]) == 1
    assert status in out


def test_wait_rides_out_a_transient_poll_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """One failed read is not a failed deploy: the deployment is still billing."""
    blip = APIError("HTTP 503: upstream")
    api = FakeAPI(post=_deployment("PENDING"), gets=[blip, _deployment("RUNNING")])
    api.install(monkeypatch)

    result, out = _run("inference", "deploy", "cat_qwen", "--wait")

    assert result.exit_code == 0, out
    assert "RUNNING" in out


def test_wait_gives_up_after_repeated_poll_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    blip = APIError("HTTP 503: upstream")
    FakeAPI(post=_deployment("PENDING"), gets=[blip] * 5).install(monkeypatch)

    result, out = _run("inference", "deploy", "cat_qwen", "--wait")

    assert result.exit_code == 1
    flat = " ".join(out.split())
    # Says the deployment outlived the command, not that the deploy failed.
    assert "Lost contact with the API" in flat
    assert "prime inference deployments get dep_1" in flat


def test_wait_times_out_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAPI(post=_deployment("PENDING"), gets=[_deployment("PENDING")] * 3).install(monkeypatch)
    clock = itertools.count(0.0, 6.0)
    monkeypatch.setattr(inference_deploy.time, "monotonic", lambda: next(clock))

    result, out = _run("inference", "deploy", "cat_qwen", "--wait", "--timeout", "10")

    assert result.exit_code == 1
    flat = " ".join(out.split())
    assert "Timed out after 10s" in flat
    assert "prime inference deployments get dep_1" in flat


def test_wait_never_sleeps_past_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A --timeout shorter than one poll interval must not sleep a full interval."""
    slept: List[float] = []
    monkeypatch.setattr(inference_deploy.time, "sleep", slept.append)
    FakeAPI(post=_deployment("PENDING"), gets=[_deployment("PENDING")] * 3).install(monkeypatch)
    # install() re-stubs sleep, so claim it back afterwards.
    monkeypatch.setattr(inference_deploy.time, "sleep", slept.append)
    clock = itertools.count(0.0, 0.25)
    monkeypatch.setattr(inference_deploy.time, "monotonic", lambda: next(clock))

    result, out = _run("inference", "deploy", "cat_qwen", "--wait", "--timeout", "1")

    assert result.exit_code == 1
    assert "Timed out after 1s" in " ".join(out.split())
    assert slept, "expected at least one poll sleep"
    assert max(slept) <= 1.0, f"slept past the 1s deadline: {slept}"


def test_train_deploy_alias_posts_checkpoint_deploy(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeAPI(post=_deployment("RUNNING", checkpoint={"runId": "run_1", "step": 100}))
    api.install(monkeypatch)

    result, out = _run("train", "deploy", "run_1", "--step", "100", "-o", "json")

    assert result.exit_code == 0, out
    # No --team: the backend defaults to the run's team.
    assert api.calls == [("POST", "/inference/checkpoint-deploy", {"runId": "run_1", "step": 100})]
    assert json.loads(result.stdout)["id"] == "dep_1"


def test_checkpoint_is_rendered_as_prose_not_a_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAPI(gets=[_deployment("RUNNING", checkpoint={"runId": "run_1", "step": 20})]).install(
        monkeypatch
    )

    result, out = _run("inference", "deployments", "get", "dep_1")

    assert result.exit_code == 0, out
    assert "Checkpoint: run run_1 @ step 20" in out
    assert "{'runId'" not in out


def test_api_error_detail_is_shown_without_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("POST", "https://api.primeintellect.ai/api/v1/inference/x")
    response = httpx.Response(409, json={"detail": "Run has no team"}, request=request)
    error = httpx.HTTPStatusError("conflict", request=request, response=response)

    def raising_request(_self: Any, method: str, url: str, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(httpx.Client, "request", raising_request)
    monkeypatch.setattr("prime_cli.core.Config.team_id", None, raising=False)

    result, out = _run("inference", "deploy-checkpoint", "run_1", "--step", "5")

    assert result.exit_code == 1
    assert "Error: HTTP 409: Run has no team" in out
    assert "Traceback" not in out


def test_deployments_get_running_prints_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeAPI(gets=[_deployment("RUNNING")])
    api.install(monkeypatch)

    result, out = _run("inference", "deployments", "get", "dep_1")

    assert result.exit_code == 0, out
    assert api.calls == [("GET", "/inference/deployments/dep_1", None)]
    assert "Endpoint: https://dep-1.inference.primeintellect.ai/v1" in out
