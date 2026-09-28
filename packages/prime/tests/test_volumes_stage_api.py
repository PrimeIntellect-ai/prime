"""`prime volumes stage` API mode: the default needs no kubectl, no
kubeconfig, and no cluster credentials - and never falls back to the
operator kubectl path."""

import json
from typing import Any, Optional

import pytest
from prime_cli.api.training import Volume
from prime_cli.commands import volumes_stage
from prime_cli.core import APIError
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {"PRIME_DISABLE_VERSION_CHECK": "1", "KUBECONFIG": ""}

STAGE_ID = "9f2c5b0e-1a4d-4f5e-9b3c-2d1e0f6a7b8c"

VOLUME = {
    "name": "sft-datasets",
    "size": "1Ti",
    "status": "RUNNING",
    "clusterId": "cluster-1",
    "namespace": "prime-user-1-local-test",
    "pvcName": "vol-sft-datasets",
    "createdBy": "user-1",
    "createdAt": "2026-09-26T00:00:00Z",
}


def _result(**over) -> dict[str, Any]:
    payload = {
        "status": "staged",
        "operationId": STAGE_ID,
        "source": "acme/tiny-sft",
        "requestedRevision": "main",
        "revision": "sha-1234",
        "datasetName": "tiny-sft",
        "bytes": 100,
        "files": 2,
        "configs": {"default": {"splits": {"train": {"rows": 10}}}},
        "elapsedSeconds": 1.0,
    }
    payload.update(over)
    return payload


def _envelope(status: str, *, result: Optional[dict] = None, error: Optional[dict] = None) -> dict:
    return {
        "stageId": STAGE_ID,
        "volume": "sft-datasets",
        "status": status,
        "source": "acme/tiny-sft",
        "requestedRevision": "main",
        "dataName": "/datasets/tiny-sft",
        "namespace": "prime-user-1-local-test",
        "clusterId": "cluster-1",
        "pvcName": "vol-sft-datasets",
        "expiresAt": None,
        "result": result,
        "error": error,
    }


class FakeStageClient:
    """HostedTrainingClient stand-in: only the three stage endpoints."""

    def __init__(self):
        self.volume = Volume.model_validate(VOLUME)
        self.states: list[dict] = []
        self.cancellations: list[str] = []
        self.admitted: list[dict] = []
        self.stage_error: Optional[APIError] = None
        self.get_error: Optional[APIError] = None

    def list_volumes(self, team_id: Optional[str] = None):
        return [self.volume]

    def stage_volume(self, volume, **kwargs):
        if self.stage_error is not None:
            raise self.stage_error
        self.admitted.append({"volume": volume, **kwargs})
        return type("E", (), {})  # replaced below

    def get_volume_stage(self, volume, stage_id, *, team_id=None):
        if self.get_error is not None:
            raise self.get_error
        return type("E", (), {})()

    def cancel_volume_stage(self, volume, stage_id, *, team_id=None):
        self.cancellations.append(stage_id)
        return {"stageId": stage_id, "status": "CANCELLING"}


def _patch_client(monkeypatch, fake: FakeStageClient):
    import prime_cli.commands.volumes_stage as vs
    from prime_cli.api.training import VolumeStage
    from pydantic import BaseModel

    class Envelope(BaseModel):
        pass

    # Use the real pydantic model so attribute access mirrors production.
    def _model(d):
        return VolumeStage.model_validate(d)

    def stage_volume(volume, **kwargs):
        if fake.stage_error is not None:
            raise fake.stage_error
        fake.admitted.append({"volume": volume, **kwargs})
        return _model(_envelope("PENDING"))

    def get_volume_stage(volume, stage_id, *, team_id=None):
        if fake.get_error is not None:
            raise fake.get_error
        if not fake.states:
            return _model(_envelope("SUCCEEDED", result=_result()))
        state = fake.states.pop(0)
        if isinstance(state, dict):
            return _model(
                _envelope(
                    state["status"],
                    result=state.get("result"),
                    error=state.get("error"),
                )
            )
        return _model(_envelope(state, result=None))

    def cancel_volume_stage(volume, stage_id, *, team_id=None):
        fake.cancellations.append(stage_id)
        return {"stageId": stage_id, "status": "CANCELLING"}

    fake.stage_volume = stage_volume
    fake.get_volume_stage = get_volume_stage
    fake.cancel_volume_stage = cancel_volume_stage

    monkeypatch.setattr("prime_cli.commands.volumes._client", lambda: (fake, None))
    monkeypatch.setattr(vs, "API_POLL_SECONDS", 0.0)
    # No kubectl anywhere: PATH must not be consulted and Kubernetes is
    # forbidden - any attempt to use the operator path fails the test.
    monkeypatch.setattr(vs.shutil, "which", lambda name: None)
    monkeypatch.setattr(vs, "Kubectl", type("Forbidden", (), {"__init__": _no_kubectl}))


def _no_kubectl(*_a, **_k):
    raise AssertionError("the operator kubectl path must never run in API mode")


def _invoke(*args):
    return runner.invoke(app, list(args), env=TEST_ENV)


BASE = ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"]


def test_api_mode_stages_without_kubectl(monkeypatch):
    fake = FakeStageClient()
    _patch_client(monkeypatch, fake)
    result = _invoke(*BASE, "--output", "json")
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "staged"
    assert payload["dataName"] == "/datasets/tiny-sft"
    assert payload["stageId"] == STAGE_ID
    assert payload["kubeContext"] is None
    assert payload["revision"] == "sha-1234"
    # one admission with a UUID idempotency key (a fresh one per
    # invocation; the backend echoes it back as the stageId)
    (admitted,) = fake.admitted
    assert volumes_stage._STAGE_UUID_RE.fullmatch(admitted["idempotency_key"])
    assert admitted["source"] == "acme/tiny-sft"


def test_api_failure_never_falls_back(monkeypatch):
    fake = FakeStageClient()
    fake.stage_error = APIError("HTTP 409: an operation is already active (stage %s)" % STAGE_ID)
    fake.stage_error.status_code = 409
    _patch_client(monkeypatch, fake)
    result = _invoke(*BASE)
    assert result.exit_code == 1, result.output
    flat = " ".join(result.output.split())
    assert "already active" in flat
    assert STAGE_ID in flat  # the conflicting operation is surfaced, not retried


def test_api_mode_uses_configured_team_id(monkeypatch):
    fake = FakeStageClient()
    _patch_client(monkeypatch, fake)

    def client_with_team():
        return fake, "team-42"

    monkeypatch.setattr("prime_cli.commands.volumes._client", client_with_team)
    result = _invoke(*BASE)
    assert result.exit_code == 0, result.output + result.stderr
    (admitted,) = fake.admitted
    assert admitted["team_id"] == "team-42"


def test_api_failure_result_is_a_failure(monkeypatch):
    fake = FakeStageClient()
    fake.states = [
        {"status": "RUNNING"},
        {
            "status": "FAILED",
            "error": {"code": "DATASET_INACCESSIBLE", "message": "private or gated"},
        },
    ]
    _patch_client(monkeypatch, fake)
    result = _invoke(*BASE)
    assert result.exit_code == 1, result.output
    flat = " ".join(result.output.split())
    assert "DATASET_INACCESSIBLE" in flat


def test_succeeded_without_result_payload_is_never_success(monkeypatch):
    fake = FakeStageClient()
    fake.states = [{"status": "SUCCEEDED", "result": None}]
    _patch_client(monkeypatch, fake)
    result = _invoke(*BASE)
    assert result.exit_code == 1, result.output
    assert "no result payload" in " ".join(result.output.split())


@pytest.mark.parametrize(
    "over",
    [
        {"bytes": True},
        {"files": -1},
        {"revision": ""},
        {"configs": {}},
    ],
)
def test_malformed_api_results_fail_closed(monkeypatch, over):
    fake = FakeStageClient()
    fake.states = [{"status": "SUCCEEDED", "result": _result(**over)}]
    _patch_client(monkeypatch, fake)
    result = _invoke(*BASE)
    assert result.exit_code == 1, result.output


def test_result_must_match_the_operation(monkeypatch):
    fake = FakeStageClient()
    fake.states = [{"status": "SUCCEEDED", "result": _result(source="acme/other")}]
    _patch_client(monkeypatch, fake)
    result = _invoke(*BASE)
    assert result.exit_code == 1, result.output


def test_interrupt_cancels_and_reports_the_stage(monkeypatch):
    fake = FakeStageClient()
    _patch_client(monkeypatch, fake)

    def interrupting_get(volume, stage_id, *, team_id=None):
        raise KeyboardInterrupt()

    fake.get_volume_stage = interrupting_get
    result = _invoke(*BASE)
    assert result.exit_code == 130
    assert fake.cancellations == [STAGE_ID]
    flat = " ".join(result.output.split())
    assert STAGE_ID in flat


def test_throttled_status_reads_back_off(monkeypatch):
    fake = FakeStageClient()
    _patch_client(monkeypatch, fake)
    throttled = APIError("HTTP 429: too many requests")
    throttled.status_code = 429
    calls = {"n": 0}

    from prime_cli.api.training import VolumeStage

    def flaky_get(volume, stage_id, *, team_id=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise throttled
        return VolumeStage.model_validate(_envelope("SUCCEEDED", result=_result()))

    fake.get_volume_stage = flaky_get
    monkeypatch.setattr(volumes_stage, "_backoff_sleep", lambda attempt: None)
    result = _invoke(*BASE)
    assert result.exit_code == 0, result.output + result.stderr
    assert calls["n"] == 2


def test_explicit_token_is_rejected_in_api_mode(monkeypatch):
    fake = FakeStageClient()
    _patch_client(monkeypatch, fake)
    result = _invoke(*BASE, "-e", "HF_TOKEN=hf_secret_value")
    assert result.exit_code == 1, result.output
    flat = " ".join(result.output.split())
    assert "not yet supported" in flat
    assert "hf_secret_value" not in flat
    assert not fake.admitted  # failed before the POST


def test_ambient_token_is_reported_unused_and_never_sent(monkeypatch):
    fake = FakeStageClient()
    _patch_client(monkeypatch, fake)
    monkeypatch.setenv("HF_TOKEN", "hf_ambient_secret")
    result = _invoke(*BASE, "--output", "json")
    assert result.exit_code == 0, result.output + result.stderr
    stderr = result.stderr
    assert "not used" in " ".join(stderr.split())
    assert "hf_ambient_secret" not in stderr + result.output
    (admitted,) = fake.admitted
    assert "hfToken" not in str(admitted)


def test_namespace_assertion_uses_server_metadata(monkeypatch):
    fake = FakeStageClient()
    _patch_client(monkeypatch, fake)
    result = _invoke(*BASE, "--namespace", "other-ns")
    assert result.exit_code == 1, result.output
    flat = " ".join(result.output.split())
    assert "does not match" in flat
    assert not fake.admitted


def test_kube_context_still_uses_the_operator_path(monkeypatch):
    """The explicit --kube-context fallback keeps driving the #964 kubectl
    path (covered in depth by test_volumes_stage.py); API mode is never
    auto-selected for it."""
    from prime_cli.commands import volumes_stage as vs

    called = {}

    def fake_kubectl_path(**kwargs):
        called["kwargs"] = kwargs
        return {
            "status": "staged",
            "source": kwargs["source"],
            "revision": "sha",
            "requestedRevision": "main",
            "volume": "sft-datasets",
            "clusterId": "cluster-1",
            "namespace": "prime-user-1-local-test",
            "pvcName": "vol-sft-datasets",
            "dataName": "/datasets/tiny-sft",
            "bytes": 1,
            "files": 1,
            "configs": {},
            "kubeContext": "ctx-ok",
            "image": "x",
        }

    fake = FakeStageClient()
    _patch_client(monkeypatch, fake)
    # kubectl presence is the operator path's own precondition; the API
    # patching removed it from PATH, so restore it for this test.
    monkeypatch.setattr(vs.shutil, "which", lambda name: "/usr/local/bin/kubectl")
    monkeypatch.setattr("prime_cli.commands.volumes.stage_dataset", fake_kubectl_path)
    result = _invoke(*BASE, "--kube-context", "ctx-ok")
    assert result.exit_code == 0, result.output + result.stderr
    assert called["kwargs"]["kube_context"] == "ctx-ok"
    assert not fake.admitted  # API mode never ran


def test_help_documents_api_mode_first(monkeypatch):
    result = runner.invoke(app, ["volumes", "stage", "--help", "--plain"], env=TEST_ENV)
    assert result.exit_code == 0
    text = " ".join(result.output.split())
    assert "no kubectl" in text
    assert "--kube-context" in text
