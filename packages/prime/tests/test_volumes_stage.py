"""Tests for `prime volumes stage` (CLI orchestration).

All Kubernetes interaction goes through a fake kubectl adapter; no test
depends on the developer's live kubeconfig or cluster. The in-pod staging
script has its own suite in test_volumes_stage_script.py.
"""

import json
from typing import Any, Optional

import pytest
from prime_cli.commands import volumes_stage
from prime_cli.commands.volumes_stage_script import RESULT_MARKER
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _flat(text: str) -> str:
    """Collapse rich line-wrapping so multi-word assertions are robust."""
    return " ".join(text.split())


TEST_ENV = {"PRIME_DISABLE_VERSION_CHECK": "1"}

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


def _pod_name(fake: "FakeKubectl") -> str:
    """The staging pod name is explicit; read it off the created manifest."""
    return fake.created_pod_manifests[0]["metadata"]["name"]


def _fake_pvc(labels: Optional[dict[str, str]] = None, phase: str = "Bound") -> dict[str, Any]:
    default = {"volume-name": "sft-datasets", "user-id": "user-1"}
    return {
        "metadata": {"name": "vol-sft-datasets", "labels": labels or default},
        "status": {"phase": phase},
    }


def _pod(status: dict[str, Any], phase: str) -> dict[str, Any]:
    return {
        "metadata": {"name": "prime-volume-stage-op", "uid": "pod-uid-1"},
        "status": {"phase": phase, "containerStatuses": [status]},
    }


def _pending(reason: str = "ContainerCreating") -> dict[str, Any]:
    return {
        "name": "stage",
        "state": {"waiting": {"reason": reason, "message": f"{reason.lower()} detail"}},
    }


def _terminal(phase: str, reason: str = "Completed", exit_code: int = 0) -> dict[str, Any]:
    return {
        "name": "stage",
        "state": {"terminated": {"reason": reason, "exitCode": exit_code}},
    }


def _result_line(operation_id: str, source: str, name: str, status: str = "staged") -> str:
    return (
        RESULT_MARKER
        + json.dumps(
            {
                "status": status,
                "operationId": operation_id,
                "source": source,
                "requestedRevision": "main",
                "revision": "sha123",
                "datasetName": name,
                "bytes": 559_000_000,
                "files": 7,
                "configs": {
                    "default": {
                        "splits": {
                            "train": {"rows": 10, "columns": ["prompt", "completion"]},
                            "math": {"rows": 5, "columns": ["prompt", "completion"]},
                        }
                    }
                },
            }
        )
        + "\n"
    )


def _result_line_factory(source: str, name: str, status: str = "staged"):
    """Build logs from the operation id the CLI actually chose."""

    def factory(operation_id: str) -> str:
        return _result_line(operation_id, source, name, status=status)

    return factory


class FakeKubectl(volumes_stage.Kubectl):
    """Scriptable fake; records every call so tests can assert on argv
    (secret values must never appear in argv)."""

    def __init__(self, context: str, namespace: str) -> None:
        super().__init__(context, namespace)
        self.created_pod_manifests: list[dict[str, Any]] = []
        self.created_secret_manifests: list[dict[str, Any]] = []
        self.argv_log: list[list[str]] = []
        self.deleted: list[tuple[str, str]] = []

    # scenario knobs (class level so tests can configure before invoke)
    pvc: Optional[dict[str, Any]] = None
    phases: list[dict[str, Any]] = []
    logs: str = ""
    can_i_result: bool = True
    fail_pod_delete: bool = False
    current_context_value: str = "ctx-ok"
    raise_in_get_pod: Optional[BaseException] = None

    def run(self, args, stdin=None, timeout=60.0):
        raise AssertionError("raw kubectl.run should not be called in tests")

    @staticmethod
    def current_context() -> str:
        return FakeKubectl.current_context_value

    def get_pvc(self, name):
        pvc = FakeKubectl.pvc
        self.argv_log.append(["get", "pvc", name])
        return pvc

    def can_i(self, verb, resource):
        self.argv_log.append(["auth", "can-i", verb, resource])
        return FakeKubectl.can_i_result

    def create_pod(self, manifest):
        self.argv_log.append(["create", "-f", "-", "-o", "json"])
        self.created_pod_manifests.append(manifest)
        return {"metadata": {"name": manifest["metadata"]["name"], "uid": "pod-uid-1"}}

    def create_secret(self, manifest):
        self.argv_log.append(["create", "-f", "-", "-o", "json"])
        self.created_secret_manifests.append(manifest)
        return {"metadata": {"name": manifest["metadata"]["name"]}}

    def get_pod(self, name):
        self.argv_log.append(["get", "pod", name])
        if FakeKubectl.raise_in_get_pod is not None:
            raise FakeKubectl.raise_in_get_pod
        if ("pod", name) in self.deleted:
            return None
        phases = FakeKubectl.phases
        if not phases:
            raise AssertionError("FakeKubectl ran out of scripted pod phases")
        index = min(sum(1 for a in self.argv_log if a[:2] == ["get", "pod"]), len(phases)) - 1
        return phases[max(0, index)]

    def pod_logs(self, name):
        self.argv_log.append(["logs", name])
        return FakeKubectl.logs

    def delete_pod(self, name):
        self.argv_log.append(["delete", "pod", name, "--ignore-not-found", "--wait=false"])
        if FakeKubectl.fail_pod_delete:
            raise volumes_stage.KubectlError(
                ["kubectl", "--context", self.context, "delete", "pod", name], 1, "boom"
            )
        self.deleted.append(("pod", name))

    def delete_secret(self, name):
        self.argv_log.append(["delete", "secret", name, "--ignore-not-found", "--wait=false"])
        if FakeKubectl.fail_pod_delete:
            raise volumes_stage.KubectlError(
                ["kubectl", "--context", self.context, "delete", "secret", name], 1, "boom"
            )
        self.deleted.append(("secret", name))


@pytest.fixture()
def scenario(monkeypatch):
    """Install the fake kubectl + volume API; return a dict the test can
    tune, plus the fake instances created during the run."""
    state: dict[str, Any] = {"fakes": []}

    class CollectingFake(FakeKubectl):
        def __init__(self, context, namespace):
            super().__init__(context, namespace)
            state["fakes"].append(self)

        def create_pod(self, manifest):
            out = super().create_pod(manifest)
            factory = state.get("result_after_pod")
            if factory is not None:
                op = manifest["metadata"]["labels"]["prime-intellect.io/volumes-stage"]
                FakeKubectl.logs = factory(op)
            return out

    monkeypatch.setattr(volumes_stage, "Kubectl", CollectingFake)
    # fast polls; the real cadence is 2s and is covered by the CLI constant
    monkeypatch.setattr(volumes_stage, "POLL_SECONDS", 0.0)
    monkeypatch.setattr(volumes_stage, "_DELETE_CONFIRM_SECONDS", 0.0)

    from prime_cli.api.training import Volume

    volumes = [Volume.model_validate(VOLUME)]

    class FakeHTC:
        def list_volumes(self, team_id=None):
            state["team_id"] = team_id
            return volumes

    monkeypatch.setattr("prime_cli.commands.volumes._client", lambda: (FakeHTC(), None))
    monkeypatch.setattr(volumes_stage, "_dataset_is_public", lambda source: True)

    def configure(
        *,
        pvc: Optional[dict[str, Any]] = _fake_pvc(),
        phases: Optional[list[dict[str, Any]]] = None,
        logs: str = "",
        can_i: bool = True,
        fail_pod_delete: bool = False,
        raise_in_get_pod: Optional[BaseException] = None,
    ) -> None:
        FakeKubectl.pvc = pvc
        FakeKubectl.can_i_result = can_i
        FakeKubectl.fail_pod_delete = fail_pod_delete
        FakeKubectl.raise_in_get_pod = raise_in_get_pod
        if phases is None:
            phases = [
                _pod(_pending(), "Pending"),
                _pod(_terminal("Succeeded"), "Succeeded"),
            ]
        FakeKubectl.phases = phases
        FakeKubectl.logs = logs

    state["configure"] = configure
    yield state
    FakeKubectl.pvc = None
    FakeKubectl.phases = []
    FakeKubectl.logs = ""
    FakeKubectl.can_i_result = True
    FakeKubectl.fail_pod_delete = False
    FakeKubectl.raise_in_get_pod = None


# ---------------------------------------------------------------------------
# Argument validation (no cluster, no API interaction needed)
# ---------------------------------------------------------------------------


def test_help_mentions_recipe_and_prerequisites() -> None:
    # plain mode avoids rich line-wrapping breaking up the recipe strings
    result = runner.invoke(app, ["volumes", "stage", "--help", "--plain"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    help_text = result.output
    assert "prime volumes stage" in help_text
    assert "prime train sft.toml --volume" in help_text
    assert "/datasets/intellect-3-sft-10k" in help_text
    assert 'splits = ["math"]' in help_text
    assert "kubectl" in help_text


def test_rejects_urls_and_aliases(scenario) -> None:
    for bad in (
        "https://huggingface.co/datasets/acme/ds",
        "hf://datasets/acme/ds",
        "huggingface.co/datasets/acme/ds",
    ):
        result = runner.invoke(
            app,
            ["volumes", "stage", bad, "--volume", "sft-datasets"],
            env=TEST_ENV,
        )
        assert result.exit_code == 1, result.output
        assert "invalid dataset source" in _flat(result.output)
        assert "not a URL or alias" in _flat(result.output)


def test_rejects_existing_local_path(scenario, tmp_path) -> None:
    local = tmp_path / "my-dataset"
    local.mkdir()
    result = runner.invoke(
        app,
        ["volumes", "stage", str(local), "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "Local upload is not supported" in _flat(result.output)
    assert "publish a dataset repository first" in _flat(result.output)


def test_rejects_invalid_dataset_path(scenario) -> None:
    for bad in ("../etc", "a/b", ".", "..", "x\\y", "/abs", "a" * 129):
        result = runner.invoke(
            app,
            [
                "volumes",
                "stage",
                "acme/tiny-sft",
                "--volume",
                "sft-datasets",
                "--path",
                bad,
            ],
            env=TEST_ENV,
        )
        assert result.exit_code == 1, result.output
        assert "invalid --path" in _flat(result.output)


def test_rejects_non_hf_env_arg(scenario) -> None:
    result = runner.invoke(
        app,
        [
            "volumes",
            "stage",
            "acme/tiny-sft",
            "--volume",
            "sft-datasets",
            "-e",
            "WANDB_API_KEY=secret",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "only HF_TOKEN is forwarded" in _flat(result.output)
    assert "WANDB_API_KEY" in _flat(result.output)


def test_rejects_missing_env_file(scenario, tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "volumes",
            "stage",
            "acme/tiny-sft",
            "--volume",
            "sft-datasets",
            "--env-file",
            str(tmp_path / "missing.env"),
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "Env file not found" in _flat(result.output)


# ---------------------------------------------------------------------------
# Token precedence and redaction (unit level + flow level)
# ---------------------------------------------------------------------------


def test_token_precedence_args_over_files_over_process(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / "token.env"
    env_file.write_text("HF_TOKEN=hf_from_file\nOTHER=ignored\n")
    monkeypatch.setenv("HF_TOKEN", "hf_from_process")

    assert volumes_stage.resolve_hf_token([], []) == "hf_from_process"
    assert volumes_stage.resolve_hf_token([], [str(env_file)]) == "hf_from_file"
    assert volumes_stage.resolve_hf_token(["HF_TOKEN=hf_from_arg"], [str(env_file)]) == (
        "hf_from_arg"
    )
    # process env can be referenced bare, like rl.py's -e handling
    assert volumes_stage.resolve_hf_token(["HF_TOKEN"], []) == "hf_from_process"


# ---------------------------------------------------------------------------
# Volume resolution (owner scoping, no cluster writes)
# ---------------------------------------------------------------------------


def test_unknown_volume_fails_without_cluster_calls(scenario) -> None:
    scenario["configure"]()
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "nope"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "no volume named 'nope'" in _flat(result.output)
    assert scenario["fakes"] == []


def test_not_running_volume_fails(scenario, monkeypatch) -> None:
    from prime_cli.api.training import Volume

    stopped = VOLUME.copy()
    stopped["status"] = "CREATING"
    monkeypatch.setattr(
        "prime_cli.commands.volumes._client",
        lambda: (
            type(
                "C",
                (),
                {
                    "list_volumes": staticmethod(
                        lambda team_id=None: [Volume.model_validate(stopped)]
                    )
                },
            )(),
            None,
        ),
    )
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "CREATING" in result.output and "RUNNING" in _flat(result.output)


def test_namespace_mismatch_fails_before_cluster(scenario) -> None:
    scenario["configure"]()
    result = runner.invoke(
        app,
        [
            "volumes",
            "stage",
            "acme/tiny-sft",
            "--volume",
            "sft-datasets",
            "--namespace",
            "other-namespace",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "--namespace 'other-namespace' does not match" in _flat(result.output)
    assert scenario["fakes"] == []


def test_volume_lookup_uses_active_team_scope(scenario) -> None:
    scenario["configure"]()
    runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    # stage ran through; the fake client recorded team_id=None
    assert scenario.get("team_id") is None


# ---------------------------------------------------------------------------
# Preflight: PVC and permissions (fail closed, no pod created)
# ---------------------------------------------------------------------------


def test_missing_pvc_fails_without_creating_pod(scenario) -> None:
    scenario["configure"](pvc=None)
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "was not found in namespace" in _flat(result.output)
    assert "cluster-1" in _flat(result.output)
    fake = scenario["fakes"][0]
    assert fake.created_pod_manifests == []


@pytest.mark.parametrize(
    "labels,message",
    [
        ({"user-id": "user-1"}, "volume-name"),
        ({"volume-name": "sft-datasets"}, "user-id"),
        ({"volume-name": "other", "user-id": "user-1"}, "volume-name"),
        ({"volume-name": "sft-datasets", "user-id": "someone-else"}, "creator"),
    ],
)
def test_pvc_label_mismatch_fails_closed(scenario, labels, message) -> None:
    scenario["configure"](pvc=_fake_pvc(labels=labels))
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert message in _flat(result.output)
    fake = scenario["fakes"][0]
    assert fake.created_pod_manifests == []


def test_unbound_pvc_fails(scenario) -> None:
    scenario["configure"](pvc=_fake_pvc(phase="Pending"))
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "not Bound" in _flat(result.output)


def test_team_label_checked_in_team_context(scenario, monkeypatch) -> None:
    from prime_cli.api.training import Volume

    scenario["configure"](pvc=_fake_pvc())  # user-id labels, no team-id
    monkeypatch.setattr(
        "prime_cli.commands.volumes._client",
        lambda: (
            type(
                "C",
                (),
                {
                    "list_volumes": staticmethod(
                        lambda team_id="t1": [Volume.model_validate(VOLUME)]
                    )
                },
            )(),
            "team-1",
        ),
    )
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "team" in _flat(result.output)


def test_permission_preflight_denied(scenario) -> None:
    scenario["configure"](can_i=False)
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "authorized operator kubeconfig" in _flat(result.output)
    assert "No RBAC is created" in _flat(result.output)
    fake = scenario["fakes"][0]
    assert fake.created_pod_manifests == []


def test_current_context_resolved_once_and_pinned(scenario) -> None:
    """Without --kube-context, the current context is resolved once and
    every cluster call carries it explicitly (immutable selection)."""
    scenario["configure"]()
    FakeKubectl.current_context_value = "ctx-from-kubeconfig"
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output  # no result line scripted
    fake = scenario["fakes"][0]
    assert fake.context == "ctx-from-kubeconfig"
    assert fake.namespace == "prime-user-1-local-test"
    FakeKubectl.current_context_value = "ctx-ok"


def test_explicit_kube_context_used(scenario) -> None:
    scenario["configure"]()
    FakeKubectl.current_context_value = "ctx-wrong"
    result = runner.invoke(
        app,
        [
            "volumes",
            "stage",
            "acme/tiny-sft",
            "--volume",
            "sft-datasets",
            "--kube-context",
            "ctx-ok",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output  # no result line scripted
    fake = scenario["fakes"][0]
    assert fake.context == "ctx-ok"
    FakeKubectl.current_context_value = "ctx-ok"


# ---------------------------------------------------------------------------
# Happy path: staged / already_staged
# ---------------------------------------------------------------------------


def test_public_dataset_stages_without_secret(scenario, monkeypatch) -> None:
    scenario["configure"]()
    scenario["result_after_pod"] = _result_line_factory("acme/tiny-sft", "tiny-sft")
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets", "--output", "json"],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output + "\nstderr: " + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "staged"
    assert payload["source"] == "acme/tiny-sft"
    assert payload["revision"] == "sha123"
    assert payload["volume"] == "sft-datasets"
    assert payload["clusterId"] == "cluster-1"
    assert payload["namespace"] == "prime-user-1-local-test"
    assert payload["pvcName"] == "vol-sft-datasets"
    assert payload["dataName"] == "/datasets/tiny-sft"
    assert payload["bytes"] == 559_000_000
    assert payload["configs"]["default"]["splits"]["math"]["rows"] == 5
    # progress went to stderr only
    assert "PRIME_STAGE_RESULT" not in result.stdout
    assert "Preflight ok" in result.stderr

    fake = scenario["fakes"][0]
    assert fake.created_secret_manifests == []
    assert ("pod", _pod_name(fake)) in fake.deleted
    assert ("secret", None) not in fake.deleted


def test_human_output_prints_recipe(scenario, monkeypatch) -> None:
    scenario["configure"]()
    scenario["result_after_pod"] = _result_line_factory("acme/tiny-sft", "tiny-sft")
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Volume sft-datasets" in out and "cluster cluster-1" in out
    assert "kube-context ctx-ok" in out
    assert "Namespace prime-user-1-local-test" in out
    assert "PVC vol-sft-datasets" in out
    assert "Pod prime-volume-stage-" in out
    assert "Staged /datasets/tiny-sft" in out
    assert "Config: default" in out and "math" in out
    assert 'name = "/datasets/tiny-sft"' in out
    assert 'splits = ["math"]' in out
    assert "prime train sft.toml --volume sft-datasets" in out
    assert "immutable" in out


def test_already_staged_is_idempotent(scenario) -> None:
    scenario["configure"]()
    scenario["result_after_pod"] = _result_line_factory(
        "acme/tiny-sft", "tiny-sft", status="already_staged"
    )
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets", "--output", "json"],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "already_staged"


def test_private_dataset_uses_token_secret_without_leaks(scenario, monkeypatch) -> None:
    scenario["configure"]()
    monkeypatch.setattr(volumes_stage, "_dataset_is_public", lambda source: False)
    token = "hf_secret_token_value_123"
    scenario["result_after_pod"] = _result_line_factory("acme/tiny-sft", "tiny-sft")
    result = runner.invoke(
        app,
        [
            "volumes",
            "stage",
            "acme/tiny-sft",
            "--volume",
            "sft-datasets",
            "-e",
            f"HF_TOKEN={token}",
            "--output",
            "json",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output + "\n" + result.stderr
    fake = scenario["fakes"][0]

    # token only ever appears inside the stdin JSON of `create -f -`
    for argv in fake.argv_log:
        assert token not in " ".join(argv)
    assert token not in result.stdout
    assert token not in result.stderr

    (pod_created,) = fake.created_pod_manifests
    (secret,) = fake.created_secret_manifests
    assert secret["metadata"]["ownerReferences"][0]["uid"] == "pod-uid-1"
    assert secret["metadata"]["ownerReferences"][0]["name"] == _pod_name(fake)
    assert secret["stringData"]["token"] == token
    # the pod references the secret by name but never its value
    secret_names = [
        v["secret"]["secretName"] for v in pod_created["spec"]["volumes"] if "secret" in v
    ]
    assert secret_names == [secret["metadata"]["name"]]
    # cleanup removes both
    assert ("pod", _pod_name(fake)) in fake.deleted
    assert ("secret", secret["metadata"]["name"]) in fake.deleted


# ---------------------------------------------------------------------------
# Pod hardening invariants (zero GPUs, no SA token, no model cache)
# ---------------------------------------------------------------------------


def test_pod_manifest_is_cpu_only_and_unprivileged(scenario, monkeypatch) -> None:
    scenario["configure"]()
    scenario["result_after_pod"] = _result_line_factory("acme/tiny-sft", "tiny-sft")
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    (pod,) = scenario["fakes"][0].created_pod_manifests
    spec = pod["spec"]
    assert pod["metadata"]["name"].startswith("prime-volume-stage-")
    assert pod["metadata"]["labels"]["prime-intellect.io/volumes-stage"]
    assert spec["restartPolicy"] == "Never"
    assert spec["automountServiceAccountToken"] is False
    assert spec["activeDeadlineSeconds"] == 3600
    (container,) = spec["containers"]
    assert container["command"][0] == "/app/.venv/bin/python"
    assert container["image"].startswith("ghcr.io/primeintellect-ai/prime-rl@sha256:")
    resources = container["resources"]
    assert "nvidia.com/gpu" not in json.dumps(resources)
    assert resources["requests"] == {"cpu": "1", "memory": "2Gi"}
    assert resources["limits"] == {"cpu": "2", "memory": "8Gi"}
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"] == {"drop": ["ALL"]}
    assert spec["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
    volumes = {v["name"] for v in spec["volumes"]}
    assert volumes == {"volume"}
    mount_paths = {m["mountPath"] for m in container["volumeMounts"]}
    assert mount_paths == {"/volume"}
    (pvc_volume,) = spec["volumes"]
    assert pvc_volume["persistentVolumeClaim"]["claimName"] == "vol-sft-datasets"
    env_names = [e["name"] for e in container["env"]]
    assert env_names == ["PRIME_STAGE_SCRIPT"]
    # no trainer offline/cache env inherited
    assert "HF_HOME" not in env_names and "HF_HUB_OFFLINE" not in env_names


# ---------------------------------------------------------------------------
# Failure modes during pod observation
# ---------------------------------------------------------------------------


def _run(scenario, monkeypatch, phases, logs="", fail_pod_delete=False, extra_args=None):
    scenario["configure"](phases=phases, logs=logs, fail_pod_delete=fail_pod_delete)
    args = ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"] + (extra_args or [])
    return runner.invoke(app, args, env=TEST_ENV)


def test_image_pull_failure_is_surfaced(scenario, monkeypatch) -> None:
    phases = [
        _pod(_pending("ErrImagePull"), "Pending"),
        _pod(
            {"name": "stage", "state": {"waiting": {"reason": "ImagePullBackOff"}}},
            "Pending",
        ),
        _pod(
            {
                "name": "stage",
                "state": {"waiting": {"reason": "ImagePullBackOff"}},
                "lastState": {"terminated": {"reason": "ImagePullBackOff", "exitCode": 1}},
            },
            "Failed",
        ),
    ]
    result = _run(scenario, monkeypatch, phases, logs="downloading ...\n")
    assert result.exit_code == 1, result.output
    assert "ImagePullBackOff" in _flat(result.output)
    fake = scenario["fakes"][0]
    assert ("pod", _pod_name(fake)) in fake.deleted


def test_nonzero_exit_is_failure(scenario, monkeypatch) -> None:
    phases = [_pod(_terminal("Failed", "Error", exit_code=3), "Failed")]
    result = _run(scenario, monkeypatch, phases, logs="boom\n")
    assert result.exit_code == 1, result.output
    assert "exit code 3" in _flat(result.output)


def test_oom_is_failure(scenario, monkeypatch) -> None:
    phases = [_pod(_terminal("Failed", "OOMKilled", exit_code=137), "Failed")]
    result = _run(scenario, monkeypatch, phases)
    assert result.exit_code == 1, result.output
    assert "OOMKilled" in _flat(result.output)


def test_missing_result_marker_is_failure(scenario, monkeypatch) -> None:
    phases = [_pod(_terminal("Succeeded"), "Succeeded")]
    result = _run(scenario, monkeypatch, phases, logs="all done, trust me\n")
    assert result.exit_code == 1, result.output
    assert "no structured result" in _flat(result.output)


def test_forged_result_is_rejected(scenario, monkeypatch) -> None:
    scenario["configure"]()
    # result line for a *different* operation id
    phases = [_pod(_terminal("Succeeded"), "Succeeded")]
    logs = _result_line("someone-elses-op", "acme/tiny-sft", "tiny-sft")
    result = _run(scenario, monkeypatch, phases, logs=logs)
    assert result.exit_code == 1, result.output
    assert "does not match this operation" in _flat(result.output)


def test_mismatched_source_in_result_is_rejected(scenario) -> None:
    scenario["configure"]()
    scenario["result_after_pod"] = _result_line_factory("evil/other-repo", "tiny-sft")
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "does not match this operation" in _flat(result.output)


def test_evicted_pod_is_failure(scenario, monkeypatch) -> None:
    phases = [None]  # pod disappears (evicted / deleted mid-run)
    result = _run(scenario, monkeypatch, phases)
    assert result.exit_code == 1, result.output
    assert "disappeared (evicted or deleted)" in _flat(result.output)


def test_deadline_times_out_and_reports_publication_state(scenario, monkeypatch) -> None:
    # pod never reaches a terminal phase; timeout is 0 seconds
    phases = [_pod(_pending(), "Pending")]
    result = _run(scenario, monkeypatch, phases, extra_args=["--timeout-seconds", "0"])
    assert result.exit_code == 1, result.output
    assert "did not finish within 0s" in _flat(result.output)
    assert "may still be running" in _flat(result.output)
    fake = scenario["fakes"][0]
    assert ("pod", _pod_name(fake)) in fake.deleted


def test_ctrl_c_cleans_up_and_exits_130(scenario, monkeypatch) -> None:
    scenario["configure"]()
    monkeypatch.setattr(volumes_stage, "POLL_SECONDS", 0.0)
    FakeKubectl.raise_in_get_pod = KeyboardInterrupt()
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 130, result.output
    fake = scenario["fakes"][0]
    assert ("pod", _pod_name(fake)) in fake.deleted
    FakeKubectl.raise_in_get_pod = None


def test_cleanup_failure_is_not_green(scenario) -> None:
    scenario["configure"]()
    scenario["result_after_pod"] = _result_line_factory("acme/tiny-sft", "tiny-sft")
    FakeKubectl.fail_pod_delete = True
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    fake = scenario["fakes"][0]
    assert "staged, but cleanup failed" in _flat(result.output)
    assert "kubectl --context ctx-ok" in _flat(result.output)
    assert f"delete pod {_pod_name(fake)}" in _flat(result.output)


def test_secret_creation_failure_fails_the_run(scenario, monkeypatch) -> None:
    scenario["configure"]()
    monkeypatch.setattr(volumes_stage, "_dataset_is_public", lambda source: False)

    created: list[FakeKubectl] = []

    class FailingSecretFake(FakeKubectl):
        def __init__(self, context: str, namespace: str) -> None:
            super().__init__(context, namespace)
            created.append(self)

        def create_secret(self, manifest):
            raise volumes_stage.KubectlError(
                ["kubectl", "create", "-f", "-", "-o", "json"],
                1,
                "forbidden: cannot create secrets",
            )

    monkeypatch.setattr(volumes_stage, "Kubectl", FailingSecretFake)
    result = runner.invoke(
        app,
        [
            "volumes",
            "stage",
            "acme/tiny-sft",
            "--volume",
            "sft-datasets",
            "-e",
            "HF_TOKEN=hf_x",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "kubectl failed" in _flat(result.output)
    assert "forbidden: cannot create secrets" in _flat(result.output)
    # the orphaned pod was cleaned up
    assert ("pod", _pod_name(created[0])) in created[0].deleted


def test_hf_confusion_message_not_found_or_private(scenario, monkeypatch) -> None:
    """A clean pod failure quoting the in-pod 'not found or private'
    message is surfaced verbatim."""
    phases = [_pod(_terminal("Failed", "Error", exit_code=1), "Failed")]
    logs = (
        "could not resolve dataset 'acme/missing' at revision 'main' "
        "(not found or private/inaccessible; set HF_TOKEN / --env-file, and "
        "accept gated-dataset terms if it is gated)\n"
    )
    result = _run(scenario, monkeypatch, phases, logs=logs)
    assert result.exit_code == 1, result.output
    assert "not found or private/inaccessible" in _flat(result.output)
    assert "HF_TOKEN" in _flat(result.output)


# ---------------------------------------------------------------------------
# JSON compatibility + volumes regressions
# ---------------------------------------------------------------------------


def test_invalid_output_format_rejected(scenario) -> None:
    result = runner.invoke(
        app,
        [
            "volumes",
            "stage",
            "acme/tiny-sft",
            "--volume",
            "sft-datasets",
            "--output",
            "yaml",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "Invalid output format" in _flat(result.output)


def test_volumes_subcommands_still_work() -> None:
    commands = (
        ["create", "x", "--help"],
        ["list", "--help"],
        ["resize", "--help"],
        ["delete", "--help"],
    )
    for command in commands:
        result = runner.invoke(app, ["volumes"] + command, env=TEST_ENV)
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Roast-driven regressions (t033 probes, converted)
# ---------------------------------------------------------------------------


REAL_DATASET_IS_PUBLIC = volumes_stage._dataset_is_public


def test_gated_metadata_keeps_supplied_token(monkeypatch) -> None:
    """Gated repos answer HTTP 200 for metadata while needing a token for
    files (live-verified: bigcode/the-stack, gated 'auto'). Metadata 200
    must not suppress a user-supplied token."""
    import httpx

    def response(payload):
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **kw: response({"id": "acme/gated", "private": False, "gated": "auto"}),
    )
    assert volumes_stage._dataset_is_public("acme/gated") is False

    monkeypatch.setattr(
        httpx, "get", lambda *a, **kw: response({"id": "acme/private", "private": True})
    )
    assert volumes_stage._dataset_is_public("acme/private") is False

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **kw: response({"id": "acme/open", "private": False, "gated": False}),
    )
    assert volumes_stage._dataset_is_public("acme/open") is True

    def boom(*a, **kw):
        raise httpx.ConnectError("network down")

    monkeypatch.setattr(httpx, "get", boom)
    assert volumes_stage._dataset_is_public("acme/unreachable") is False


def test_private_flow_keeps_token_when_gated_metadata_is_200(scenario, monkeypatch) -> None:
    """End-to-end: a supplied token still produces the pod-owned Secret
    when the anonymous metadata probe answers 200 with gated metadata."""
    import httpx

    scenario["configure"]()
    monkeypatch.setattr(volumes_stage, "_dataset_is_public", REAL_DATASET_IS_PUBLIC)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **kw: httpx.Response(
            200, json={"id": "acme/gated", "private": False, "gated": "auto"}
        ),
    )
    token = "hf_gated_token_xyz"
    scenario["result_after_pod"] = _result_line_factory("acme/tiny-sft", "tiny-sft")

    result = runner.invoke(
        app,
        [
            "volumes",
            "stage",
            "acme/tiny-sft",
            "--volume",
            "sft-datasets",
            "-e",
            f"HF_TOKEN={token}",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    fake = scenario["fakes"][0]
    assert len(fake.created_secret_manifests) == 1
    assert fake.created_secret_manifests[0]["stringData"]["token"] == token


def test_secret_rbac_preflight_fails_closed(scenario, monkeypatch) -> None:
    scenario["configure"]()
    monkeypatch.setattr(volumes_stage, "_dataset_is_public", lambda source: False)
    scenario["result_after_pod"] = _result_line_factory("acme/tiny-sft", "tiny-sft")
    FakeKubectl.can_i_result = False  # no permissions at all
    FakeKubectl.can_i_secret_result = False

    created: list[FakeKubectl] = []

    class DenySecretsFake(FakeKubectl):
        def __init__(self, context: str, namespace: str) -> None:
            super().__init__(context, namespace)
            created.append(self)

        def can_i(self, verb, resource):
            self.argv_log.append(["auth", "can-i", verb, resource])
            if resource == "secrets":
                return False
            return True

    monkeypatch.setattr(volumes_stage, "Kubectl", DenySecretsFake)
    result = runner.invoke(
        app,
        [
            "volumes",
            "stage",
            "acme/tiny-sft",
            "--volume",
            "sft-datasets",
            "-e",
            "HF_TOKEN=hf_x",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    fake = created[0]
    # both secret permissions probed BEFORE the pod was created
    calls = [call for call in fake.argv_log if call[:2] == ["auth", "can-i"]]
    assert ["auth", "can-i", "create", "secrets"] in calls
    assert ["auth", "can-i", "delete", "secrets"] in calls
    assert fake.created_pod_manifests == []
    assert "missing required permissions" in _flat(result.output)
    assert "create secrets" in _flat(result.output)
    assert "delete secrets" in _flat(result.output)


def test_terminating_pvc_fails(scenario) -> None:
    pvc = _fake_pvc()
    pvc["metadata"]["deletionTimestamp"] = "2026-09-28T00:00:00Z"
    scenario["configure"](pvc=pvc)
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "terminating" in _flat(result.output)
    assert scenario["fakes"][0].created_pod_manifests == []


def test_unknown_personal_owner_fails_closed(scenario, monkeypatch) -> None:
    from prime_cli.api.training import Volume

    scenario["configure"]()
    no_creator = VOLUME.copy()
    no_creator["createdBy"] = None
    monkeypatch.setattr(
        "prime_cli.commands.volumes._client",
        lambda: (
            type(
                "C",
                (),
                {
                    "list_volumes": staticmethod(
                        lambda team_id=None: [Volume.model_validate(no_creator)]
                    )
                },
            )(),
            None,
        ),
    )
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "creator is unknown from the API" in _flat(result.output)
    assert scenario["fakes"][0].created_pod_manifests == []


def test_empty_owner_label_fails_closed(scenario) -> None:
    scenario["configure"](pvc=_fake_pvc(labels={"volume-name": "sft-datasets", "user-id": ""}))
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output


def test_failed_result_status_is_never_success(scenario) -> None:
    scenario["configure"]()
    scenario["result_after_pod"] = _result_line_factory(
        "acme/tiny-sft", "tiny-sft", status="failed"
    )
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "unexpected status ('failed')" in _flat(result.output)


def test_result_without_configs_is_rejected(scenario) -> None:
    scenario["configure"]()

    def factory(op: str) -> str:
        payload = json.loads(_result_line(op, "acme/tiny-sft", "tiny-sft")[len(RESULT_MARKER) :])
        payload.pop("configs")
        return RESULT_MARKER + json.dumps(payload)

    scenario["result_after_pod"] = factory
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "no verification summary" in _flat(result.output)


def test_missing_container_exit_code_is_failure(scenario) -> None:
    phases = [
        {
            "metadata": {"name": "prime-volume-stage-op", "uid": "pod-uid-1"},
            "status": {"phase": "Succeeded"},
        }
    ]
    scenario["configure"](phases=phases, logs=_result_line("op", "acme/tiny-sft", "tiny-sft"))
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "no valid container exit code" in _flat(result.output)


def test_interrupt_right_after_pod_reporter_cleans_up(scenario, monkeypatch) -> None:
    scenario["configure"]()

    def interrupting_reporter(self, message, *, dim=False):
        if message.startswith("Pod "):
            raise KeyboardInterrupt()

    monkeypatch.setattr(volumes_stage._Reporter, "__call__", interrupting_reporter)
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 130, result.output
    fake = scenario["fakes"][0]
    assert any(kind == "pod" for kind, _ in fake.deleted)


def test_failed_pod_create_still_cleans_known_name(scenario, monkeypatch) -> None:
    scenario["configure"]()

    created: list[FakeKubectl] = []

    class FailingCreateFake(FakeKubectl):
        def __init__(self, context: str, namespace: str) -> None:
            super().__init__(context, namespace)
            created.append(self)

        def create_pod(self, manifest):
            self.argv_log.append(["create", "-f", "-", "-o", "json"])
            self.created_pod_manifests.append(manifest)
            raise volumes_stage.KubectlError(
                ["kubectl", "create", "-f", "-", "-o", "json"], 1, "timeout after create"
            )

    monkeypatch.setattr(volumes_stage, "Kubectl", FailingCreateFake)
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    fake = created[0]
    # the explicit pre-computed pod name is deleted even though create failed
    assert ("pod", _pod_name(fake)) in fake.deleted


def test_token_echo_in_logs_is_redacted(scenario, monkeypatch) -> None:
    """A pod failure that echoes the token (e.g. an HF auth error carrying
    the request header) must never reach the user's console."""
    scenario["configure"]()
    token = "hf_leaked_secret_token"
    monkeypatch.setattr(volumes_stage, "_dataset_is_public", lambda source: False)

    def factory(op: str) -> str:
        return (
            "resolved acme/tiny-sft@main -> sha1\n"
            "HTTP 401 for token hf_leaked_secret_token\n"
            + _result_line(op, "acme/tiny-sft", "tiny-sft")
        )

    scenario["result_after_pod"] = factory
    result = runner.invoke(
        app,
        [
            "volumes",
            "stage",
            "acme/tiny-sft",
            "--volume",
            "sft-datasets",
            "-e",
            f"HF_TOKEN={token}",
            "--output",
            "json",
        ],
        env=TEST_ENV,
    )
    assert token not in result.stdout
    assert token not in result.stderr
    assert "[redacted]" in result.stderr
    # and the human-mode failure path redacts too
    result2 = runner.invoke(
        app,
        [
            "volumes",
            "stage",
            "acme/tiny-sft",
            "--volume",
            "sft-datasets",
            "-e",
            f"HF_TOKEN={token}",
        ],
        env=TEST_ENV,
    )
    assert token not in result2.output


def test_multi_config_output_does_not_pick_last_config(scenario) -> None:
    scenario["configure"]()

    def factory(op: str) -> str:
        payload = json.loads(_result_line(op, "acme/tiny-sft", "tiny-sft")[len(RESULT_MARKER) :])
        payload["configs"] = {
            "default": {"splits": {"train": {"rows": 2, "columns": ["a"]}}},
            "tools": {"splits": {"tool_calls": {"rows": 3, "columns": ["b"]}}},
        }
        return RESULT_MARKER + json.dumps(payload)

    scenario["result_after_pod"] = factory
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets"],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert "pick ONE config above" in out
    # no fabricated ready-to-run splits line for multi-config datasets
    assert "splits = [" not in out


def test_help_recipe_path_matches_toml_name() -> None:
    """The recipe must be internally consistent: the stage example's
    --path (or the unchanged repo basename when no --path is given) must
    equal the name written in the TOML block."""

    from prime_cli.commands import volumes as volumes_module

    docstring = volumes_module.stage.__doc__
    stage_line = next(line for line in docstring.splitlines() if "prime volumes stage " in line)
    name_line = next(line for line in docstring.splitlines() if 'name = "/datasets/' in line)
    toml_name = name_line.split('name = "')[1].split('"')[0]
    if "--path" in stage_line:
        staged_path = stage_line.split("--path")[1].split()[0]
    else:
        source = stage_line.split("prime volumes stage ")[1].split()[0]
        staged_path = volumes_stage.validate_dataset_path(source.split("/")[-1])
    assert f"/datasets/{staged_path}" == toml_name


# ---------------------------------------------------------------------------
# Re-review probes (t033 rereview-040fff6f), converted
# ---------------------------------------------------------------------------


def test_cleanup_confirmation_transport_failure_is_not_success() -> None:
    """A get_pod error during deletion confirmation means UNKNOWN
    cleanup, not confirmed: only a NotFound read proves the pod is gone."""

    class Kube:
        context = "ctx"
        namespace = "ns"

        def delete_pod(self, name):
            pass

        def get_pod(self, name):
            raise volumes_stage.StageError("API transport timeout")

    errors = volumes_stage._cleanup(Kube(), "pod", None, lambda message: None, "op")
    assert errors, "unknown deletion status reported as cleaned"
    assert "could not be confirmed" in errors[0]


def test_missing_public_metadata_flags_keeps_token(monkeypatch) -> None:
    """HTTP 200 metadata without explicit private/gated flags is NOT
    proof of public: absence keeps the user-supplied token."""
    import httpx

    monkeypatch.setattr(
        httpx, "get", lambda *a, **kw: httpx.Response(200, json={"id": "acme/data"})
    )
    assert volumes_stage._dataset_is_public("acme/data") is False

    # the same holds for partially-declared metadata
    monkeypatch.setattr(
        httpx, "get", lambda *a, **kw: httpx.Response(200, json={"id": "acme/x", "private": False})
    )
    assert volumes_stage._dataset_is_public("acme/x") is False
    # malformed metadata also keeps the token
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(200, json="not-a-dict"))
    assert volumes_stage._dataset_is_public("acme/y") is False


@pytest.mark.parametrize("bad_field", ["null_exit", "empty_config_splits", "non_numeric_bytes"])
def test_invalid_result_schema_fails(scenario, bad_field) -> None:
    """A malformed result must never be a success: null exit codes, config
    summaries without split details, and nonnumeric byte counts all fail."""
    terminal = _terminal("Succeeded")
    if bad_field == "null_exit":
        terminal["state"]["terminated"]["exitCode"] = None
    scenario["configure"](phases=[_pod(terminal, "Succeeded")])

    def factory(operation):
        raw = _result_line(operation, "acme/tiny-sft", "tiny-sft")
        payload = json.loads(raw[len(RESULT_MARKER) :])
        if bad_field == "empty_config_splits":
            payload["configs"] = {"default": {}}
        if bad_field == "non_numeric_bytes":
            payload["bytes"] = "not-an-integer"
        return RESULT_MARKER + json.dumps(payload)

    scenario["result_after_pod"] = factory
    result = runner.invoke(
        app,
        ["volumes", "stage", "acme/tiny-sft", "--volume", "sft-datasets", "--output", "json"],
        env=TEST_ENV,
    )
    assert result.exit_code != 0, result.output + "\n" + result.stderr
    if bad_field == "null_exit":
        assert "exit code" in _flat(result.output + result.stderr)
