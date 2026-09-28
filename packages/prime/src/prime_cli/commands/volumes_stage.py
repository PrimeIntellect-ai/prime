"""CLI orchestration for ``prime volumes stage``.

Drives one short-lived CPU pod per staging operation through the user's
installed ``kubectl`` and an existing authorized kubeconfig context. The
CLI never talks to Kubernetes directly: kubectl handles kubeconfig,
exec-credential, and request authentication. No RBAC is created, no admin
credentials fetched.

Prerequisite: the caller needs a kubeconfig whose context can read the
volume PVC, create/get/delete pods, read pod logs and (for private
datasets) create/delete Secrets in the volume namespace. This is the
authorized-operator recipe as one command; it is not self-service for
every API-key holder (the spec flags that as a separate GA blocker).

The pod runs the pinned prime-rl image (immutable digest) with the
bundled staging script from ``volumes_stage_script.py`` - no GPU
requests, no model cache, no service-account token, no boot-time pip.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from prime_cli.core import APIError

from ..utils.env_vars import EnvParseError, parse_env_arg, parse_env_file
from ..utils.plain import get_console
from . import volumes_stage_script
from .volumes_stage_script import RESULT_MARKER

# Pinned staging runtime: the prime-rl image from the verified integration
# run (commit-6f4ab3b73), resolved to its immutable OCI index digest, so
# the library stack that stages the data is the same one the trainer
# reads it with. Dataset/library versions at staging time are recorded in
# the on-volume stage manifest.
STAGING_IMAGE = (
    "ghcr.io/primeintellect-ai/prime-rl@sha256:"
    "af4b52b481fa1178228315b9667d3c3aa8836e4a3aff183a84743acee0cb6e88"
)
# Human-readable tag recorded next to the digest for `kubectl` output.
STAGING_IMAGE_REF = "ghcr.io/primeintellect-ai/prime-rl:commit-6f4ab3b73"

REQUEST_TIMEOUT = "30s"
POLL_SECONDS = 2.0
_DELETE_CONFIRM_POLLS = 3
_DELETE_CONFIRM_SECONDS = 1.0
KUBECTL_BINARY = "kubectl"
HF_DATASET_API_URL = "https://huggingface.co/api/datasets/{repo}"

_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")
_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StageError(Exception):
    """Clean CLI failure: message printed verbatim, no stack trace."""


class KubectlError(StageError):
    """kubectl failed. Carries only the command argv and stderr - never
    the stdin payload (the Secret JSON travels over stdin and must not
    leak into error text or logs)."""

    def __init__(self, argv: list[str], returncode: int, stderr: str):
        self.argv = argv
        self.returncode = returncode
        self.stderr = stderr.strip()
        rendered = " ".join(argv[:7])
        detail = f": {self.stderr}" if self.stderr else ""
        super().__init__(f"kubectl failed (exit {returncode}): kubectl {rendered} ...{detail}")


def redact(text: str, token: Optional[str]) -> str:
    """Replace the raw token value anywhere it might appear (streamed pod
    logs, error messages) so a token-echoing failure cannot leak it."""
    if not token:
        return text
    return text.replace(token, "[redacted]")


class _Reporter:
    """Progress lines: stdout in human mode, stderr with --output json."""

    def __init__(self, json_mode: bool, token: Optional[str] = None) -> None:
        self.json_mode = json_mode
        self.token = token

    def __call__(self, message: str, *, dim: bool = False) -> None:
        message = redact(message, self.token)
        if self.json_mode:
            print(message, file=sys.stderr, flush=True)
            return
        console = get_console()
        if dim:
            console.print(f"[dim]{message}[/dim]")
        else:
            console.print(message)


class Kubectl:
    """Thin adapter around the installed kubectl executable.

    Every cluster call is a real argument array (never a shell string),
    carries the resolved ``--context`` explicitly, stays in the volume
    namespace, and is bounded by ``--request-timeout``. JSON manifests
    are piped via stdin, never argv. KUBECONFIG and exec-credential
    plugins are honored by kubectl itself; this never writes the
    kubeconfig or the Prime global config.
    """

    def __init__(self, context: str, namespace: str) -> None:
        self.context = context
        self.namespace = namespace

    def _base(self) -> list[str]:
        return [
            KUBECTL_BINARY,
            "--context",
            self.context,
            "--namespace",
            self.namespace,
            "--request-timeout=" + REQUEST_TIMEOUT,
        ]

    def run(
        self, args: list[str], stdin: Optional[str] = None, timeout: float = 60.0
    ) -> tuple[str, str]:
        argv = self._base() + args
        try:
            completed = subprocess.run(
                argv,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise KubectlError(argv, 124, str(exc) or "timeout") from exc
        except FileNotFoundError as exc:
            raise StageError(
                "kubectl executable not found on PATH. Install kubectl and "
                "configure an authorized kubeconfig context, then retry."
            ) from exc
        if completed.returncode != 0:
            raise KubectlError(argv, completed.returncode, completed.stderr)
        return completed.stdout, completed.stderr

    @staticmethod
    def current_context() -> str:
        try:
            completed = subprocess.run(
                [KUBECTL_BINARY, "config", "current-context"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            raise StageError(
                "could not resolve the current kubeconfig context. Set "
                "--kube-context or KUBECONFIG to an authorized kubeconfig."
            ) from exc
        except FileNotFoundError as exc:
            raise StageError(
                "kubectl executable not found on PATH. Install kubectl and "
                "configure an authorized kubeconfig context, then retry."
            ) from exc
        context = completed.stdout.strip()
        if not context:
            raise StageError(
                "kubeconfig has no current context. Set --kube-context or "
                "switch contexts with kubectl config use-context."
            )
        return context

    # -- resource helpers ---------------------------------------------------

    def get_pvc(self, name: str) -> Optional[dict[str, Any]]:
        try:
            stdout, _ = self.run(["get", "pvc", name, "-o", "json"])
        except KubectlError as exc:
            if "(NotFound)" in exc.stderr or "NotFound" in exc.stderr:
                return None
            raise
        return json.loads(stdout)

    def can_i(self, verb: str, resource: str) -> bool:
        stdout, _ = self.run(["auth", "can-i", verb, resource], timeout=20.0)
        return stdout.strip() == "yes"

    def create_pod(self, manifest: dict[str, Any]) -> dict[str, Any]:
        stdout, _ = self.run(["create", "-f", "-", "-o", "json"], stdin=json.dumps(manifest))
        return json.loads(stdout)

    def create_secret(self, manifest: dict[str, Any]) -> dict[str, Any]:
        stdout, _ = self.run(["create", "-f", "-", "-o", "json"], stdin=json.dumps(manifest))
        return json.loads(stdout)

    def get_pod(self, name: str) -> Optional[dict[str, Any]]:
        try:
            stdout, _ = self.run(["get", "pod", name, "-o", "json"])
        except KubectlError as exc:
            if "(NotFound)" in exc.stderr or "NotFound" in exc.stderr:
                return None
            raise
        return json.loads(stdout)

    def pod_logs(self, name: str) -> str:
        try:
            stdout, _ = self.run(["logs", name])
        except KubectlError:
            # Pending pods have no logs yet (container not started).
            return ""
        return stdout

    def delete_pod(self, name: str) -> None:
        self.run(["delete", "pod", name, "--ignore-not-found", "--wait=false"])

    def delete_secret(self, name: str) -> None:
        self.run(["delete", "secret", name, "--ignore-not-found", "--wait=false"])


def validate_source(source: str) -> str:
    """HF dataset repository ID: 'name' or 'owner/name'. Everything else -
    URLs, hf:// aliases, globs, local paths - is rejected."""
    raw = source.strip()
    lowered = raw.lower()
    if (
        lowered.startswith(("http://", "https://", "hf://", "file://"))
        or lowered.startswith("huggingface.co/")
        or "\\" in raw
    ):
        raise StageError(
            f"invalid dataset source '{source}': give an HF dataset repository "
            "ID ('name' or 'owner/name'), not a URL or alias."
        )
    if Path(raw).exists():
        raise StageError(
            f"'{source}' exists locally. Local upload is not supported; publish "
            "a dataset repository first, then stage the repo ID."
        )
    if not _HF_REPO_ID_RE.match(raw):
        raise StageError(
            f"invalid dataset source '{source}': expected an HF dataset "
            "repository ID ('name' or 'owner/name')."
        )
    return raw


def validate_dataset_path(path: str) -> str:
    """One directory component under datasets/. Exact name preserved; no
    silent sanitization, no absolute paths or traversal targets."""
    if path in (".", "..") or "/" in path or "\\" in path or "\x00" in path:
        raise StageError(
            f"invalid --path '{path}': it must be a single directory component "
            "(no slashes, no traversal, no absolute paths)."
        )
    if not _PATH_RE.match(path):
        raise StageError(
            f"invalid --path '{path}': use 1-128 characters of letters, "
            "digits, '.', '_' or '-', starting with a letter or digit."
        )
    return path


def resolve_hf_token(env_args: list[str], env_files: list[str]) -> Optional[str]:
    """HF token precedence: explicit env args > env files > process HF_TOKEN.
    Only HF_TOKEN is ever forwarded. Unrelated env-file entries are
    ignored; explicit non-HF env args are rejected."""
    token: Optional[str] = os.environ.get("HF_TOKEN")

    for env_file in env_files:
        path = Path(env_file)
        if not path.is_file():
            raise EnvParseError(f"Env file not found: {env_file}")
        entries = parse_env_file(path)
        if "HF_TOKEN" in entries:
            token = entries["HF_TOKEN"]

    for arg in env_args:
        entry = parse_env_arg(arg)
        if "HF_TOKEN" not in entry:
            key = sorted(entry)[0]
            raise StageError(
                f"only HF_TOKEN is forwarded by this command; got '{key}'. "
                "Pass other secrets to the training run instead."
            )
        token = entry["HF_TOKEN"]

    return token


def _get(client: Any, team_id: Optional[str], name: str) -> Any:
    try:
        volumes = client.list_volumes(team_id=team_id)
    except APIError as exc:
        raise StageError(f"could not list volumes: {exc}") from exc
    matches = [v for v in volumes if v.name == name]
    if not matches:
        available = ", ".join(sorted(v.name for v in volumes)) or "none"
        raise StageError(
            f"no volume named '{name}' in the active Prime context (available: {available})."
        )
    if len(matches) > 1:
        raise StageError(
            f"volume name '{name}' is ambiguous ({len(matches)} matches); "
            "disambiguate with the team context."
        )
    volume = matches[0]
    if volume.status != "RUNNING":
        raise StageError(
            f"volume '{name}' is {volume.status}, not RUNNING; staging "
            "requires a bound, running volume."
        )
    return volume


def _preflight(
    kubectl: Kubectl, volume: Any, team_id: Optional[str], needs_secret: bool = False
) -> None:
    """Read the exact PVC, verify ownership labels against API ownership,
    and confirm namespace permissions. No writes; fails closed."""
    pvc = kubectl.get_pvc(volume.pvc_name)
    if pvc is None:
        raise StageError(
            f"PVC '{volume.pvc_name}' was not found in namespace "
            f"'{volume.namespace}'. The volume lives on Prime cluster "
            f"'{volume.cluster_id}' - point --kube-context at that "
            "cluster's authorized kubeconfig and retry."
        )
    if pvc.get("metadata", {}).get("deletionTimestamp"):
        raise StageError(
            f"PVC '{volume.pvc_name}' is terminating (deletionTimestamp set); "
            "staging requires a Bound, non-terminating volume."
        )
    if pvc.get("status", {}).get("phase") != "Bound":
        phase = pvc.get("status", {}).get("phase")
        raise StageError(
            f"PVC '{volume.pvc_name}' is {phase}, not Bound; staging requires a bound volume."
        )
    labels = pvc.get("metadata", {}).get("labels") or {}
    if labels.get("volume-name") != volume.name:
        raise StageError(
            f"PVC '{volume.pvc_name}' does not carry the expected "
            f"volume-name label '{volume.name}' (labels: {labels or 'none'}); "
            "refusing to stage onto a mismatching PVC."
        )
    if team_id:
        if labels.get("team-id") != team_id:
            raise StageError(
                f"PVC '{volume.pvc_name}' is not owned by the active team "
                f"(team-id label: {labels.get('team-id', 'missing')})."
            )
    else:
        owner_label = labels.get("user-id")
        if owner_label is None:
            raise StageError(
                f"PVC '{volume.pvc_name}' has no user-id ownership label; "
                "refusing to stage onto an unowned PVC."
            )
        if volume.created_by is None:
            raise StageError(
                f"PVC '{volume.pvc_name}' carries user-id '{owner_label}' but the "
                "volume's creator is unknown from the API; refusing to stage "
                "onto a PVC whose ownership cannot be verified."
            )
        if owner_label != volume.created_by:
            raise StageError(
                f"PVC '{volume.pvc_name}' user-id label does not match the "
                f"volume creator ({volume.created_by})."
            )

    required = [
        ("create", "pods"),
        ("get", "pods"),
        ("delete", "pods"),
        ("get", "pods/log"),
    ]
    if needs_secret:
        required += [("create", "secrets"), ("delete", "secrets")]
    denied = []
    for verb, resource in required:
        if not kubectl.can_i(verb, resource):
            denied.append(f"{verb} {resource.replace('/', ' ')}")
    if denied:
        raise StageError(
            "this kubeconfig context is missing required permissions in namespace "
            f"'{volume.namespace}': {', '.join(denied)}. Staging needs an "
            "authorized operator kubeconfig for the volume namespace; ask the "
            "volume owner for access. (No RBAC is created by this command.)"
        )


def _dataset_is_public(source: str) -> bool:
    """Anonymous metadata check to decide whether a Secret is needed.

    A repo only counts as public when it is reachable anonymously AND its
    metadata declares it neither private nor gated: gated repos often
    answer HTTP 200 for metadata while requiring a token for files
    (live-verified: bigcode/the-stack returns 200 with "gated": "auto"),
    so metadata 200 alone must never drop a user-supplied token. Network
    errors also default to 'possibly private' (fail-safe); the pod's own
    resolution with the token stays authoritative."""
    import httpx

    try:
        response = httpx.get(HF_DATASET_API_URL.format(repo=source), timeout=15.0)
    except httpx.HTTPError:
        return False
    if response.status_code != 200:
        return False
    try:
        metadata = response.json()
    except ValueError:
        return False
    if not isinstance(metadata, dict):
        return False
    # Absence is not proof of public: only explicit, correctly typed
    # private == False AND gated == False count as provably public;
    # unknown/malformed metadata keeps a user-supplied token.
    return metadata.get("private") is False and metadata.get("gated") is False


def _pod_manifest(
    *,
    namespace: str,
    pvc_name: str,
    source: str,
    revision: str,
    dataset_name: str,
    operation_id: str,
    timeout_seconds: int,
    token_secret: Optional[str],
    pod_name: str,
) -> dict[str, Any]:
    """One CPU container on the pinned prime-rl image: no GPU requests or
    tolerations, no privileged/host mounts, no service-account token,
    Never restart, bounded by activeDeadlineSeconds, PVC at /volume."""
    script_source = inspect.getsource(volumes_stage_script)
    command = [
        "/app/.venv/bin/python",
        "-c",
        script_source,
        "stage",
        "--source",
        source,
        "--revision",
        revision,
        "--dataset-name",
        dataset_name,
        "--volume-root",
        "/volume",
        "--operation-id",
        operation_id,
    ]
    volume_mounts = [{"name": "volume", "mountPath": "/volume"}]
    if token_secret:
        # The token travels as a file, never argv or env. The pod is
        # created first and stays Pending until the Secret exists.
        command += ["--hf-token-file", "/stage-secret/token"]
        volume_mounts.append({"name": "hf-token", "mountPath": "/stage-secret", "readOnly": True})
    volumes: list[dict[str, Any]] = [
        {"name": "volume", "persistentVolumeClaim": {"claimName": pvc_name}}
    ]
    if token_secret:
        volumes.append({"name": "hf-token", "secret": {"secretName": token_secret}})
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/managed-by": "prime-cli",
                "prime-intellect.io/volumes-stage": operation_id,
            },
        },
        "spec": {
            "restartPolicy": "Never",
            "activeDeadlineSeconds": timeout_seconds,
            "automountServiceAccountToken": False,
            "securityContext": {
                "runAsUser": 0,
                "runAsGroup": 0,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [
                {
                    "name": "stage",
                    "image": STAGING_IMAGE,
                    "command": command,
                    "env": [{"name": "PRIME_STAGE_SCRIPT", "value": script_source}],
                    "resources": {
                        "requests": {"cpu": "1", "memory": "2Gi"},
                        "limits": {"cpu": "2", "memory": "8Gi"},
                    },
                    "securityContext": {
                        "runAsUser": 0,
                        "runAsGroup": 0,
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "volumeMounts": volume_mounts,
                }
            ],
            "volumes": volumes,
        },
    }


def _secret_manifest(
    *, namespace: str, name: str, token: str, pod_name: str, pod_uid: str
) -> dict[str, Any]:
    """Token Secret owned by the staging pod, so deletion of the pod
    garbage-collects the Secret even if this CLI dies."""
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "ownerReferences": [
                {
                    "apiVersion": "v1",
                    "kind": "Pod",
                    "name": pod_name,
                    "uid": pod_uid,
                    "controller": False,
                    "blockOwnerDeletion": False,
                }
            ],
        },
        "type": "Opaque",
        "stringData": {"token": token},
    }


def _container_status(pod: dict[str, Any]) -> dict[str, Any]:
    statuses = pod.get("status", {}).get("containerStatuses") or [{}]
    return statuses[0]


def _result_problems(result: dict[str, Any]) -> list[str]:
    """One place validates the in-pod result before it can ever be
    returned or printed as success: identity, status, revision, nested
    verification summary (at least one nonempty split), and numeric
    bytes/files."""
    problems: list[str] = []
    if result.get("status") not in ("staged", "already_staged"):
        problems.append(f"staging result reports an unexpected status ({result.get('status')!r})")
    revision = result.get("revision")
    if not isinstance(revision, str) or not revision:
        problems.append("staging result has no resolved revision")
    configs = result.get("configs")
    if not isinstance(configs, dict) or not configs:
        problems.append("staging result has no verification summary")
    else:
        any_nonempty_split = False
        for config, summary in configs.items():
            splits = summary.get("splits") if isinstance(summary, dict) else None
            if not isinstance(splits, dict) or not splits:
                problems.append(f"staging result config {config!r} has no split summary")
                continue
            for split, info in splits.items():
                rows = info.get("rows") if isinstance(info, dict) else None
                if not isinstance(rows, int) or rows <= 0:
                    problems.append(f"staging result split {split!r} has no row count")
                else:
                    any_nonempty_split = True
        if not any_nonempty_split:
            problems.append("staging result summary lists no nonempty split")
    for field in ("bytes", "files"):
        value = result.get(field)
        if value is not None and not isinstance(value, int):
            problems.append(f"staging result {field} is not a number")
    return problems


def _human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{count} B"


def stage_dataset(
    *,
    client: Any,
    team_id: Optional[str],
    source: str,
    volume_name: str,
    path: Optional[str],
    namespace: str,
    kube_context: Optional[str],
    revision: str,
    token: Optional[str],
    timeout_seconds: int,
    json_mode: bool,
) -> dict[str, Any]:
    """Full orchestration; returns the result object for output rendering.

    Raises StageError for every expected failure (printed without a stack
    trace by the command layer). KeyboardInterrupt cancels: cleanup runs,
    then the exit code is 130.
    """
    reporter = _Reporter(json_mode, token)
    source = validate_source(source)
    dataset_name = validate_dataset_path(path or source.split("/")[-1])

    if shutil.which(KUBECTL_BINARY) is None:
        raise StageError(
            "kubectl executable not found on PATH. Staging drives a "
            "short-lived CPU pod through your authorized kubeconfig; install "
            "kubectl first."
        )

    volume = _get(client, team_id, volume_name)

    if namespace != "auto" and namespace != volume.namespace:
        raise StageError(
            f"--namespace '{namespace}' does not match the volume namespace "
            f"'{volume.namespace}'. The API-returned namespace is authoritative."
        )

    if kube_context:
        context = kube_context
    else:
        # Resolve once; every later call passes --context explicitly, so a
        # concurrent `kubectl config use-context` cannot mid-flight switch
        # the operation onto another cluster.
        context = Kubectl.current_context()
    kubectl = Kubectl(context, volume.namespace)

    reporter(f"Volume {volume.name} · cluster {volume.cluster_id} · kube-context {context}")
    reporter(f"Namespace {volume.namespace} · PVC {volume.pvc_name}")

    operation_id = uuid.uuid4().hex[:10]
    secret_name = f"prime-volume-stage-{operation_id}-hf-token"
    # A user-supplied token is kept through the metadata probe: gated repos
    # can answer HTTP 200 for metadata while requiring a token for files,
    # so only a repo that is provably public drops the token.
    needs_secret = token is not None and not _dataset_is_public(source)

    _preflight(kubectl, volume, team_id, needs_secret=needs_secret)
    reporter("Preflight ok (PVC bound, ownership labels verified, permissions present)")

    # The pod name is explicit and known BEFORE creation: a create whose
    # response is lost (timeout after the API accepted it) still leaves a
    # known name to delete - generateName would not.
    pod_name = f"prime-volume-stage-{operation_id}"
    manifest = _pod_manifest(
        namespace=volume.namespace,
        pvc_name=volume.pvc_name,
        source=source,
        revision=revision,
        dataset_name=dataset_name,
        operation_id=operation_id,
        timeout_seconds=timeout_seconds,
        token_secret=secret_name if needs_secret else None,
        pod_name=pod_name,
    )

    started = time.monotonic()
    logs_offset = [0]

    def _tail_logs() -> str:
        logs = kubectl.pod_logs(pod_name)
        if logs and len(logs) > logs_offset[0]:
            reporter(logs[logs_offset[0] :].rstrip(), dim=True)
            logs_offset[0] = len(logs)
        return logs

    def _owned_secret() -> Optional[str]:
        return secret_name if needs_secret else None

    failure: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    # the polled pod status view; the created pod object stays `pod`
    pod_view: dict[str, Any] = {}
    # The WHOLE pod lifecycle - creation, secret creation, polling, result
    # capture - lives inside the cleanup guard: any failure after the pod
    # manifest was accepted still removes this operation's pod and secret
    # (the ownerReference covers a SIGKILL).
    try:
        created_pod = kubectl.create_pod(manifest)
        reporter(f"Pod {pod_name}")
        reporter(
            f"kubectl --context {context} --namespace {volume.namespace} delete pod "
            f"{pod_name}   # manual cleanup if this command dies",
            dim=True,
        )

        if needs_secret:
            # Created after the pod, via stdin (never argv), owned by the
            # pod: the Pending pod starts as soon as the Secret appears,
            # and the ownerReference garbage-collects the Secret with the
            # pod even if this CLI dies here.
            pod_uid = created_pod.get("metadata", {}).get("uid")
            if not pod_uid:
                raise StageError(
                    f"staging pod {pod_name} was created but its UID could not "
                    "be read; the private-dataset Secret was not created"
                )
            kubectl.create_secret(
                _secret_manifest(
                    namespace=volume.namespace,
                    name=secret_name,
                    token=token or "",
                    pod_name=pod_name,
                    pod_uid=pod_uid,
                )
            )

        while True:
            pod_view = kubectl.get_pod(pod_name) or {}
            if not pod_view:
                failure = (
                    f"staging pod {pod_name} disappeared (evicted or deleted) before completing"
                )
                break
            _tail_logs()
            phase = pod_view.get("status", {}).get("phase")
            if phase == "Pending":
                waiting = _container_status(pod_view).get("state", {}).get("waiting", {})
                if waiting:
                    reporter(
                        f"Pending: {waiting.get('reason', 'unknown')} - "
                        f"{waiting.get('message', '')}",
                        dim=True,
                    )
            if phase == "Succeeded":
                break
            if phase == "Failed":
                state = _container_status(pod_view)
                terminated = state.get("state", {}).get("terminated", {})
                last = state.get("lastState", {}).get("terminated", {})
                done = terminated or last
                failure = (
                    f"staging pod failed: reason {done.get('reason', 'Unknown')}, "
                    f"exit code {done.get('exitCode', '?')}"
                    + (f" ({done.get('message')})" if done.get("message") else "")
                )
                break
            if time.monotonic() - started > timeout_seconds:
                failure = (
                    f"staging did not finish within {timeout_seconds}s; the pod "
                    "may still be running (bare pods have no TTL) - remove it "
                    f"with: kubectl --context {context} --namespace "
                    f"{volume.namespace} delete pod {pod_name}"
                )
                break
            time.sleep(POLL_SECONDS)

        # Capture output before deleting the pod (and its logs).
        logs = _tail_logs()
        result = _result_from_log_text(logs)

        if failure is None:
            terminated = _container_status(pod_view).get("state", {}).get("terminated", {})
            exit_code = terminated.get("exitCode")
            # A real integer zero is required: missing or null exit codes
            # must not be coerced into success.
            if not isinstance(exit_code, int) or isinstance(exit_code, bool):
                failure = "staging pod succeeded but reported no valid container exit code"
            elif exit_code != 0:
                failure = f"staging container exited with code {exit_code}"

        if failure is None:
            if result is None:
                failure = (
                    "staging pod succeeded but produced no structured result "
                    f"({RESULT_MARKER} line missing)"
                )
            elif (
                result.get("operationId") != operation_id
                or result.get("source") != source
                or result.get("datasetName") != dataset_name
            ):
                failure = "staging result does not match this operation/source/destination"
            elif _result_problems(result):
                failure = "; ".join(_result_problems(result)) + "; not treating it as success"
    except KeyboardInterrupt:
        # Cancellation: cleanup runs before exiting with code 130.
        try:
            _cleanup(kubectl, pod_name, _owned_secret(), reporter, operation_id)
        except Exception:  # noqa: BLE001 - best effort during unwind
            pass
        raise
    except BaseException:
        try:
            _cleanup(kubectl, pod_name, _owned_secret(), reporter, operation_id)
        except Exception:  # noqa: BLE001 - best effort during unwind
            pass
        raise

    cleanup_errors = _cleanup(kubectl, pod_name, _owned_secret(), reporter, operation_id)

    if failure is not None:
        raise StageError(redact(failure, token))

    if cleanup_errors:
        # Success requires verified publication AND successful cleanup.
        raise StageError(
            "staged, but cleanup failed - remove the leftovers with the "
            "command above; the command is not a success until they are gone"
        )

    assert result is not None  # a missing/mismatched result raised above
    return {
        "status": str(result["status"]),
        "source": source,
        "revision": result.get("revision"),
        "requestedRevision": revision,
        "volume": volume.name,
        "clusterId": volume.cluster_id,
        "namespace": volume.namespace,
        "pvcName": volume.pvc_name,
        "dataName": f"/datasets/{dataset_name}",
        "bytes": result.get("bytes"),
        "files": result.get("files"),
        "configs": result.get("configs", {}),
        "podName": pod_name,
        "kubeContext": context,
        "image": STAGING_IMAGE_REF,
        "elapsedSeconds": round(time.monotonic() - started, 1),
    }


def _result_from_log_text(logs: str) -> Optional[dict[str, Any]]:
    for line in reversed((logs or "").strip().splitlines()):
        line = line.strip()
        if line.startswith(RESULT_MARKER):
            try:
                payload = json.loads(line[len(RESULT_MARKER) :])
            except ValueError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def _cleanup(
    kubectl: Kubectl,
    pod_name: str,
    secret_name: Optional[str],
    reporter: _Reporter,
    operation_id: Optional[str] = None,
) -> list[str]:
    """Delete exactly this operation's pod and secret. Cleanup errors are
    explicit: the caller turns a non-empty list into a non-zero exit even
    when the data was staged. Never deletes the volume or anything else."""
    errors: list[str] = []
    try:
        kubectl.delete_pod(pod_name)
    except StageError as exc:
        errors.append(f"pod {pod_name}: {exc}")
    else:
        # An accepted delete request is not deletion: bound-verify the pod
        # is actually gone before claiming clean cleanup. ONLY a successful
        # NotFound read confirms deletion; any other read error means the
        # cleanup status is UNKNOWN and must surface, never pass.
        confirmed = False
        for _ in range(_DELETE_CONFIRM_POLLS):
            try:
                if kubectl.get_pod(pod_name) is None:
                    confirmed = True
                    break
            except StageError as exc:
                errors.append(f"pod {pod_name}: deletion could not be confirmed ({exc})")
                break
            time.sleep(_DELETE_CONFIRM_SECONDS)
        if not confirmed and not errors:
            errors.append(f"pod {pod_name}: delete accepted but still present")
    if secret_name:
        try:
            kubectl.delete_secret(secret_name)
        except StageError as exc:
            errors.append(f"secret {secret_name}: {exc}")
    for _ in errors:
        # Two separate commands: `delete pod A secret B` would parse the
        # trailing words as pod names.
        reporter(
            f"kubectl --context {kubectl.context} --namespace "
            f"{kubectl.namespace} delete pod {pod_name}"
        )
        if secret_name:
            reporter(
                f"kubectl --context {kubectl.context} --namespace "
                f"{kubectl.namespace} delete secret {secret_name}"
            )
    return errors


def print_stage_output(output: dict[str, Any], json_mode: bool) -> None:
    """Human recipe output (spec section 4) or one JSON object on stdout."""
    from ..utils import output_data_as_json

    console = get_console()
    if json_mode:
        output_data_as_json(output, console)
        return
    data_name = output["dataName"]
    if output["status"] == "already_staged":
        console.print(f"[green]Already staged {data_name}[/green] · revision {output['revision']}")
    else:
        console.print(
            f"[green]Staged {data_name}[/green] · "
            f"{_human_bytes(int(output['bytes'] or 0))} · revision {output['revision']}"
        )
    configs = output.get("configs") or {}
    splits: list[str] = []
    for config, summary in configs.items():
        split_names = sorted((summary or {}).get("splits", {}))
        console.print(f"Config: {config} · splits: {', '.join(split_names)}")
        splits = split_names
    console.print()
    if len(configs) > 1:
        # A multi-config dataset needs an explicit config selection; do not
        # print a TOML block that silently picks the last listed config.
        console.print("# SFT data block: pick ONE config above and its splits - e.g. for")
        console.print(f'# the "{next(iter(configs))}" config:')
        console.print("[data]")
        console.print(f'name = "{data_name}"')
        console.print(f"# choose splits from: {', '.join(splits)}")
    else:
        console.print("# SFT data block - choose a split listed above:")
        console.print("[data]")
        console.print(f'name = "{data_name}"')
        if splits:
            console.print(f'splits = ["{splits[0]}"]')
    console.print()
    console.print(f"prime train sft.toml --volume {output['volume']}")
    console.print(
        "[dim]Staged datasets are immutable: re-staging the same revision is "
        "idempotent; changed upstream content needs a new --path.[/dim]"
    )
