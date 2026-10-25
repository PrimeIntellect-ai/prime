"""End-to-end: `prime cluster login` → kubectl → `prime auth k8s-token` → apiserver.

Runs the installed `prime` binary and a real `kubectl` against a real
Kubernetes apiserver. Only the Prime platform is stood in for, by a small HTTP
server that serves the two routes the CLI depends on with the platform's
response shapes:

* ``GET  /api/v1/clusters/{name}/kubeconfig-info`` →
  ``{server, certificateAuthorityData, grants: [{pool, namespace}]}``
* ``POST /api/v1/clusters/{name}/kube-token?pool=`` → ExecCredential, minted as
  a real one-hour ServiceAccount token via the TokenRequest API

Opt-in, because it needs a disposable cluster it may create namespaces in:

    kind create cluster --name prime-e2e
    kind get kubeconfig --name prime-e2e > /tmp/admin.yaml
    PRIME_K8S_E2E_ADMIN_KUBECONFIG=/tmp/admin.yaml \\
        uv run pytest tests/test_cluster_kube_e2e.py -v

The tests share one login and run in file order; the revocation test is last
because it permanently removes a grant. Do not run this file under xdist.
"""

import base64
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
import yaml

ADMIN_KUBECONFIG = os.environ.get("PRIME_K8S_E2E_ADMIN_KUBECONFIG")

pytestmark = pytest.mark.skipif(
    not ADMIN_KUBECONFIG or not shutil.which("kubectl"),
    reason="set PRIME_K8S_E2E_ADMIN_KUBECONFIG and install kubectl to run the k8s e2e",
)

CLUSTER = "e2e-k8s"
API_KEY = "pit_e2e_researcher_key"
SA = "researcher"
_SUFFIX = uuid.uuid4().hex[:6]
NS_TRAIN = f"prime-e2e-{_SUFFIX}-train"
NS_EVAL = f"prime-e2e-{_SUFFIX}-eval"
NS_OTHER = f"prime-e2e-{_SUFFIX}-other"
PRIME_BIN = str(Path(sys.executable).parent / "prime")

_ROUTE = re.compile(r"^/api/v1/clusters/([^/]+)/(kubeconfig-info|kube-token)$")


def _admin_kubectl(*args: str, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", "--kubeconfig", ADMIN_KUBECONFIG, *args],
        input=input,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )


def _grant_manifest(namespace: str, with_sa: bool) -> str:
    docs = [{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": namespace}}]
    if with_sa:
        docs += [
            {
                "apiVersion": "v1",
                "kind": "ServiceAccount",
                "metadata": {"name": SA, "namespace": namespace},
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "RoleBinding",
                "metadata": {"name": f"{SA}-edit", "namespace": namespace},
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "ClusterRole",
                    "name": "edit",
                },
                "subjects": [{"kind": "ServiceAccount", "name": SA, "namespace": namespace}],
            },
        ]
    return yaml.safe_dump_all(docs)


class _Platform:
    """Stand-in for the platform's cluster-access routes."""

    def __init__(self, server: str, ca_data: str) -> None:
        # The real platform must answer with the cluster's public endpoint;
        # here that is simply the apiserver this test can reach.
        self.server = server
        self.ca_data = ca_data
        self.api_key = API_KEY
        self.grants = {"train": NS_TRAIN, "eval": NS_EVAL}
        self.rate_limited = False
        self.requests: list[tuple[str, str, str | None]] = []
        self._lock = threading.Lock()

    def revoke(self, pool: str) -> None:
        namespace = self.grants.pop(pool)
        # What the platform's revocation does to the cluster: the
        # ServiceAccount goes, so tokens already issued stop working.
        _admin_kubectl("delete", "serviceaccount", SA, "-n", namespace)

    def mint(self, namespace: str) -> dict:
        token = _admin_kubectl(
            "create", "token", SA, "-n", namespace, "--duration", "1h"
        ).stdout.strip()
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        return {
            "apiVersion": "client.authentication.k8s.io/v1",
            "kind": "ExecCredential",
            "status": {
                "token": token,
                "expirationTimestamp": expiry.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        }

    def handle(self, method: str, raw_path: str, auth: str | None) -> tuple[int, dict, dict]:
        parsed = urlparse(raw_path)
        pool = (parse_qs(parsed.query).get("pool") or [None])[0]
        with self._lock:
            self.requests.append((method, parsed.path, pool))

        if auth != f"Bearer {self.api_key}":
            return 401, {"detail": "Invalid or expired API key"}, {}

        match = _ROUTE.match(parsed.path)
        if not match:
            return 404, {"detail": "Not Found"}, {}
        cluster, route = match.groups()
        if cluster != CLUSTER:
            return 404, {"detail": f"Cluster '{cluster}' not found"}, {}

        if route == "kubeconfig-info" and method == "GET":
            if not self.grants:
                return 404, {"detail": f"No active cluster access grant for '{cluster}'."}, {}
            return (
                200,
                {
                    "server": self.server,
                    "certificateAuthorityData": self.ca_data,
                    "grants": [{"pool": p, "namespace": ns} for p, ns in self.grants.items()],
                },
                {},
            )

        if route == "kube-token" and method == "POST":
            if self.rate_limited:
                return 429, {"detail": "Too many requests"}, {"Retry-After": "7"}
            if pool is None and len(self.grants) > 1:
                pools = ", ".join(sorted(self.grants))
                return (
                    409,
                    {"detail": f"You have access to several pools on '{cluster}' ({pools})."},
                    {},
                )
            if pool is None and len(self.grants) == 1:
                pool = next(iter(self.grants))
            if pool not in self.grants:
                return (
                    403,
                    {
                        "detail": f"No active cluster access grant for '{cluster}' "
                        f"(pool '{pool}'). The grant was revoked or never existed."
                    },
                    {},
                )
            return 200, self.mint(self.grants[pool]), {}

        return 405, {"detail": "Method Not Allowed"}, {}


def _serve(platform: _Platform) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _respond(self) -> None:
            status, body, headers = platform.handle(
                self.command, self.path, self.headers.get("Authorization")
            )
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        do_GET = _respond
        do_POST = _respond

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Env:
    def __init__(self, home: Path, platform: _Platform, base_url: str) -> None:
        self.home = home
        self.platform = platform
        self.base_url = base_url
        self.kubeconfig = home / ".prime" / "kube" / f"{CLUSTER}.yaml"
        self.login_result: subprocess.CompletedProcess | None = None

    def environ(self, **extra: str) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.startswith("PRIME_")}
        env.pop("KUBECONFIG", None)
        env["HOME"] = str(self.home)
        # kubectl resolves the exec `command: prime` through PATH.
        env["PATH"] = f"{Path(PRIME_BIN).parent}{os.pathsep}{env.get('PATH', '')}"
        env.update(extra)
        return env

    def prime(self, *args: str, **extra_env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [PRIME_BIN, *args],
            env=self.environ(PRIME_DISABLE_VERSION_CHECK="1", **extra_env),
            capture_output=True,
            text=True,
            timeout=60,
        )

    def kubectl(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["kubectl", *args],
            env=self.environ(KUBECONFIG=str(self.kubeconfig)),
            capture_output=True,
            text=True,
            timeout=60,
        )

    def exec_plugin(self, context: str, **extra_env: str) -> subprocess.CompletedProcess:
        """Run exactly what kubectl runs for `context`, from the written file."""
        config = yaml.safe_load(self.kubeconfig.read_text())
        user = next(c["context"]["user"] for c in config["contexts"] if c["name"] == context)
        exec_block = next(u["user"]["exec"] for u in config["users"] if u["name"] == user)
        pinned = {item["name"]: item["value"] for item in exec_block["env"]}
        return subprocess.run(
            [shutil.which(exec_block["command"], path=self.environ()["PATH"]), *exec_block["args"]],
            env=self.environ(**{**pinned, **extra_env}),
            capture_output=True,
            text=True,
            timeout=60,
        )


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    admin = json.loads(_admin_kubectl("config", "view", "--raw", "--minify", "-o", "json").stdout)
    cluster_entry = admin["clusters"][0]["cluster"]

    _admin_kubectl("apply", "-f", "-", input=_grant_manifest(NS_TRAIN, with_sa=True))
    _admin_kubectl("apply", "-f", "-", input=_grant_manifest(NS_EVAL, with_sa=True))
    _admin_kubectl("apply", "-f", "-", input=_grant_manifest(NS_OTHER, with_sa=False))

    platform = _Platform(cluster_entry["server"], cluster_entry["certificate-authority-data"])
    server = _serve(platform)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    env = _Env(tmp_path_factory.mktemp("home"), platform, base_url)

    # Configure the CLI the way a researcher would, so the plugin later reads
    # the key from ~/.prime rather than from a PRIME_API_KEY variable.
    assert env.prime("config", "set-base-url", base_url).returncode == 0
    assert env.prime("config", "set-api-key", API_KEY).returncode == 0
    env.login_result = env.prime("cluster", "login", CLUSTER)

    try:
        yield env
    finally:
        server.shutdown()
        for namespace in (NS_TRAIN, NS_EVAL, NS_OTHER):
            subprocess.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    ADMIN_KUBECONFIG,
                    "delete",
                    "namespace",
                    namespace,
                    "--ignore-not-found",
                    "--wait=false",
                ],
                capture_output=True,
                timeout=60,
            )


def test_login_writes_an_exec_only_kubeconfig(e2e):
    result = e2e.login_result
    assert result.returncode == 0, result.stderr + result.stdout
    assert ("GET", f"/api/v1/clusters/{CLUSTER}/kubeconfig-info", None) in e2e.platform.requests

    assert e2e.kubeconfig.stat().st_mode & 0o777 == 0o600
    rendered = e2e.kubeconfig.read_text()
    assert "token:" not in rendered

    config = yaml.safe_load(rendered)
    assert config["clusters"][0]["cluster"]["server"] == e2e.platform.server
    assert {c["name"]: c["context"]["namespace"] for c in config["contexts"]} == {
        f"{CLUSTER}-train": NS_TRAIN,
        f"{CLUSTER}-eval": NS_EVAL,
    }
    exec_block = config["users"][0]["user"]["exec"]
    assert exec_block["args"] == ["auth", "k8s-token", "--cluster", CLUSTER, "--pool", "train"]
    assert {"name": "PRIME_API_BASE_URL", "value": e2e.base_url} in exec_block["env"]


def test_kubectl_authenticates_as_the_granted_service_account(e2e):
    whoami = e2e.kubectl("auth", "whoami", "-o", "jsonpath={.status.userInfo.username}")
    assert whoami.returncode == 0, whoami.stderr
    assert whoami.stdout == f"system:serviceaccount:{NS_TRAIN}:{SA}"

    namespace = e2e.kubectl("config", "view", "--minify", "-o", "jsonpath={..namespace}")
    assert namespace.stdout == NS_TRAIN

    pods = e2e.kubectl("get", "pods")
    assert pods.returncode == 0, pods.stderr
    created = e2e.kubectl("create", "configmap", "e2e-probe", "--from-literal=k=v")
    assert created.returncode == 0, created.stderr

    assert ("POST", f"/api/v1/clusters/{CLUSTER}/kube-token", "train") in e2e.platform.requests


def test_kubectl_is_confined_to_the_grant_namespace(e2e):
    namespaces = e2e.kubectl("get", "ns")
    assert namespaces.returncode != 0
    assert "forbidden" in namespaces.stderr.lower()

    other = e2e.kubectl("get", "pods", "-n", NS_OTHER)
    assert other.returncode != 0
    assert "forbidden" in other.stderr.lower()

    nodes = e2e.kubectl("auth", "can-i", "list", "nodes")
    assert nodes.stdout.strip() == "no"


def test_each_pool_context_mints_for_its_own_pool(e2e):
    whoami = e2e.kubectl(
        "--context",
        f"{CLUSTER}-eval",
        "auth",
        "whoami",
        "-o",
        "jsonpath={.status.userInfo.username}",
    )
    assert whoami.returncode == 0, whoami.stderr
    assert whoami.stdout == f"system:serviceaccount:{NS_EVAL}:{SA}"
    assert ("POST", f"/api/v1/clusters/{CLUSTER}/kube-token", "eval") in e2e.platform.requests


def test_plugin_exit_codes_through_the_real_binary(e2e):
    ok = e2e.exec_plugin(f"{CLUSTER}-train")
    assert ok.returncode == 0, ok.stderr
    credential = json.loads(ok.stdout)
    assert credential["kind"] == "ExecCredential"
    assert credential["status"]["token"]

    ambiguous = e2e.prime("auth", "k8s-token", "--cluster", CLUSTER)
    assert (ambiguous.returncode, ambiguous.stdout) == (5, "")
    assert "several pools" in ambiguous.stderr

    e2e.platform.rate_limited = True
    try:
        limited = e2e.exec_plugin(f"{CLUSTER}-train")
        assert (limited.returncode, limited.stdout) == (3, "")
        via_kubectl = e2e.kubectl("get", "pods")
        assert via_kubectl.returncode != 0
        assert "Retry after 7s" in via_kubectl.stderr
    finally:
        e2e.platform.rate_limited = False

    e2e.platform.api_key = "rotated"
    try:
        expired = e2e.exec_plugin(f"{CLUSTER}-train")
        assert (expired.returncode, expired.stdout) == (4, "")
        assert "prime login" in expired.stderr
    finally:
        e2e.platform.api_key = API_KEY

    unreachable = e2e.exec_plugin(
        f"{CLUSTER}-train", PRIME_API_BASE_URL=f"http://127.0.0.1:{_closed_port()}"
    )
    assert (unreachable.returncode, unreachable.stdout) == (1, "")


def test_revocation_cuts_off_kubectl_and_issued_tokens(e2e, tmp_path):
    issued = json.loads(e2e.exec_plugin(f"{CLUSTER}-eval").stdout)["status"]["token"]

    e2e.platform.revoke("eval")

    refused = e2e.kubectl("--context", f"{CLUSTER}-eval", "get", "pods")
    assert refused.returncode != 0
    assert "No active cluster access grant" in refused.stderr

    plugin = e2e.exec_plugin(f"{CLUSTER}-eval")
    assert (plugin.returncode, plugin.stdout) == (2, "")

    ca_file = tmp_path / "ca.crt"
    ca_file.write_bytes(base64.b64decode(e2e.platform.ca_data))
    empty_kubeconfig = tmp_path / "empty.yaml"
    empty_kubeconfig.write_text("")
    stale = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            str(empty_kubeconfig),
            "--server",
            e2e.platform.server,
            "--certificate-authority",
            str(ca_file),
            "--token",
            issued,
            "auth",
            "whoami",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert stale.returncode != 0
    assert "unauthorized" in stale.stderr.lower()

    relogin = e2e.prime("cluster", "login", CLUSTER)
    assert relogin.returncode == 0, relogin.stderr + relogin.stdout
    contexts = [c["name"] for c in yaml.safe_load(e2e.kubeconfig.read_text())["contexts"]]
    assert contexts == [f"{CLUSTER}-train"]
    still_ok = e2e.kubectl("auth", "whoami", "-o", "jsonpath={.status.userInfo.username}")
    assert still_ok.stdout == f"system:serviceaccount:{NS_TRAIN}:{SA}"
