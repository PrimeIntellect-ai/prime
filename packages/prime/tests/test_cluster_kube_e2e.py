"""End-to-end: `prime cluster login` → kubectl → `prime auth k8s-token` → apiserver.

Runs the installed `prime` binary and a real `kubectl` against a real
Kubernetes apiserver. Two things are stood in for:

* The Prime platform, by a small HTTP server that serves the two routes the
  CLI depends on with the platform's response shapes:

  * ``GET  /api/v1/clusters/{name}/kubeconfig-info`` →
    ``{cluster, certificateAuthorityData, grants: [{pool, namespace, server,
    tlsServerName, unavailableReason}]}``
  * ``POST /api/v1/clusters/{name}/kube-token?pool=`` → ExecCredential, minted
    as a real one-hour ServiceAccount token via the TokenRequest API

* Each pool's Tailscale API proxy, by a plain TCP pass-through per pool on its
  own port. Like the real proxy it does not terminate TLS, so kubectl only
  connects if it verifies the apiserver's own certificate under
  ``tls-server-name: kubernetes.default.svc`` (a SAN kind and k3s both issue).

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
import ssl
import subprocess
import sys
import threading
import uuid
from contextlib import suppress
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

TLS_SERVER_NAME = "kubernetes.default.svc"
PENDING_REASON = "This reservation's API proxy has not joined your tailnet yet; try again shortly."

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


class _PoolProxy:
    """Stand-in for one pool's Tailscale API proxy: raw TCP to the apiserver.

    TLS passes straight through, so the certificate kubectl sees is the
    apiserver's, issued for in-cluster names rather than for this address.
    It listens on ``::1`` where it can: kind's apiserver certificate does not
    cover that address, so — just as with a real proxy's MagicDNS name —
    kubectl can only connect by verifying against ``tls-server-name``.
    """

    def __init__(self, upstream: tuple[str, int]) -> None:
        self.upstream = upstream
        self.connections = 0
        try:
            self._listener = socket.create_server(("::1", 0), family=socket.AF_INET6)
            self.host = "::1"
            self.url = f"https://[::1]:{self._listener.getsockname()[1]}"
        except OSError:
            self._listener = socket.create_server(("127.0.0.1", 0))
            self.host = "127.0.0.1"
            self.url = f"https://127.0.0.1:{self._listener.getsockname()[1]}"
        self.port = self._listener.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            self.connections += 1
            try:
                upstream = socket.create_connection(self.upstream, timeout=10)
            except OSError:
                client.close()
                continue
            upstream.settimeout(None)
            for src, dst in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pipe, args=(src, dst), daemon=True).start()

    @staticmethod
    def _pipe(src: socket.socket, dst: socket.socket) -> None:
        with suppress(OSError):
            while chunk := src.recv(65536):
                dst.sendall(chunk)
        for sock in (src, dst):
            with suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)

    def close(self) -> None:
        self._listener.close()


class _Platform:
    """Stand-in for the platform's cluster-access routes."""

    def __init__(self, apiserver: str, ca_data: str) -> None:
        parsed = urlparse(apiserver)
        assert parsed.hostname and parsed.port, apiserver
        self.apiserver = apiserver
        self.ca_data = ca_data
        self.api_key = API_KEY
        self.grants = {"train": NS_TRAIN, "eval": NS_EVAL}
        # One proxy per pool, as on the real tailnet: nothing is shared.
        self.proxies = {pool: _PoolProxy((parsed.hostname, parsed.port)) for pool in self.grants}
        # A grant the platform knows about but whose proxy is not up yet.
        self.pending = {"debug": f"prime-e2e-{_SUFFIX}-debug"}
        # What an older platform returned at the top level. It points at a
        # closed port so that any kubeconfig built from it cannot connect.
        self.legacy_server = f"https://127.0.0.1:{_closed_port()}"
        self.rate_limited = False
        self.unavailable: str | None = None
        self.requests: list[tuple[str, str, str | None]] = []
        self._lock = threading.Lock()

    def kubeconfig_info(self, cluster: str) -> dict:
        grants = [
            {
                "pool": pool,
                "namespace": namespace,
                "server": self.proxies[pool].url,
                "tlsServerName": TLS_SERVER_NAME,
                "unavailableReason": None,
            }
            for pool, namespace in self.grants.items()
        ] + [
            {
                "pool": pool,
                "namespace": namespace,
                "server": None,
                "tlsServerName": None,
                "unavailableReason": PENDING_REASON,
            }
            for pool, namespace in self.pending.items()
        ]
        return {
            "cluster": cluster,
            "server": self.legacy_server,
            "certificateAuthorityData": self.ca_data,
            "grants": grants,
        }

    def close(self) -> None:
        for proxy in self.proxies.values():
            proxy.close()

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
            return 200, self.kubeconfig_info(cluster), {}

        if route == "kube-token" and method == "POST":
            if self.rate_limited:
                return 429, {"detail": "Too many requests"}, {"Retry-After": "7"}
            if self.unavailable:
                return 503, {"detail": self.unavailable}, {}
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
        platform.close()
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
    # One cluster entry per pool, each pointing at that pool's own proxy, and
    # never at the legacy top-level server or the apiserver itself.
    assert {c["name"]: c["cluster"] for c in config["clusters"]} == {
        f"{CLUSTER}-{pool}": {
            "server": e2e.platform.proxies[pool].url,
            "certificate-authority-data": e2e.platform.ca_data,
            "tls-server-name": TLS_SERVER_NAME,
        }
        for pool in ("train", "eval")
    }
    assert e2e.platform.legacy_server not in rendered
    assert e2e.platform.apiserver not in rendered
    assert {c["name"]: c["context"]["namespace"] for c in config["contexts"]} == {
        f"{CLUSTER}-train": NS_TRAIN,
        f"{CLUSTER}-eval": NS_EVAL,
    }
    assert {c["name"]: c["context"]["cluster"] for c in config["contexts"]} == {
        f"{CLUSTER}-train": f"{CLUSTER}-train",
        f"{CLUSTER}-eval": f"{CLUSTER}-eval",
    }
    # The grant whose proxy has not joined yet is reported, not written.
    output = " ".join((result.stdout + result.stderr).split())
    assert f"Skipping pool 'debug': {PENDING_REASON}" in output
    assert "debug" not in rendered
    exec_block = config["users"][0]["user"]["exec"]
    assert exec_block["args"] == ["auth", "k8s-token", "--cluster", CLUSTER, "--pool", "train"]
    assert {"name": "PRIME_API_BASE_URL", "value": e2e.base_url} in exec_block["env"]


def test_kubectl_authenticates_as_the_granted_service_account(e2e):
    before = e2e.platform.proxies["train"].connections
    whoami = e2e.kubectl("auth", "whoami", "-o", "jsonpath={.status.userInfo.username}")
    assert whoami.returncode == 0, whoami.stderr
    assert whoami.stdout == f"system:serviceaccount:{NS_TRAIN}:{SA}"
    # It got there through the train pool's proxy.
    assert e2e.platform.proxies["train"].connections > before

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


def _cert_covers(proxy: _PoolProxy, ca_data: str, name: str) -> bool:
    """Whether the apiserver certificate behind `proxy` is valid for `name`."""
    context = ssl.create_default_context(cadata=base64.b64decode(ca_data).decode())
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    with socket.create_connection((proxy.host, proxy.port), timeout=10) as raw:
        try:
            with context.wrap_socket(raw, server_hostname=name):
                return True
        except ssl.SSLCertVerificationError:
            return False


def _kubectl_with(e2e, tmp_path, tls_server_name: str | None) -> subprocess.CompletedProcess:
    config = yaml.safe_load(e2e.kubeconfig.read_text())
    for entry in config["clusters"]:
        entry["cluster"].pop("tls-server-name", None)
        if tls_server_name:
            entry["cluster"]["tls-server-name"] = tls_server_name
    edited = tmp_path / f"edited-{tls_server_name}.yaml"
    edited.write_text(yaml.safe_dump(config))
    return subprocess.run(
        ["kubectl", "get", "pods"],
        env=e2e.environ(KUBECONFIG=str(edited)),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_tls_is_verified_against_the_apiserver_name(e2e, tmp_path):
    # The proxy passes TLS through, so the name the apiserver certificate is
    # checked against comes from `tls-server-name`. Prove kubectl really uses
    # it: the same kubeconfig with a different name must fail verification.
    wrong = _kubectl_with(e2e, tmp_path, "not-the-apiserver.invalid")
    assert wrong.returncode != 0
    assert "not-the-apiserver.invalid" in wrong.stderr
    assert "certificate" in wrong.stderr.lower()

    # And without the field, kubectl checks the proxy's own address, which a
    # real tailnet proxy's certificate never covers. Only assert that where
    # this apiserver's certificate does not cover the stand-in proxy either
    # (kind on ::1); k3s also issues for ::1 and 127.0.0.1.
    proxy = e2e.platform.proxies["train"]
    assert _cert_covers(proxy, e2e.platform.ca_data, TLS_SERVER_NAME)
    if not _cert_covers(proxy, e2e.platform.ca_data, proxy.host):
        missing = _kubectl_with(e2e, tmp_path, None)
        assert missing.returncode != 0
        assert "certificate" in missing.stderr.lower()


def test_each_pool_context_mints_for_its_own_pool(e2e):
    train_before = e2e.platform.proxies["train"].connections
    eval_before = e2e.platform.proxies["eval"].connections
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
    # Each context dials its own pool's proxy, never another pool's.
    assert e2e.platform.proxies["eval"].connections > eval_before
    assert e2e.platform.proxies["train"].connections == train_before


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

    e2e.platform.unavailable = "Cluster 'e2e-k8s' has not reported a heartbeat recently."
    try:
        down = e2e.exec_plugin(f"{CLUSTER}-train")
        assert (down.returncode, down.stdout) == (1, "")
        assert "has not reported a heartbeat recently" in down.stderr
        via_kubectl = e2e.kubectl("get", "pods")
        assert via_kubectl.returncode != 0
        assert "has not reported a heartbeat recently" in via_kubectl.stderr
    finally:
        e2e.platform.unavailable = None

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
    # Through the surviving pool's proxy, the way a researcher would hold on
    # to an old token.
    proxy_url = e2e.platform.proxies["train"].url

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
            proxy_url,
            "--tls-server-name",
            TLS_SERVER_NAME,
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
