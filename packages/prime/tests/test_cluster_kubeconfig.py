"""Coverage for `prime cluster login` and the `prime auth k8s-token` plugin.

The plugin's contract is unusually strict because kubectl, not a human, is the
caller: stdout is protocol, and the exit code is how anything wrapping it tells
"retry later" from "this will never work". Those are the properties worth
pinning down — a regression in them shows up as a confusing kubectl error
rather than a test failure anywhere else.
"""

import json
import re

import httpx
import pytest
import yaml
from prime_cli.commands.auth import (
    EXIT_AMBIGUOUS,
    EXIT_AUTH_EXPIRED,
    EXIT_FORBIDDEN,
    EXIT_RATE_LIMITED,
    EXIT_UNREACHABLE,
)
from prime_cli.commands.auth import app as auth_app
from prime_cli.commands.cluster import _build_kubeconfig
from typer.testing import CliRunner

runner = CliRunner()


class _FakeConfig:
    api_key = "test-key"
    base_url = "https://api.example.com"


@pytest.fixture
def patch_config(monkeypatch):
    monkeypatch.setattr("prime_cli.commands.auth.Config", lambda: _FakeConfig())


def respond(monkeypatch, *, status_code, json_body=None, headers=None):
    requested_urls = []

    def _post(url, **kwargs):
        requested_urls.append(url)
        return httpx.Response(
            status_code=status_code,
            json=json_body if json_body is not None else {},
            headers=headers or {},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("prime_cli.commands.auth.httpx.post", _post)
    return requested_urls


def _grant(pool, *, server=None, tls_server_name="kubernetes.default.svc", reason=None):
    """A kubeconfig-info grant. Reachable by default, through its own proxy."""
    if server is None and reason is None:
        server = f"https://k8s-{pool}.tail1234.ts.net"
    return {
        "pool": pool,
        "namespace": f"ada-{pool}",
        "server": server,
        "tlsServerName": tls_server_name if server else None,
        "unavailableReason": reason,
    }


class TestKubeconfigRendering:
    def test_exec_block_carries_no_token(self, monkeypatch):
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        config = _build_kubeconfig(
            cluster="alpha-cluster",
            ca_data="Y2E=",
            grants=[_grant("alpha")],
            base_url="https://api.example.com",
        )
        rendered = yaml.safe_dump(config)
        # No credential material anywhere in the file — the only "token" that
        # may appear is the plugin's own subcommand name.
        assert "token:" not in rendered
        assert "client-certificate" not in rendered
        exec_block = config["users"][0]["user"]["exec"]
        assert exec_block["command"] == "prime"
        assert exec_block["args"][:2] == ["auth", "k8s-token"]
        # Never prompt: kubectl may be running with no terminal attached.
        assert exec_block["interactiveMode"] == "Never"

    def test_exec_args_carry_no_context_flag_by_default(self, monkeypatch):
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        config = _build_kubeconfig(
            cluster="c1",
            ca_data="Y2E=",
            grants=[_grant("alpha")],
            base_url="https://api.example.com",
        )
        args = config["users"][0]["user"]["exec"]["args"]
        assert args == ["auth", "k8s-token", "--cluster", "c1", "--pool", "alpha"]

    def test_exec_args_preserve_the_prime_context(self, monkeypatch):
        # `prime --context staging cluster login` must write a kubeconfig whose
        # refreshes also run against staging — kubectl invokes the plugin with
        # no PRIME_CONTEXT in its environment, so the flag has to be in the
        # file itself.
        monkeypatch.setenv("PRIME_CONTEXT", "staging")
        config = _build_kubeconfig(
            cluster="c1",
            ca_data="Y2E=",
            grants=[_grant("alpha")],
            base_url="https://api.example.com",
        )
        args = config["users"][0]["user"]["exec"]["args"]
        assert args == [
            "--context",
            "staging",
            "auth",
            "k8s-token",
            "--cluster",
            "c1",
            "--pool",
            "alpha",
        ]

    def test_exec_env_pins_the_resolved_base_url(self, monkeypatch):
        # The --context flag alone regressed custom API URLs: reloading the
        # built-in production context at refresh time forces base_url back to
        # the public default. The login-time URL therefore rides in the exec
        # env, where it outranks context resolution.
        monkeypatch.setenv("PRIME_CONTEXT", "production")
        config = _build_kubeconfig(
            cluster="c1",
            ca_data="Y2E=",
            grants=[_grant("alpha")],
            base_url="https://api.internal.example.com",
        )
        exec_block = config["users"][0]["user"]["exec"]
        assert {"name": "PRIME_API_BASE_URL", "value": "https://api.internal.example.com"} in (
            exec_block["env"]
        )
        # The context intent is preserved alongside, not replaced.
        assert exec_block["args"][:2] == ["--context", "production"]

    def test_one_context_per_pool(self, monkeypatch):
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        config = _build_kubeconfig(
            cluster="c1",
            ca_data="Y2E=",
            grants=[_grant("alpha"), _grant("beta")],
            base_url="https://api.example.com",
        )
        assert [c["name"] for c in config["contexts"]] == ["c1-alpha", "c1-beta"]
        assert config["contexts"][0]["context"]["namespace"] == "ada-alpha"
        assert config["current-context"] == "c1-alpha"
        # Each context's plugin invocation names its own pool, otherwise the
        # server would refuse the ambiguous request on every kubectl call.
        assert config["users"][1]["user"]["exec"]["args"][-1] == "beta"

    def test_each_pool_gets_its_own_server_and_a_shared_ca(self, monkeypatch):
        # Every pool has its own API proxy on the researcher's tailnet, so
        # there is no cluster-wide server: each context must point at the
        # cluster entry for its own proxy.
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        config = _build_kubeconfig(
            cluster="c1",
            ca_data="Y2E=",
            grants=[_grant("alpha"), _grant("beta")],
            base_url="https://api.example.com",
        )
        assert config["clusters"] == [
            {
                "name": "c1-alpha",
                "cluster": {
                    "server": "https://k8s-alpha.tail1234.ts.net",
                    "certificate-authority-data": "Y2E=",
                    "tls-server-name": "kubernetes.default.svc",
                },
            },
            {
                "name": "c1-beta",
                "cluster": {
                    "server": "https://k8s-beta.tail1234.ts.net",
                    "certificate-authority-data": "Y2E=",
                    "tls-server-name": "kubernetes.default.svc",
                },
            },
        ]
        assert {c["name"]: c["context"]["cluster"] for c in config["contexts"]} == {
            "c1-alpha": "c1-alpha",
            "c1-beta": "c1-beta",
        }

    def test_tls_server_name_is_omitted_when_the_platform_sends_none(self, monkeypatch):
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        grant = {**_grant("alpha"), "tlsServerName": None}
        config = _build_kubeconfig(
            cluster="c1", ca_data="Y2E=", grants=[grant], base_url="https://api.example.com"
        )
        assert "tls-server-name" not in config["clusters"][0]["cluster"]

    def test_exec_env_disables_the_update_banner(self, monkeypatch):
        # kubectl forwards plugin stderr on every call; an update notice there
        # would sit above the output of every kubectl command.
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        config = _build_kubeconfig(
            cluster="c1",
            ca_data="Y2E=",
            grants=[_grant("alpha")],
            base_url="https://api.example.com",
        )
        env = config["users"][0]["user"]["exec"]["env"]
        assert {"name": "PRIME_DISABLE_VERSION_CHECK", "value": "1"} in env


class _FakeAPIClient:
    base_url = "https://api.example.com"
    payload: dict = {}
    error: Exception | None = None
    requested: list = []

    def get(self, endpoint):
        type(self).requested.append(endpoint)
        if type(self).error is not None:
            raise type(self).error
        return type(self).payload


class TestClusterLogin:
    """`prime cluster login` through the root CLI, with the platform's
    `kubeconfig-info` response stubbed at the client boundary."""

    INFO = {
        "cluster": "c1",
        "certificateAuthorityData": "Y2E=",
        "grants": [_grant("train"), _grant("eval")],
    }

    @pytest.fixture
    def login(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        monkeypatch.setattr("prime_cli.commands.cluster.KUBECONFIG_DIR", tmp_path / "kube")
        monkeypatch.setattr("prime_cli.commands.cluster.APIClient", _FakeAPIClient)
        _FakeAPIClient.payload = dict(self.INFO)
        _FakeAPIClient.error = None
        _FakeAPIClient.requested = []

        def _run(*args):
            from prime_cli.main import app as root_app

            return runner.invoke(root_app, ["cluster", "login", *args])

        return _run

    def test_writes_a_private_kubeconfig_from_kubeconfig_info(self, login, tmp_path):
        result = login("c1")

        assert result.exit_code == 0, result.output
        assert _FakeAPIClient.requested == ["/clusters/c1/kubeconfig-info"]
        path = tmp_path / "kube" / "c1.yaml"
        assert path.stat().st_mode & 0o777 == 0o600
        config = yaml.safe_load(path.read_text())
        assert config["clusters"][1] == {
            "name": "c1-eval",
            "cluster": {
                "server": "https://k8s-eval.tail1234.ts.net",
                "certificate-authority-data": "Y2E=",
                "tls-server-name": "kubernetes.default.svc",
            },
        }
        assert [c["name"] for c in config["contexts"]] == ["c1-train", "c1-eval"]
        assert config["contexts"][1]["context"] == {
            "cluster": "c1-eval",
            "user": "c1-eval",
            "namespace": "ada-eval",
        }
        # Atomic write: nothing but the kubeconfig is left behind.
        assert [p.name for p in path.parent.iterdir()] == ["c1.yaml"]

    def test_top_level_server_and_api_url_never_reach_the_kubeconfig(self, login, tmp_path):
        # An older platform sends a cluster-wide `server`; researchers must
        # only ever go through their pool's tailnet API proxy.
        _FakeAPIClient.payload = {
            **self.INFO,
            "server": "https://k8s.public.example.com:6443",
            "apiUrl": "https://100.64.0.10:6443",
        }

        result = login("c1")

        assert result.exit_code == 0, result.output
        rendered = (tmp_path / "kube" / "c1.yaml").read_text()
        assert "k8s.public.example.com" not in rendered
        assert "100.64.0.10" not in rendered
        servers = [c["cluster"]["server"] for c in yaml.safe_load(rendered)["clusters"]]
        assert servers == [
            "https://k8s-train.tail1234.ts.net",
            "https://k8s-eval.tail1234.ts.net",
        ]

    def test_old_platform_shape_writes_nothing(self, login, tmp_path):
        # Pre-#7204 grants have no `server`; the top-level one must not be
        # used as a fallback.
        _FakeAPIClient.payload = {
            "server": "https://k8s.public.example.com:6443",
            "certificateAuthorityData": "Y2E=",
            "grants": [{"pool": "train", "namespace": "ada-train"}],
        }

        result = login("c1")

        assert result.exit_code == 1
        assert "Skipping pool 'train'" in result.output
        assert not (tmp_path / "kube").exists() or not any((tmp_path / "kube").iterdir())

    def test_unavailable_grants_are_skipped_with_their_reason(self, login, tmp_path):
        reason = "This reservation's API proxy has not joined your tailnet yet; try again shortly."
        _FakeAPIClient.payload = {
            **self.INFO,
            "grants": [_grant("train"), _grant("eval", reason=reason)],
        }

        result = login("c1")

        assert result.exit_code == 0, result.output
        # Rich wraps long lines; compare on normalized whitespace.
        output = " ".join(result.output.split())
        assert f"Skipping pool 'eval': {reason}" in output
        config = yaml.safe_load((tmp_path / "kube" / "c1.yaml").read_text())
        assert [c["name"] for c in config["contexts"]] == ["c1-train"]
        assert [c["name"] for c in config["clusters"]] == ["c1-train"]
        assert [u["name"] for u in config["users"]] == ["c1-train"]
        assert config["current-context"] == "c1-train"

    def test_relogin_drops_grants_that_are_gone_or_unavailable(self, login, tmp_path):
        _FakeAPIClient.payload = {
            **self.INFO,
            "grants": [_grant("train"), _grant("eval"), _grant("debug")],
        }
        assert login("c1").exit_code == 0

        _FakeAPIClient.payload = {
            **self.INFO,
            "grants": [_grant("train", reason="Proxy is restarting."), _grant("eval")],
        }
        result = login("c1")

        assert result.exit_code == 0, result.output
        config = yaml.safe_load((tmp_path / "kube" / "c1.yaml").read_text())
        assert [c["name"] for c in config["contexts"]] == ["c1-eval"]
        assert [c["name"] for c in config["clusters"]] == ["c1-eval"]
        assert config["current-context"] == "c1-eval"

    def test_no_reachable_grant_fails_and_keeps_the_previous_file(self, login, tmp_path):
        assert login("c1").exit_code == 0
        path = tmp_path / "kube" / "c1.yaml"
        before = path.read_text()

        _FakeAPIClient.payload = {
            **self.INFO,
            "grants": [_grant("train", reason="Not joined yet."), _grant("eval", reason="Gone.")],
        }
        result = login("c1")

        assert result.exit_code == 1
        output = " ".join(result.output.split())
        assert "Skipping pool 'train': Not joined yet." in output
        assert "Skipping pool 'eval': Gone." in output
        assert "None of your pools on 'c1' is reachable yet" in output
        assert path.read_text() == before
        assert [p.name for p in path.parent.iterdir()] == ["c1.yaml"]

    def test_no_reachable_grant_on_first_login_writes_nothing(self, login, tmp_path):
        _FakeAPIClient.payload = {**self.INFO, "grants": [_grant("train", reason="Not yet.")]}

        result = login("c1")

        assert result.exit_code == 1
        assert not (tmp_path / "kube" / "c1.yaml").exists()

    def test_failed_write_leaves_the_previous_file_and_no_temp(self, login, tmp_path, monkeypatch):
        assert login("c1").exit_code == 0
        path = tmp_path / "kube" / "c1.yaml"
        before = path.read_text()

        def _boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr("prime_cli.commands.cluster.os.replace", _boom)
        _FakeAPIClient.payload = {**self.INFO, "grants": [_grant("other")]}
        result = login("c1")

        assert result.exit_code == 1
        assert "disk full" in result.output
        assert path.read_text() == before
        assert [p.name for p in path.parent.iterdir()] == ["c1.yaml"]

    def test_missing_ca_fails_cleanly(self, login, tmp_path):
        _FakeAPIClient.payload = {
            k: v for k, v in self.INFO.items() if k != "certificateAuthorityData"
        }

        result = login("c1")

        assert result.exit_code == 1
        assert "incomplete connection info" in result.output
        assert not (tmp_path / "kube" / "c1.yaml").exists()

    def test_no_grants_writes_nothing(self, login, tmp_path):
        _FakeAPIClient.payload = {**self.INFO, "grants": []}

        result = login("c1")

        assert result.exit_code == 1
        assert "No active access grant" in result.output
        assert not (tmp_path / "kube" / "c1.yaml").exists()

    def test_platform_refusal_is_reported(self, login, tmp_path):
        from prime_cli.core.client import NotFoundError

        _FakeAPIClient.error = NotFoundError("HTTP 404: No active grant for 'c1'")

        result = login("c1")

        assert result.exit_code == 1
        assert "No active grant" in result.output
        assert not (tmp_path / "kube" / "c1.yaml").exists()


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class TestRootWiring:
    def test_auth_k8s_token_is_a_real_subcommand(self):
        # Every generated kubeconfig runs exactly `prime auth k8s-token`.
        from prime_cli.main import app as root_app

        result = runner.invoke(root_app, ["auth", "k8s-token", "--help"])
        assert result.exit_code == 0
        assert "--cluster" in _ANSI.sub("", result.output)

    def test_cluster_login_is_a_real_subcommand(self):
        from prime_cli.main import app as root_app

        result = runner.invoke(root_app, ["cluster", "login", "--help"])
        assert result.exit_code == 0
        assert "CLUSTER" in _ANSI.sub("", result.output)


class TestCustomBaseUrlSurvivesContextRefresh:
    """The regression this pins down: with `PRIME_CONTEXT=production` (the
    baked-in --context flag), every kubectl refresh reloaded the built-in
    production environment and forced base_url back to the public default —
    ignoring a custom URL set via `prime config set-base-url`. The exec env's
    PRIME_API_BASE_URL must outrank that reload."""

    @pytest.fixture
    def custom_url_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("PRIME_API_BASE_URL", raising=False)
        monkeypatch.delenv("PRIME_BASE_URL", raising=False)
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        from prime_cli.core import Config

        config = Config()
        config.set_base_url("https://api.internal.example.com")
        return tmp_path

    def test_production_context_alone_loses_the_custom_url(self, custom_url_home, monkeypatch):
        # The failure mode being fixed, kept as documentation: context
        # resolution at refresh time discards the custom URL.
        from prime_cli.core import Config

        monkeypatch.setenv("PRIME_CONTEXT", "production")
        assert Config().base_url == "https://api.primeintellect.ai"

    def test_exec_env_base_url_outranks_the_production_context(self, custom_url_home, monkeypatch):
        # What actually happens at refresh time now: kubectl exports the exec
        # env block, so the plugin's Config resolves the pinned URL even while
        # --context production reloads the built-in environment.
        from prime_cli.core import Config

        monkeypatch.setenv("PRIME_CONTEXT", "production")
        monkeypatch.setenv("PRIME_API_BASE_URL", "https://api.internal.example.com")
        assert Config().base_url == "https://api.internal.example.com"


class TestCredentialPluginSuccess:
    def test_writes_only_the_credential_to_stdout(self, patch_config, monkeypatch):
        credential = {
            "apiVersion": "client.authentication.k8s.io/v1",
            "kind": "ExecCredential",
            "status": {"token": "abc", "expirationTimestamp": "2026-07-25T07:00:00Z"},
        }
        respond(monkeypatch, status_code=200, json_body=credential)

        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])

        assert result.exit_code == 0
        # kubectl parses stdout as JSON — anything else on it breaks auth.
        assert json.loads(result.stdout) == credential

    def test_posts_to_the_public_api_route(self, patch_config, monkeypatch):
        # Config.base_url strips /api/v1, so the plugin must add it back —
        # this URL shipped without the prefix once and every call 404'd.
        urls = respond(monkeypatch, status_code=200, json_body={"status": {}})

        runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])

        assert urls == ["https://api.example.com/api/v1/clusters/c1/kube-token"]

    def test_forwards_the_pool(self, patch_config, monkeypatch):
        seen = {}

        def _post(url, **kwargs):
            seen.update(kwargs)
            return httpx.Response(200, json={"status": {}}, request=httpx.Request("POST", url))

        monkeypatch.setattr("prime_cli.commands.auth.httpx.post", _post)
        runner.invoke(auth_app, ["k8s-token", "--cluster", "c1", "--pool", "train"])

        assert seen["params"] == {"pool": "train"}
        assert seen["headers"]["Authorization"] == "Bearer test-key"


class TestCredentialPluginFailures:
    """Each status maps to a distinct exit code so a caller can tell whether
    retrying is pointless."""

    def test_revoked_grant_is_permanent(self, patch_config, monkeypatch):
        respond(
            monkeypatch,
            status_code=403,
            json_body={"detail": "No active cluster access grant"},
        )
        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])
        assert result.exit_code == EXIT_FORBIDDEN
        assert result.stdout.strip() == ""
        # The platform's reason beats anything reconstructed client-side.
        assert "No active cluster access grant" in result.stderr

    def test_expired_platform_auth_points_at_prime_login(self, patch_config, monkeypatch):
        respond(monkeypatch, status_code=401)
        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])
        assert result.exit_code == EXIT_AUTH_EXPIRED

    def test_rate_limited_reports_retry_after(self, patch_config, monkeypatch):
        respond(monkeypatch, status_code=429, headers={"Retry-After": "42"})
        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])
        assert result.exit_code == EXIT_RATE_LIMITED

    def test_rate_limited_without_retry_after_still_exits_cleanly(self, patch_config, monkeypatch):
        respond(monkeypatch, status_code=429)
        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])
        assert result.exit_code == EXIT_RATE_LIMITED

    def test_ambiguous_pool_is_its_own_code(self, patch_config, monkeypatch):
        respond(
            monkeypatch,
            status_code=409,
            json_body={"detail": "You have access to several pools (alpha, beta)."},
        )
        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])
        assert result.exit_code == EXIT_AMBIGUOUS

    def test_unavailable_cluster_is_transient_and_shows_the_reason(self, patch_config, monkeypatch):
        # 503 means "not ready / stale heartbeat / capability withdrawn":
        # retryable, and the platform's reason is the useful part.
        respond(
            monkeypatch,
            status_code=503,
            json_body={"detail": "Cluster 'c1' has not reported a heartbeat recently."},
        )
        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])
        assert result.exit_code == EXIT_UNREACHABLE
        assert result.stdout.strip() == ""
        assert "has not reported a heartbeat recently" in result.stderr

    def test_unavailable_cluster_without_detail_still_explains(self, patch_config, monkeypatch):
        respond(monkeypatch, status_code=503)
        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])
        assert result.exit_code == EXIT_UNREACHABLE
        assert result.stdout.strip() == ""
        assert "temporarily unavailable" in result.stderr

    def test_other_server_errors_are_transient(self, patch_config, monkeypatch):
        respond(monkeypatch, status_code=500, json_body={"detail": "Traceback ..."})
        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])
        assert result.exit_code == EXIT_UNREACHABLE
        assert result.stdout.strip() == ""
        assert "Prime platform error (500)" in result.stderr

    def test_unreachable_platform_is_transient(self, patch_config, monkeypatch):
        def _post(url, **kwargs):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr("prime_cli.commands.auth.httpx.post", _post)
        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])
        assert result.exit_code == EXIT_UNREACHABLE

    def test_malformed_credential_is_not_passed_through(self, patch_config, monkeypatch):
        # A 200 with the wrong shape must not reach kubectl as if it were valid.
        respond(monkeypatch, status_code=200, json_body={"nope": True})
        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])
        assert result.exit_code == EXIT_UNREACHABLE
        assert result.stdout.strip() == ""

    def test_missing_api_key_does_not_call_the_platform(self, monkeypatch):
        class _NoKey:
            api_key = ""
            base_url = "https://api.example.com"

        monkeypatch.setattr("prime_cli.commands.auth.Config", lambda: _NoKey())

        def _explode(*args, **kwargs):
            raise AssertionError("should not have made a request")

        monkeypatch.setattr("prime_cli.commands.auth.httpx.post", _explode)
        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])
        assert result.exit_code == EXIT_AUTH_EXPIRED
