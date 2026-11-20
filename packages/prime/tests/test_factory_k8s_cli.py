"""Coverage for `prime factory k8s login` and the `prime auth k8s-token` plugin.

The plugin's contract is unusually strict because kubectl, not a human, is the
caller: stdout is protocol, and the exit code is how anything wrapping it tells
"retry later" from "this will never work". Those are the properties worth
pinning down — a regression in them shows up as a confusing kubectl error
rather than a test failure anywhere else.
"""

import itertools
import json
import os
import re
import socket
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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
from prime_cli.commands.factory_k8s import _build_kubeconfig
from prime_cli.utils import kube_token_cache
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def token_cache_dir(tmp_path, monkeypatch):
    """Keep every test's credential cache out of the real ~/.prime, and make
    each plugin run look like it came from a new kubectl process unless a test
    says otherwise."""
    cache = tmp_path / "kube" / "cache"
    monkeypatch.setattr("prime_cli.utils.kube_token_cache.CACHE_DIR", cache)
    pids = itertools.count(1000)
    monkeypatch.setattr("prime_cli.commands.auth.os.getppid", lambda: next(pids))
    return cache


@pytest.fixture(autouse=True)
def proxy_names_resolve(monkeypatch):
    """No test reaches a real resolver: every proxy name resolves unless a
    test says otherwise."""
    looked_up = []
    monkeypatch.setattr("prime_cli.commands.factory_k8s._resolve", looked_up.append)
    return looked_up


class _FakeConfig:
    api_key = "test-key"
    base_url = "https://api.example.com"


@pytest.fixture(autouse=True)
def login_fingerprint(monkeypatch):
    """Plugin runs carry the fingerprint the login wrote for _FakeConfig's key."""
    fingerprint = kube_token_cache.key_fingerprint(
        base_url=_FakeConfig.base_url, api_key=_FakeConfig.api_key
    )
    monkeypatch.setenv(kube_token_cache.KEY_FINGERPRINT_ENV, fingerprint)
    return fingerprint


@pytest.fixture
def patch_config(monkeypatch):
    monkeypatch.setattr("prime_cli.commands.auth.Config", lambda: _FakeConfig())


def respond(monkeypatch, *, status_code, json_body=None, headers=None):
    requested_urls = []

    def _post(url, **kwargs):
        requested_urls.append(url)
        assert kwargs.get("follow_redirects") is False
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

    def test_exec_args_pin_a_persisted_custom_environment(self, monkeypatch):
        # `prime config use staging` then `factory k8s login`, with no
        # --context: switching environments later must not send the new
        # environment's key to staging's pinned URL.
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        config = _build_kubeconfig(
            cluster="c1",
            ca_data="Y2E=",
            grants=[_grant("alpha")],
            base_url="https://api.staging.example.com",
            environment="staging",
        )
        assert config["users"][0]["user"]["exec"]["args"][:2] == ["--context", "staging"]

    def test_exec_args_do_not_pin_the_production_environment(self, monkeypatch):
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        config = _build_kubeconfig(
            cluster="c1",
            ca_data="Y2E=",
            grants=[_grant("alpha")],
            base_url="https://api.example.com",
            environment="production",
        )
        assert "--context" not in config["users"][0]["user"]["exec"]["args"]

    def test_an_explicit_context_wins_over_the_persisted_environment(self, monkeypatch):
        monkeypatch.setenv("PRIME_CONTEXT", "dev")
        config = _build_kubeconfig(
            cluster="c1",
            ca_data="Y2E=",
            grants=[_grant("alpha")],
            base_url="https://api.dev.example.com",
            environment="staging",
        )
        assert config["users"][0]["user"]["exec"]["args"][:2] == ["--context", "dev"]

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
    api_key = "test-key"
    config = SimpleNamespace(current_environment="production")
    payload: dict = {}
    error: Exception | None = None
    requested: list = []

    def get(self, endpoint):
        type(self).requested.append(endpoint)
        if type(self).error is not None:
            raise type(self).error
        return type(self).payload


def _flat(output: str) -> str:
    """Rich wraps long lines; compare on normalized whitespace."""
    return " ".join(output.split())


NO_GRANT_DETAIL = (
    "You are not a member of any Kubernetes reservation on 'c1'. "
    "Ask your team admin to add you, or reserve nodes from the dashboard."
)


class TestFactoryK8sLogin:
    """`prime factory k8s login` through the root CLI, with the platform's
    `kubeconfig-info` response stubbed at the client boundary."""

    INFO = {
        "cluster": "c1",
        "certificateAuthorityData": "Y2E=",
        "grants": [_grant("train"), _grant("eval")],
    }

    @pytest.fixture
    def login(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        monkeypatch.setattr("prime_cli.commands.factory_k8s.KUBECONFIG_DIR", tmp_path / "kube")
        monkeypatch.setattr("prime_cli.commands.factory_k8s.APIClient", _FakeAPIClient)
        _FakeAPIClient.payload = dict(self.INFO)
        _FakeAPIClient.error = None
        _FakeAPIClient.requested = []

        def _run(*args):
            from prime_cli.main import app as root_app

            return runner.invoke(root_app, ["factory", "k8s", "login", *args])

        return _run

    def test_login_pins_the_persisted_environment(self, login, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _FakeAPIClient, "config", SimpleNamespace(current_environment="staging")
        )
        result = login("c1")

        assert result.exit_code == 0, result.output
        config = yaml.safe_load((tmp_path / "kube" / "c1.yaml").read_text())
        for user in config["users"]:
            assert user["user"]["exec"]["args"][:2] == ["--context", "staging"]

    def test_login_binds_the_kubeconfig_to_its_key(self, login, tmp_path):
        result = login("c1")

        assert result.exit_code == 0, result.output
        text = (tmp_path / "kube" / "c1.yaml").read_text()
        expected = kube_token_cache.key_fingerprint(
            base_url=_FakeAPIClient.base_url, api_key=_FakeAPIClient.api_key
        )
        assert _FakeAPIClient.api_key not in text
        for user in yaml.safe_load(text)["users"]:
            env = {item["name"]: item["value"] for item in user["user"]["exec"]["env"]}
            assert env[kube_token_cache.KEY_FINGERPRINT_ENV] == expected

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

        monkeypatch.setattr("prime_cli.commands.factory_k8s.os.replace", _boom)
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

    def test_no_grant_shows_the_platform_message_and_points_at_list(self, login, tmp_path):
        from prime_cli.core.client import APIError

        _FakeAPIClient.error = APIError(f"HTTP 403: {NO_GRANT_DETAIL}")

        result = login("c1")

        assert result.exit_code == 1
        output = _flat(result.output)
        assert NO_GRANT_DETAIL in output
        assert "HTTP 403" not in output
        assert "prime factory k8s list" in output
        assert "Removed" not in output
        assert not (tmp_path / "kube" / "c1.yaml").exists()

    def test_no_grant_removes_the_stale_kubeconfig(self, login, tmp_path):
        from prime_cli.core.client import APIError

        assert login("c1").exit_code == 0
        path = tmp_path / "kube" / "c1.yaml"
        other = tmp_path / "kube" / "c2.yaml"
        other.write_text("untouched")
        assert path.exists()

        _FakeAPIClient.error = APIError(f"HTTP 403: {NO_GRANT_DETAIL}")
        result = login("c1")

        assert result.exit_code == 1
        assert not path.exists()
        assert other.read_text() == "untouched"
        # Rich may break the long temp path anywhere; drop all whitespace.
        assert f"Removedthestalekubeconfig{path}." in "".join(result.output.split())

    def test_other_errors_keep_the_kubeconfig(self, login, tmp_path):
        from prime_cli.core.client import ServerError

        assert login("c1").exit_code == 0
        path = tmp_path / "kube" / "c1.yaml"
        before = path.read_text()

        _FakeAPIClient.error = ServerError("HTTP 503: try later")
        result = login("c1")

        assert result.exit_code == 1
        assert path.read_text() == before
        assert "Removed" not in result.output

    def test_no_reachable_grant_says_the_kept_file_may_be_stale(self, login, tmp_path):
        assert login("c1").exit_code == 0
        _FakeAPIClient.payload = {**self.INFO, "grants": [_grant("train", reason="Not yet.")]}

        result = login("c1")

        assert result.exit_code == 1
        path = tmp_path / "kube" / "c1.yaml"
        assert path.exists()
        squashed = "".join(result.output.split())
        assert f"Theexisting{path}waskept,butitscontextsmaybestale." in squashed

    def test_default_context_is_the_first_and_says_how_to_switch(self, login, tmp_path):
        result = login("c1")

        assert result.exit_code == 0, result.output
        config = yaml.safe_load((tmp_path / "kube" / "c1.yaml").read_text())
        assert config["current-context"] == "c1-train"
        output = _flat(result.output)
        assert "Current context: c1-train" in output
        assert "kubectl config use-context c1-eval" in output

    def test_a_single_context_has_nothing_to_switch_to(self, login):
        _FakeAPIClient.payload = {**self.INFO, "grants": [_grant("train")]}

        result = login("c1")

        assert result.exit_code == 0, result.output
        assert "Current context: c1-train" in result.output
        assert "use-context" not in result.output

    def test_pool_selects_the_current_context(self, login, tmp_path):
        result = login("c1", "--pool", "eval")

        assert result.exit_code == 0, result.output
        config = yaml.safe_load((tmp_path / "kube" / "c1.yaml").read_text())
        # Every reachable context is still written; only the current one moves.
        assert [c["name"] for c in config["contexts"]] == ["c1-train", "c1-eval"]
        assert config["current-context"] == "c1-eval"
        output = _flat(result.output)
        assert "Current context: c1-eval" in output
        assert "kubectl config use-context c1-train" in output

    def test_unknown_pool_lists_the_available_ones_and_writes_nothing(self, login, tmp_path):
        assert login("c1").exit_code == 0
        path = tmp_path / "kube" / "c1.yaml"
        before = path.read_text()

        result = login("c1", "--pool", "nope")

        assert result.exit_code == 1
        output = _flat(result.output)
        assert "Pool 'nope' on 'c1' is not one of your reservations there." in output
        assert "Available pools: train, eval" in output
        assert path.read_text() == before

    def test_unreachable_pool_says_why_and_writes_nothing(self, login, tmp_path):
        _FakeAPIClient.payload = {
            **self.INFO,
            "grants": [_grant("train"), _grant("eval", reason="Proxy not joined.")],
        }

        result = login("c1", "--pool", "eval")

        assert result.exit_code == 1
        output = _flat(result.output)
        assert "Pool 'eval' on 'c1' is not reachable: Proxy not joined." in output
        assert "Available pools: train" in output
        assert not (tmp_path / "kube" / "c1.yaml").exists()


class TestProxyDnsCheck:
    """After writing, each context's proxy name is looked up once; a name
    that does not resolve is a Tailscale hint, never a failed login."""

    @pytest.fixture
    def login(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        monkeypatch.setattr("prime_cli.commands.factory_k8s.KUBECONFIG_DIR", tmp_path / "kube")
        monkeypatch.setattr("prime_cli.commands.factory_k8s.APIClient", _FakeAPIClient)
        _FakeAPIClient.payload = dict(TestFactoryK8sLogin.INFO)
        _FakeAPIClient.error = None
        _FakeAPIClient.requested = []

        def _run(*args):
            from prime_cli.main import app as root_app

            return runner.invoke(root_app, ["factory", "k8s", "login", *args])

        return _run

    def test_resolvable_names_print_no_warning(self, login, proxy_names_resolve):
        result = login("c1")

        assert result.exit_code == 0, result.output
        assert sorted(proxy_names_resolve) == [
            "k8s-eval.tail1234.ts.net",
            "k8s-train.tail1234.ts.net",
        ]
        assert "Can't resolve" not in result.output

    def test_an_unresolvable_name_warns_but_the_login_succeeds(self, login, monkeypatch, tmp_path):
        def _resolve(host):
            if host.startswith("k8s-eval"):
                raise socket.gaierror(8, "nodename nor servname provided")

        monkeypatch.setattr("prime_cli.commands.factory_k8s._resolve", _resolve)

        result = login("c1")

        assert result.exit_code == 0, result.output
        assert (tmp_path / "kube" / "c1.yaml").exists()
        output = _flat(result.output)
        assert (
            "Can't resolve k8s-eval.tail1234.ts.net. Make sure Tailscale is connected "
            "to your team's tailnet (the one the reservation's proxy joined), then retry."
        ) in output
        assert "Can't resolve k8s-train" not in output

    def test_a_slow_resolver_is_cut_off(self, login, monkeypatch):
        release = threading.Event()

        def _hang(host):
            release.wait(10)

        monkeypatch.setattr("prime_cli.commands.factory_k8s._resolve", _hang)
        monkeypatch.setattr("prime_cli.commands.factory_k8s.DNS_CHECK_TIMEOUT_SECONDS", 0.2)
        started = time.monotonic()
        try:
            result = login("c1")
        finally:
            release.set()

        # One budget for every name, not one per name.
        assert time.monotonic() - started < 2
        assert result.exit_code == 0, result.output
        assert _flat(result.output).count("Can't resolve") == 2

    def test_ip_literals_are_not_looked_up(self, login, proxy_names_resolve):
        _FakeAPIClient.payload = {
            **TestFactoryK8sLogin.INFO,
            "grants": [
                _grant("train", server="https://[::1]:6443"),
                _grant("eval", server="https://127.0.0.1:6443"),
            ],
        }

        result = login("c1")

        assert result.exit_code == 0, result.output
        assert proxy_names_resolve == []


LIST_PAYLOAD = {
    "clusters": [
        {
            "cluster": "c1",
            "displayName": "Cluster One",
            "reservations": [
                {
                    "pool": "eval",
                    "namespace": "prime-user-u1-k8s-eval",
                    "teamId": "team-1",
                    "status": "Ready",
                    "proxyReady": True,
                    "server": "https://eval.tail1234.ts.net",
                    "unavailableReason": None,
                },
                {
                    "pool": "train",
                    "namespace": "prime-user-u1-k8s-train",
                    "teamId": "team-1",
                    "status": "Ready",
                    "proxyReady": True,
                    "server": "https://train.tail1234.ts.net",
                    "unavailableReason": None,
                },
            ],
        },
        {
            "cluster": "c2",
            "displayName": None,
            "reservations": [
                {
                    "pool": "debug",
                    "namespace": "prime-user-u1-k8s-debug",
                    "teamId": "team-1",
                    "status": "Provisioning",
                    "proxyReady": False,
                    "server": None,
                    "unavailableReason": "Proxy has not joined your tailnet yet.",
                }
            ],
        },
    ]
}


class TestFactoryK8sList:
    @pytest.fixture
    def cluster_list(self, monkeypatch):
        monkeypatch.setattr("prime_cli.commands.factory_k8s.APIClient", _FakeAPIClient)
        _FakeAPIClient.payload = LIST_PAYLOAD
        _FakeAPIClient.error = None
        _FakeAPIClient.requested = []

        def _run(*args):
            from prime_cli.main import app as root_app

            return runner.invoke(
                root_app, ["factory", "k8s", "list", *args], env={"COLUMNS": "200"}
            )

        return _run

    def test_table_lists_each_reservation_with_its_next_step(self, cluster_list):
        result = cluster_list()

        assert result.exit_code == 0, result.output
        assert _FakeAPIClient.requested == ["/clusters"]
        output = result.output
        for text in (
            "prime-user-u1-k8s-eval",
            "prime-user-u1-k8s-train",
            "prime-user-u1-k8s-debug",
            "Cluster One",
            "Provisioning",
            "not ready",
            # Two loginable pools on c1: each step names its pool.
            "prime factory k8s login c1 --pool eval",
            "prime factory k8s login c1 --pool train",
            "Proxy has not joined your tailnet yet.",
        ):
            assert text in output, text

    @pytest.mark.parametrize("columns", ["60", "80", "120"])
    def test_long_values_are_never_cut_off(self, cluster_list, monkeypatch, columns):
        """Namespaces and login commands are meant to be copied, so no
        terminal width may shorten them to "…"."""
        long_cluster = "dell-h100-nj-local-minh"
        long_ns = "prime-user-cmpm13dx20000wv33iwqk1sh7-k8s-dev"
        _FakeAPIClient.payload = {
            "clusters": [
                {
                    "cluster": long_cluster,
                    "displayName": None,
                    "reservations": [
                        {
                            "pool": "dev",
                            "namespace": long_ns,
                            "teamId": "t",
                            "status": "Ready",
                            "proxyReady": True,
                            "server": "https://dev-1.tail1234.ts.net",
                            "unavailableReason": None,
                        }
                    ],
                }
            ]
        }
        from prime_cli.main import app as root_app

        result = runner.invoke(root_app, ["factory", "k8s", "list"], env={"COLUMNS": columns})

        assert result.exit_code == 0, result.output
        assert "…" not in result.output
        assert long_ns in result.output
        assert f"prime factory k8s login {long_cluster}" in result.output

    def test_one_loginable_pool_needs_no_pool_flag(self, cluster_list):
        _FakeAPIClient.payload = {
            "clusters": [
                {
                    **LIST_PAYLOAD["clusters"][0],
                    "reservations": [LIST_PAYLOAD["clusters"][0]["reservations"][0]],
                }
            ]
        }

        result = cluster_list()

        assert result.exit_code == 0, result.output
        assert "prime factory k8s login c1" in result.output
        assert "--pool" not in result.output

    def test_json_is_the_platforms_shape(self, cluster_list):
        result = cluster_list("--output", "json")

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == LIST_PAYLOAD

    def test_empty_list_says_how_to_get_access(self, cluster_list):
        _FakeAPIClient.payload = {"clusters": []}

        result = cluster_list()

        assert result.exit_code == 0, result.output
        assert _flat(result.output) == (
            "You're not a member of any Kubernetes reservation. Ask your team admin "
            "to add you, or reserve nodes from the dashboard."
        )

    def test_empty_json_is_an_empty_list(self, cluster_list):
        _FakeAPIClient.payload = {"clusters": []}

        result = cluster_list("-o", "json")

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"clusters": []}

    def test_api_errors_are_reported(self, cluster_list):
        from prime_cli.core.client import UnauthorizedError

        _FakeAPIClient.error = UnauthorizedError("Token permission missing: scope 'clusters'")

        result = cluster_list()

        assert result.exit_code == 1
        assert "scope 'clusters'" in result.output

    def test_bad_output_format_is_refused(self, cluster_list):
        result = cluster_list("-o", "yaml")

        assert result.exit_code == 1
        assert _FakeAPIClient.requested == []


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class TestRootWiring:
    def test_auth_k8s_token_is_a_real_subcommand(self):
        # Every generated kubeconfig runs exactly `prime auth k8s-token`.
        from prime_cli.main import app as root_app

        result = runner.invoke(root_app, ["auth", "k8s-token", "--help"])
        assert result.exit_code == 0
        assert "--cluster" in _ANSI.sub("", result.output)

    def test_factory_k8s_login_is_a_real_subcommand(self):
        from prime_cli.main import app as root_app

        result = runner.invoke(root_app, ["factory", "k8s", "login", "--help"])
        assert result.exit_code == 0

    def test_factory_k8s_list_is_a_real_subcommand(self):
        from prime_cli.main import app as root_app

        result = runner.invoke(root_app, ["factory", "k8s", "list", "--help"])
        assert result.exit_code == 0
        assert "--output" in _ANSI.sub("", result.output)

    def test_factory_k8s_help_lists_list_and_login(self):
        from prime_cli.main import app as root_app

        result = runner.invoke(root_app, ["factory", "k8s", "--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0
        commands = re.findall(r"^\W*(\w[\w-]*)\s{2,}", _ANSI.sub("", result.output), re.M)
        assert {"list", "login"} <= set(commands)

    def test_factory_help_lists_k8s_but_not_the_kubernetes_alias(self):
        from prime_cli.main import app as root_app

        result = runner.invoke(root_app, ["factory", "--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0
        output = _ANSI.sub("", result.output)
        assert re.search(r"\bk8s\b", output)
        assert "kubernetes" not in output

    def test_kubernetes_is_a_hidden_alias_of_factory_k8s(self):
        from prime_cli.main import app as root_app

        result = runner.invoke(root_app, ["factory", "kubernetes", "login", "--help"])
        assert result.exit_code == 0

    def test_top_level_cluster_group_is_gone(self):
        from prime_cli.main import app as root_app

        result = runner.invoke(root_app, ["cluster", "--help"])
        assert result.exit_code != 0
        assert "No such command" in _ANSI.sub("", result.output)


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


def _credential(token, *, expires_in=timedelta(hours=1)):
    expiry = datetime.now(timezone.utc) + expires_in
    return {
        "apiVersion": "client.authentication.k8s.io/v1",
        "kind": "ExecCredential",
        "status": {"token": token, "expirationTimestamp": expiry.strftime("%Y-%m-%dT%H:%M:%SZ")},
    }


class _MintingPlatform:
    """`kube-token`, minting a new token on every successful call."""

    def __init__(self):
        self.calls = []
        self.status = 200
        self.expires_in = timedelta(hours=1)

    def post(self, url, **kwargs):
        self.calls.append(url)
        request = httpx.Request("POST", url)
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "refused"}, request=request)
        token = f"tok-{len(self.calls)}"
        return httpx.Response(
            200, json=_credential(token, expires_in=self.expires_in), request=request
        )


class TestTokenCache:
    """kubectl runs the plugin once per command, so without the cache every
    command minted a token and a polling loop hit the platform's rate limit."""

    @pytest.fixture
    def platform(self, monkeypatch):
        platform = _MintingPlatform()
        monkeypatch.setattr("prime_cli.commands.auth.httpx.post", platform.post)
        return platform

    @pytest.fixture
    def config(self, monkeypatch):
        class _Config:
            api_key = "key-one"
            base_url = "https://api.example.com"

        monkeypatch.setattr("prime_cli.commands.auth.Config", lambda: _Config)
        # These tests switch keys and platforms; key binding has its own tests.
        monkeypatch.setattr("prime_cli.commands.auth._check_key_binding", lambda *_: None)
        return _Config

    @staticmethod
    def run(*args, env=None):
        return runner.invoke(auth_app, ["k8s-token", "--cluster", "c1", *args], env=env)

    @classmethod
    def token(cls, *args, env=None):
        result = cls.run(*args, env=env)
        assert result.exit_code == 0, result.output
        return json.loads(result.stdout)["status"]["token"]

    @staticmethod
    def path(config, *, cluster="c1", pool=None):
        return kube_token_cache.entry_path(
            base_url=config.base_url, cluster=cluster, pool=pool, api_key=config.api_key
        )

    def seed(self, config, *, expires_in, pool=None, served_to=None):
        path = self.path(config, pool=pool)
        credential = _credential("seeded", expires_in=expires_in)
        kube_token_cache.save(path, {"credential": credential, "servedTo": served_to})
        return path

    def test_a_cache_hit_makes_no_request(self, config, platform):
        assert self.token("--pool", "train") == "tok-1"
        result = self.run("--pool", "train")

        assert result.exit_code == 0
        assert len(platform.calls) == 1
        # A hit still writes nothing but the credential to stdout.
        assert json.loads(result.stdout)["status"]["token"] == "tok-1"
        assert result.stderr == ""

    def test_a_token_well_inside_its_lifetime_is_served(self, config, platform):
        self.seed(config, expires_in=timedelta(minutes=6))
        assert self.token() == "seeded"
        assert platform.calls == []

    def test_a_token_inside_the_margin_is_replaced(self, config, platform):
        path = self.seed(config, expires_in=timedelta(minutes=4))

        assert self.token() == "tok-1"
        assert kube_token_cache.load(path)["credential"]["status"]["token"] == "tok-1"

    def test_an_expired_token_is_replaced(self, config, platform):
        self.seed(config, expires_in=timedelta(minutes=-1))
        assert self.token() == "tok-1"

    def test_a_credential_without_expiry_is_not_cached(self, config, monkeypatch):
        def _post(url, **kwargs):
            return httpx.Response(
                200, json={"status": {"token": "t"}}, request=httpx.Request("POST", url)
            )

        monkeypatch.setattr("prime_cli.commands.auth.httpx.post", _post)
        assert self.token() == "t"
        assert not self.path(config).exists()

    @pytest.mark.parametrize("content", ["", "{not json", "[]", '{"credential": "x"}'])
    def test_a_corrupt_entry_is_a_miss_and_is_overwritten(self, config, platform, content):
        path = self.seed(config, expires_in=timedelta(hours=1))
        path.write_text(content)

        assert self.token() == "tok-1"
        assert kube_token_cache.load(path)["credential"]["status"]["token"] == "tok-1"

    def test_entries_are_separated_by_platform_cluster_pool_and_key(self, config, platform):
        assert self.token("--pool", "train") == "tok-1"

        config.base_url = "https://staging.example.com"
        assert self.token("--pool", "train") == "tok-2"
        config.base_url = "https://api.example.com"

        assert self.token("--pool", "eval") == "tok-3"
        assert self.token() == "tok-4"

        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c2", "--pool", "train"])
        assert json.loads(result.stdout)["status"]["token"] == "tok-5"

        config.api_key = "key-two"
        assert self.token("--pool", "train") == "tok-6"

        # Each of those is now a hit of its own.
        config.api_key = "key-one"
        assert self.token("--pool", "train") == "tok-1"
        assert self.token("--pool", "eval") == "tok-3"
        assert len(platform.calls) == 6

    def test_nothing_on_disk_reveals_the_api_key(self, config, platform, token_cache_dir):
        config.api_key = "pit_secret_api_key"
        self.token("--pool", "train")

        files = [p for p in token_cache_dir.rglob("*") if p.is_file()]
        assert len(files) == 1
        for path in files:
            assert "pit_secret" not in str(path.relative_to(token_cache_dir))
            assert "pit_secret" not in path.read_text()

    def test_files_are_0600_and_directories_0700(self, config, platform, token_cache_dir):
        self.token("--pool", "train")

        path = self.path(config, pool="train")
        assert path.stat().st_mode & 0o777 == 0o600
        assert token_cache_dir.stat().st_mode & 0o777 == 0o700
        assert path.parent.stat().st_mode & 0o777 == 0o700
        # Atomic write: no temp file is left beside the entry.
        assert [p.name for p in path.parent.iterdir()] == [path.name]

    @pytest.mark.parametrize("status", [401, 403, 409])
    def test_a_refused_mint_evicts_the_entry(self, config, platform, monkeypatch, status):
        # The token kubectl was handed is rejected, kubectl re-runs the plugin
        # from the same process, and the platform refuses the new mint.
        monkeypatch.setattr("prime_cli.commands.auth.os.getppid", lambda: 42)
        path = self.seed(config, expires_in=timedelta(hours=1), served_to=42)
        platform.status = status

        assert self.run().exit_code != 0
        assert not path.exists()

    def test_a_refused_mint_with_no_cache_still_evicts(self, config, platform):
        path = self.seed(config, expires_in=timedelta(hours=1))
        platform.status = 403

        assert self.run("--no-cache").exit_code == EXIT_FORBIDDEN
        assert not path.exists()

    def test_a_transient_failure_keeps_the_entry(self, config, platform, monkeypatch):
        monkeypatch.setattr("prime_cli.commands.auth.os.getppid", lambda: 42)
        path = self.seed(config, expires_in=timedelta(hours=1), served_to=42)
        platform.status = 503

        assert self.run().exit_code == EXIT_UNREACHABLE
        assert path.exists()

    def test_a_rejected_token_is_replaced_once_per_process(self, config, platform, monkeypatch):
        # client-go re-runs the plugin from the same process after a 401.
        # Serving the same token again would just repeat the 401.
        assert self.token() == "tok-1"
        monkeypatch.setattr("prime_cli.commands.auth.os.getppid", lambda: 42)
        assert self.token() == "tok-1"  # a new kubectl, served from cache
        assert self.token() == "tok-2"  # the same kubectl again: rejected
        # A caller that re-runs the plugin without caching does not mint on
        # every call.
        assert self.token() == "tok-2"
        assert self.token() == "tok-2"
        assert len(platform.calls) == 2

    def test_a_revoked_grant_surfaces_the_platform_reason(self, config, platform, monkeypatch):
        assert self.token() == "tok-1"
        monkeypatch.setattr("prime_cli.commands.auth.os.getppid", lambda: 42)
        assert self.token() == "tok-1"
        platform.status = 403

        result = self.run()

        assert result.exit_code == EXIT_FORBIDDEN
        assert result.stdout == ""
        assert "refused" in result.stderr
        assert not self.path(config).exists()

    @pytest.mark.parametrize(
        "args, env",
        [(["--no-cache"], None), ([], {"PRIME_K8S_TOKEN_NO_CACHE": "1"})],
        ids=["flag", "env"],
    )
    def test_no_cache_always_mints_and_writes_nothing(
        self, config, platform, token_cache_dir, args, env
    ):
        self.seed(config, expires_in=timedelta(hours=1))
        before = {p: p.read_text() for p in token_cache_dir.rglob("*.json")}

        assert self.token(*args, env=env) == "tok-1"
        assert self.token(*args, env=env) == "tok-2"
        assert {p: p.read_text() for p in token_cache_dir.rglob("*.json")} == before

    def test_no_cache_env_can_be_turned_off(self, config, platform):
        assert self.token(env={"PRIME_K8S_TOKEN_NO_CACHE": "0"}) == "tok-1"
        assert self.token(env={"PRIME_K8S_TOKEN_NO_CACHE": "0"}) == "tok-1"

    def test_a_cache_write_failure_still_prints_the_credential(self, config, platform, monkeypatch):
        def _boom(path, entry):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr("prime_cli.commands.auth.token_cache.save", _boom)
        result = self.run()

        assert result.exit_code == 0
        assert json.loads(result.stdout)["status"]["token"] == "tok-1"
        assert "could not cache the credential" in result.stderr
        assert "Permission denied" in result.stderr
        assert "tok-1" not in result.stderr

    def test_an_unwritable_cache_directory_still_prints_the_credential(
        self, config, platform, token_cache_dir
    ):
        if os.geteuid() == 0:
            pytest.skip("root ignores directory permissions")
        token_cache_dir.parent.mkdir(parents=True)
        token_cache_dir.parent.chmod(0o500)
        try:
            result = self.run()
        finally:
            token_cache_dir.parent.chmod(0o700)

        assert result.exit_code == 0
        assert json.loads(result.stdout)["status"]["token"] == "tok-1"
        assert "could not cache the credential" in result.stderr


class TestFactoryK8sLoginClearsCachedTokens:
    @pytest.fixture
    def seeded(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        monkeypatch.setattr("prime_cli.commands.factory_k8s.KUBECONFIG_DIR", tmp_path / "kube")
        monkeypatch.setattr("prime_cli.commands.factory_k8s.APIClient", _FakeAPIClient)
        _FakeAPIClient.payload = dict(TestFactoryK8sLogin.INFO)
        _FakeAPIClient.error = None
        _FakeAPIClient.requested = []

        paths = {}
        for cluster, pool, base_url in [
            ("c1", "train", "https://api.example.com"),
            ("c1", "eval", "https://api.example.com"),
            ("c1", None, "https://api.example.com"),
            ("c1", "train", "https://staging.example.com"),
            ("c2", "train", "https://api.example.com"),
        ]:
            path = kube_token_cache.entry_path(
                base_url=base_url, cluster=cluster, pool=pool, api_key="k"
            )
            kube_token_cache.save(path, {"credential": _credential("t")})
            paths[(cluster, pool, base_url)] = path
        return paths

    def _login(self, cluster):
        from prime_cli.main import app as root_app

        return runner.invoke(root_app, ["factory", "k8s", "login", cluster])

    def test_login_clears_only_that_clusters_tokens(self, seeded):
        assert self._login("c1").exit_code == 0

        remaining = {key for key, path in seeded.items() if path.exists()}
        assert remaining == {("c2", "train", "https://api.example.com")}

    def test_a_failed_login_still_clears_them(self, seeded):
        from prime_cli.core.client import NotFoundError

        _FakeAPIClient.error = NotFoundError("HTTP 404: No active grant for 'c1'")

        assert self._login("c1").exit_code == 1
        assert not any(path.exists() for key, path in seeded.items() if key[0] == "c1")

    def test_login_with_nothing_cached_is_fine(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        monkeypatch.setattr("prime_cli.commands.factory_k8s.KUBECONFIG_DIR", tmp_path / "kube")
        monkeypatch.setattr("prime_cli.commands.factory_k8s.APIClient", _FakeAPIClient)
        _FakeAPIClient.payload = dict(TestFactoryK8sLogin.INFO)
        _FakeAPIClient.error = None

        result = self._login("c1")
        assert result.exit_code == 0, result.output
        assert "Could not clear" not in result.output


# --- logout / config reset clear the token cache -------------------------------


def _seed_cache(cache_dir):
    entry = cache_dir / "abc" / "def.json"
    entry.parent.mkdir(parents=True)
    entry.write_text("{}")
    return entry


@pytest.mark.parametrize("command", [["logout", "--yes"], ["config", "reset", "--yes"]])
def test_logout_and_reset_clear_cached_cluster_tokens(
    command, monkeypatch, tmp_path, token_cache_dir
):
    from prime_cli.core import Config
    from prime_cli.main import app as main_app

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    Config().set_api_key("secret")
    _seed_cache(token_cache_dir)

    result = runner.invoke(main_app, command)

    assert result.exit_code == 0, result.output
    assert not token_cache_dir.exists()


def test_logout_clears_tokens_even_when_not_logged_in(monkeypatch, tmp_path, token_cache_dir):
    """Tokens minted with PRIME_API_KEY from the environment are cached as well."""
    from prime_cli.main import app as main_app

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    _seed_cache(token_cache_dir)

    result = runner.invoke(main_app, ["logout", "--yes"])

    assert result.exit_code == 0, result.output
    assert not token_cache_dir.exists()


def test_clear_all_without_a_cache_is_a_no_op(token_cache_dir):
    assert kube_token_cache.clear_all() is False


class TestKeyBinding:
    """The plugin sends the current key only if it is the login's key, for the
    pinned platform, and never follows a redirect to another host."""

    def test_the_login_key_is_sent(self, patch_config, monkeypatch):
        urls = respond(monkeypatch, status_code=200, json_body={"status": {}})

        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])

        assert result.exit_code == 0, result.output
        assert urls == ["https://api.example.com/api/v1/clusters/c1/kube-token"]

    @pytest.mark.parametrize(
        "config",
        [
            SimpleNamespace(api_key="another-key", base_url="https://api.example.com"),
            SimpleNamespace(api_key="test-key", base_url="https://dev-tunnel.example.net"),
        ],
        ids=["other-key", "other-platform"],
    )
    def test_another_key_or_platform_is_refused_before_any_request(self, monkeypatch, config):
        monkeypatch.setattr("prime_cli.commands.auth.Config", lambda: config)
        urls = respond(monkeypatch, status_code=200, json_body={"status": {}})

        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])

        assert result.exit_code == 4
        assert "prime factory k8s login c1" in result.output
        assert urls == []

    def test_a_kubeconfig_without_a_fingerprint_is_refused(self, patch_config, monkeypatch):
        monkeypatch.delenv(kube_token_cache.KEY_FINGERPRINT_ENV)
        urls = respond(monkeypatch, status_code=200, json_body={"status": {}})

        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])

        assert result.exit_code == 4
        assert "prime factory k8s login c1" in result.output
        assert urls == []

    def test_the_fingerprint_does_not_reveal_the_key(self, login_fingerprint):
        assert _FakeConfig.api_key not in login_fingerprint
        assert len(login_fingerprint) == 32

    @staticmethod
    def _redirecting(monkeypatch, location):
        calls = []

        def _post(url, **kwargs):
            calls.append((url, kwargs["headers"].get("Authorization")))
            assert kwargs.get("follow_redirects") is False
            if len(calls) == 1:
                return httpx.Response(
                    307, headers={"Location": location}, request=httpx.Request("POST", url)
                )
            return httpx.Response(200, json={"status": {}}, request=httpx.Request("POST", url))

        monkeypatch.setattr("prime_cli.commands.auth.httpx.post", _post)
        return calls

    def test_a_same_origin_redirect_is_followed(self, patch_config, monkeypatch):
        calls = self._redirecting(monkeypatch, "/v2/clusters/c1/kube-token")

        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])

        assert result.exit_code == 0, result.output
        assert [url for url, _ in calls] == [
            "https://api.example.com/api/v1/clusters/c1/kube-token",
            "https://api.example.com/v2/clusters/c1/kube-token",
        ]

    def test_a_redirect_to_another_host_is_refused(self, patch_config, monkeypatch):
        calls = self._redirecting(monkeypatch, "https://elsewhere.example.net/kube-token")

        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])

        assert result.exit_code == 1
        assert "elsewhere.example.net" in result.output
        assert len(calls) == 1
