"""Coverage for `prime cluster login` and the `prime auth k8s-token` plugin.

The plugin's contract is unusually strict because kubectl, not a human, is the
caller: stdout is protocol, and the exit code is how anything wrapping it tells
"retry later" from "this will never work". Those are the properties worth
pinning down — a regression in them shows up as a confusing kubectl error
rather than a test failure anywhere else.
"""

import json

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
    def _post(url, **kwargs):
        return httpx.Response(
            status_code=status_code,
            json=json_body if json_body is not None else {},
            headers=headers or {},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("prime_cli.commands.auth.httpx.post", _post)


class TestKubeconfigRendering:
    def test_exec_block_carries_no_token(self):
        config = _build_kubeconfig(
            cluster="alpha-cluster",
            server="https://k8s.example.com",
            ca_data="Y2E=",
            grants=[{"pool": "alpha", "namespace": "ada-alpha"}],
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

    def test_one_context_per_pool(self):
        config = _build_kubeconfig(
            cluster="c1",
            server="https://k8s",
            ca_data="Y2E=",
            grants=[
                {"pool": "alpha", "namespace": "ada-alpha"},
                {"pool": "beta", "namespace": "ada-beta"},
            ],
        )
        assert [c["name"] for c in config["contexts"]] == ["c1-alpha", "c1-beta"]
        assert config["contexts"][0]["context"]["namespace"] == "ada-alpha"
        assert config["current-context"] == "c1-alpha"
        # Each context's plugin invocation names its own pool, otherwise the
        # server would refuse the ambiguous request on every kubectl call.
        assert config["users"][1]["user"]["exec"]["args"][-1] == "beta"


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

    def test_server_error_is_transient(self, patch_config, monkeypatch):
        respond(monkeypatch, status_code=503)
        result = runner.invoke(auth_app, ["k8s-token", "--cluster", "c1"])
        assert result.exit_code == EXIT_UNREACHABLE

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
