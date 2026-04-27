import json
from types import SimpleNamespace

from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "PRIME_API_KEY": "test-token",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}


def _fork_response() -> dict:
    return {
        "data": {
            "success": True,
            "message": "Environment forked successfully",
            "environment_id": "env-fork",
            "version_id": "version-fork",
            "name": "gsm8k",
            "owner": {"type": "user", "name": "alice"},
            "slug": "alice/gsm8k",
        }
    }


def test_prime_fork_posts_to_environment_fork_endpoint(monkeypatch):
    captured = {}

    class DummyAPIClient:
        config = SimpleNamespace(team_id=None)

        def post(self, endpoint, json=None):
            captured["endpoint"] = endpoint
            captured["json"] = json
            return _fork_response()

    monkeypatch.setattr("prime_cli.commands.fork.APIClient", DummyAPIClient)

    result = runner.invoke(app, ["fork", "openai/gsm8k"], env=TEST_ENV)

    assert result.exit_code == 0
    assert captured == {
        "endpoint": "/environmentshub/openai/gsm8k/fork",
        "json": {},
    }
    assert "Forked openai/gsm8k to alice/gsm8k" in result.output
    assert "prime env pull alice/gsm8k" in result.output


def test_prime_fork_uses_configured_team_id(monkeypatch):
    captured = {}

    class DummyAPIClient:
        config = SimpleNamespace(team_id="configured-team")

        def post(self, endpoint, json=None):
            captured["endpoint"] = endpoint
            captured["json"] = json
            return _fork_response()

    monkeypatch.setattr("prime_cli.commands.fork.APIClient", DummyAPIClient)

    result = runner.invoke(app, ["fork", "openai/gsm8k"], env=TEST_ENV)

    assert result.exit_code == 0
    assert captured == {
        "endpoint": "/environmentshub/openai/gsm8k/fork",
        "json": {"team_id": "configured-team"},
    }


def test_prime_fork_team_slug_overrides_configured_team_id(monkeypatch):
    captured = {}

    class DummyAPIClient:
        config = SimpleNamespace(team_id="configured-team")

        def post(self, endpoint, json=None):
            captured["endpoint"] = endpoint
            captured["json"] = json
            return _fork_response()

    monkeypatch.setattr("prime_cli.commands.fork.APIClient", DummyAPIClient)

    result = runner.invoke(
        app,
        ["fork", "openai/gsm8k", "--team", "research", "--output", "json"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0
    assert captured == {
        "endpoint": "/environmentshub/openai/gsm8k/fork",
        "json": {"team_slug": "research"},
    }
    assert json.loads(result.output)["slug"] == "alice/gsm8k"


def test_prime_fork_rejects_versioned_sources():
    result = runner.invoke(
        app,
        ["fork", "openai/gsm8k@0.1.0"],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "Forking a specific version is not supported" in result.output
