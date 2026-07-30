"""The traces commands must build their client from the CLI config, so that
`prime --context <env> traces ...` talks to that context's deployment instead
of silently falling back to the SDK's static config."""

from prime_cli.commands import traces as traces_cmd
from prime_cli.core import Config


class _StubConfig:
    api_key = "ctx-key"
    traces_url = "https://traces.staging.primeintellect.ai"
    team_id = "team-ctx"


def test_traces_client_uses_cli_config(monkeypatch):
    monkeypatch.setattr(traces_cmd, "Config", _StubConfig)
    api = traces_cmd._traces_client().client
    assert api.api_key == "ctx-key"
    assert api.base_url == "https://traces.staging.primeintellect.ai"
    assert api.team_id == "team-ctx"
    assert api.client.headers["X-Prime-Team-Id"] == "team-ctx"


def test_cli_config_traces_url_precedence(monkeypatch):
    monkeypatch.delenv("PRIME_TRACES_URL", raising=False)
    monkeypatch.delenv("PRIME_API_BASE_URL", raising=False)
    monkeypatch.delenv("PRIME_BASE_URL", raising=False)

    # Bypass __init__ so the test never touches ~/.prime on the dev machine.
    config = Config.__new__(Config)
    config.config = {"base_url": "https://api.staging.primeintellect.ai"}

    # No traces_url anywhere: fall back to the context's platform base URL.
    assert config.traces_url == "https://api.staging.primeintellect.ai"

    # Context file value wins over the fallback.
    config.config["traces_url"] = "https://traces.staging.primeintellect.ai/"
    assert config.traces_url == "https://traces.staging.primeintellect.ai"

    # Env var wins over everything.
    monkeypatch.setenv("PRIME_TRACES_URL", "http://localhost:8083")
    assert config.traces_url == "http://localhost:8083"
