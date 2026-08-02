"""The traces commands must build their client from the CLI config, so that
`prime --context <env> traces ...` talks to that context's deployment instead
of silently falling back to the SDK's static config."""

from prime_cli.commands import traces as traces_cmd
from prime_cli.core import Config
from prime_cli.core.config import ConfigModel


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


class _EmptyContextConfig:
    """A context with no API key and no team — must stay that way."""

    api_key = ""
    traces_url = "https://traces.ctx.primeintellect.ai"
    team_id = None


class _SdkFileConfig:
    """What the SDK's static ~/.prime/config.json would resolve to."""

    api_key = "file-key"
    traces_url = "https://file.primeintellect.ai"
    team_id = "file-team"


def test_empty_context_never_falls_back_to_sdk_config(monkeypatch):
    """An unset api_key/team in the active context must not re-resolve from
    the SDK's static config — that would attribute traffic to the default
    context's credentials and team."""
    import prime_traces.core.client as sdk_client

    monkeypatch.setattr(traces_cmd, "Config", _EmptyContextConfig)
    monkeypatch.setattr(sdk_client, "Config", _SdkFileConfig)

    api = traces_cmd._traces_client().client
    assert api.api_key == ""
    assert api.team_id == ""
    assert "Authorization" not in api.client.headers
    assert "X-Prime-Team-Id" not in api.client.headers


def test_config_model_round_trips_traces_url():
    """traces_url must survive the load path, which round-trips the config
    file through ConfigModel — a missing field there is silently dropped."""
    config = Config.__new__(Config)
    config.config = ConfigModel(traces_url="https://traces.x.ai/api/v1/").model_dump()
    assert config.traces_url == "https://traces.x.ai"


def test_cli_config_traces_url_precedence(monkeypatch):
    monkeypatch.delenv("PRIME_TRACES_URL", raising=False)
    monkeypatch.delenv("PRIME_API_BASE_URL", raising=False)
    monkeypatch.delenv("PRIME_BASE_URL", raising=False)

    # Bypass __init__ so the test never touches ~/.prime on the dev machine.
    config = Config.__new__(Config)
    config.config = {"base_url": "https://api.staging.primeintellect.ai"}

    # No traces_url anywhere: fall back to the context's platform base URL.
    assert config.traces_url == "https://api.staging.primeintellect.ai"

    # Context file value wins over the fallback; /api/v1 is normalized away
    # like base_url does (the client appends the prefix itself).
    config.config["traces_url"] = "https://traces.staging.primeintellect.ai/api/v1"
    assert config.traces_url == "https://traces.staging.primeintellect.ai"

    # Env var wins over everything.
    monkeypatch.setenv("PRIME_TRACES_URL", "http://localhost:8083")
    assert config.traces_url == "http://localhost:8083"
