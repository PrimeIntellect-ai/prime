"""Constructor sentinel semantics: None means "resolve from the SDK's static
config"; any explicit value — including "" — is final.

Injectors (the prime CLI's `--context` support) rely on this: a context whose
api_key or team is unset must fail or go teamless, never silently borrow the
static config's credentials.
"""

import pytest

import prime_traces.core.client as client_module
from prime_traces import APIError, TracesAPIClient


class _FileConfig:
    """Stands in for what ~/.prime/config.json would resolve to."""

    api_key = "file-key"
    traces_url = "https://file.primeintellect.ai"
    team_id = "file-team"


@pytest.fixture(autouse=True)
def stub_sdk_config(monkeypatch):
    monkeypatch.setattr(client_module, "Config", _FileConfig)


def test_none_resolves_from_config():
    api = TracesAPIClient()
    assert api.api_key == "file-key"
    assert api.base_url == "https://file.primeintellect.ai"
    assert api.team_id == "file-team"
    assert api.client.headers["Authorization"] == "Bearer file-key"
    assert api.client.headers["X-Prime-Team-Id"] == "file-team"


def test_explicit_empty_is_final_not_a_fallback_trigger():
    api = TracesAPIClient(api_key="", base_url="http://testserver", team_id="")
    assert api.api_key == ""
    assert api.team_id == ""
    assert "Authorization" not in api.client.headers
    assert "X-Prime-Team-Id" not in api.client.headers


def test_empty_api_key_fails_loudly_at_request_time():
    api = TracesAPIClient(api_key="", base_url="http://testserver", team_id="")
    with pytest.raises(APIError, match="No API key configured"):
        api.get_json("/traces")
