"""SDK Config file loading: degrade, never crash.

`TracesAPIClient` builds a `Config` even when every constructor parameter is
explicit, so any crash in config loading takes down fully-parameterized
clients too. (The conftest's home isolation points Path.home at a tmpdir, so
these tests write ~/.prime/config.json freely.)
"""

import json
from pathlib import Path

from prime_traces import Config, TracesAPIClient


def _write_config(content: str) -> Path:
    config_dir = Path.home() / ".prime"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "config.json"
    path.write_text(content)
    return path


def test_non_dict_config_json_degrades_to_empty():
    """Valid JSON that is not an object must degrade exactly like invalid
    JSON: every accessor assumes a dict."""
    _write_config(json.dumps(["not", "a", "dict"]))
    assert Config().config == {}
    # The construction path that must survive it: all parameters explicit.
    api = TracesAPIClient(api_key="k", base_url="http://testserver", team_id="")
    assert api.api_key == "k"


def test_invalid_config_json_degrades_to_empty():
    _write_config("{not json")
    assert Config().config == {}


def test_non_utf8_config_degrades_to_empty():
    path = _write_config("")
    path.write_bytes(b"\xff")
    assert Config().config == {}
    # Config loading must not take down a client whose values are all explicit.
    api = TracesAPIClient(api_key="k", base_url="http://testserver", team_id="")
    assert api.api_key == "k"


def test_config_values_read_from_file():
    _write_config(
        json.dumps({"api_key": "file-key", "traces_url": "https://traces.example/api/v1"})
    )
    config = Config()
    assert config.api_key == "file-key"
    assert config.traces_url == "https://traces.example"


def test_traces_url_defaults_to_the_traces_service(monkeypatch):
    """No override anywhere: the traces service's own domain, never the platform
    API — ``/api/v1/traces`` is not routed there, so that fallback 404s."""
    assert Config().traces_url == Config.DEFAULT_TRACES_URL
    assert Config().traces_url != Config().base_url
    # A platform override says nothing about where traces lives.
    monkeypatch.setenv("PRIME_API_BASE_URL", "https://api.dev.example/api/v1")
    config = Config()
    assert config.base_url == "https://api.dev.example"
    assert config.traces_url == Config.DEFAULT_TRACES_URL


def test_traces_url_env_overrides_file(monkeypatch):
    _write_config(json.dumps({"traces_url": "https://traces.file.example"}))
    assert Config().traces_url == "https://traces.file.example"
    monkeypatch.setenv("PRIME_TRACES_URL", "http://localhost:8083/api/v1/")
    assert Config().traces_url == "http://localhost:8083"
