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
