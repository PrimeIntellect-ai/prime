"""Project-local config discovery: PRIME_CONFIG_DIR > nearest .prime/ > ~/.prime.

The conftest points Path.home at tmp_path and starts each test there.
"""

import json
import os
from pathlib import Path

from prime_evals import Config


def _write(directory: Path, **values: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(json.dumps(values))
    return path


def test_discovers_local_config_from_subdirectory(tmp_path, monkeypatch):
    _write(tmp_path / ".prime", api_key="global-key")
    project = tmp_path / "code" / "prime"
    _write(project / ".prime", api_key="local-key")
    workdir = project / "envs" / "my-env"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)

    config = Config()

    assert config.config_source == "local"
    assert config.config_dir == project / ".prime"
    assert config.api_key == "local-key"


def test_local_config_replaces_global_rather_than_merging(tmp_path, monkeypatch):
    _write(tmp_path / ".prime", api_key="global-key")
    project = tmp_path / "proj"
    _write(project / ".prime", base_url="https://local.example")
    monkeypatch.chdir(project)
    monkeypatch.delenv("PRIME_API_KEY", raising=False)

    config = Config()

    assert config.base_url == "https://local.example"
    assert config.api_key == ""


def test_falls_back_to_global_and_env_var_still_wins(tmp_path, monkeypatch):
    _write(tmp_path / ".prime", api_key="global-key")
    monkeypatch.delenv("PRIME_API_KEY", raising=False)

    config = Config()
    assert config.config_source == "global"
    assert config.api_key == "global-key"

    monkeypatch.setenv("PRIME_API_KEY", "env-key")
    assert Config().api_key == "env-key"


def test_prime_config_dir_env_var_beats_discovery(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    _write(project / ".prime", api_key="local-key")
    explicit = tmp_path / "elsewhere"
    _write(explicit, api_key="explicit-key")
    monkeypatch.chdir(project)
    monkeypatch.setenv("PRIME_CONFIG_DIR", str(explicit))
    monkeypatch.delenv("PRIME_API_KEY", raising=False)

    config = Config()

    assert config.config_source == "env"
    assert config.config_dir == explicit
    assert config.api_key == "explicit-key"


def test_ignores_local_config_owned_by_another_user(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    _write(project / ".prime", api_key="planted-key")
    monkeypatch.chdir(project)
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

    assert Config().config_source == "global"
