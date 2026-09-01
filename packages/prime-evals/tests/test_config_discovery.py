"""Project-local config discovery: PRIME_CONFIG_DIR > nearest trusted .prime/ > ~/.prime.

The conftest points Path.home at tmp_path and starts each test there.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from prime_evals import Config
from prime_evals.core import config as config_module


@pytest.fixture(autouse=True)
def reset_warning_dedup(monkeypatch):
    monkeypatch.setattr(config_module, "_warned_untrusted", set())


def _write(directory: Path, *, trusted: bool = True, **values: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(json.dumps(values))
    if trusted and directory != Path.home() / ".prime":
        _trust(path)
    return path


def _trust(path: Path) -> None:
    """Approve a local config the way `prime config trust` does."""
    registry = Path.home() / ".prime" / "trusted_configs.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    trusted = json.loads(registry.read_text())["trusted"] if registry.exists() else {}
    trusted[str(path.resolve())] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    registry.write_text(json.dumps({"version": 1, "trusted": trusted}))


def test_discovers_trusted_local_config_from_subdirectory(tmp_path, monkeypatch):
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
    assert config.untrusted_local_configs == []


def test_untrusted_local_config_is_ignored_with_a_warning(tmp_path, monkeypatch):
    _write(tmp_path / ".prime", api_key="global-key")
    project = tmp_path / "proj"
    planted = _write(project / ".prime", trusted=False, base_url="https://evil.example")
    monkeypatch.chdir(project)

    with pytest.warns(RuntimeWarning, match="prime config trust"):
        config = Config()

    assert config.config_source == "global"
    assert config.base_url == Config.DEFAULT_BASE_URL
    assert config.untrusted_local_configs == [planted]


def test_trust_is_bound_to_file_content(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    local = _write(project / ".prime", api_key="local-key")
    monkeypatch.chdir(project)
    assert Config().api_key == "local-key"

    local.write_text(json.dumps({"api_key": "local-key", "base_url": "https://evil.example"}))

    with pytest.warns(RuntimeWarning):
        assert Config().config_source == "global"


def test_local_config_replaces_global_rather_than_merging(tmp_path, monkeypatch):
    _write(tmp_path / ".prime", api_key="global-key")
    project = tmp_path / "proj"
    _write(project / ".prime", base_url="https://local.example")
    monkeypatch.chdir(project)

    config = Config()

    assert config.base_url == "https://local.example"
    assert config.api_key == ""


def test_falls_back_to_global_and_env_var_still_wins(tmp_path, monkeypatch):
    _write(tmp_path / ".prime", api_key="global-key")

    config = Config()
    assert config.config_source == "global"
    assert config.api_key == "global-key"

    monkeypatch.setenv("PRIME_API_KEY", "env-key")
    assert Config().api_key == "env-key"


def test_prime_config_dir_env_var_beats_discovery(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    _write(project / ".prime", api_key="local-key")
    explicit = tmp_path / "elsewhere"
    _write(explicit, trusted=False, api_key="explicit-key")
    monkeypatch.chdir(project)
    monkeypatch.setenv("PRIME_CONFIG_DIR", str(explicit))

    config = Config()

    assert config.config_source == "env"
    assert config.config_dir == explicit
    assert config.api_key == "explicit-key"


def test_trust_requires_ownership_even_with_matching_digest(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    local = _write(project / ".prime", api_key="local-key")
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

    assert not config_module.is_trusted_local_config(local)


def test_ignores_local_config_owned_by_another_user(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    _write(project / ".prime", api_key="planted-key")
    monkeypatch.chdir(project)
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

    config = Config()
    assert config.config_source == "global"
    assert config.untrusted_local_configs == []
