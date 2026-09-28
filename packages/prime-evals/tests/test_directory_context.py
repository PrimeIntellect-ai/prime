"""Directory context (.prime/context.json) resolution, mirrored from the Prime CLI."""

import json
from pathlib import Path

import pytest

from prime_evals.core.config import Config


@pytest.fixture
def repo(monkeypatch, tmp_path) -> Path:
    home = tmp_path / "home"
    repo = home / "code" / "edison"
    (repo / "src").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    for name in ("PRIME_CONTEXT", "PRIME_API_KEY", "PRIME_TEAM_ID", "PRIME_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("PRIME_API_BASE_URL", raising=False)
    config_dir = home / ".prime"
    (config_dir / "environments").mkdir(parents=True)
    (config_dir / "config.json").write_text(
        json.dumps({"api_key": "global-key", "team_id": "global-team"})
    )
    (config_dir / "environments" / "customer.json").write_text(
        json.dumps(
            {
                "api_key": "customer-key",
                "team_id": "customer-team",
                "base_url": "https://api.customer.example/api/v1",
            }
        )
    )
    monkeypatch.chdir(repo / "src")
    return repo


def _pin(directory: Path, data: dict) -> None:
    (directory / ".prime").mkdir(exist_ok=True)
    (directory / ".prime" / "context.json").write_text(json.dumps(data))


def test_no_directory_context_uses_global_config(repo) -> None:
    config = Config()
    assert config.local_context_file is None
    assert config.team_id == "global-team"


def test_team_pin_overrides_only_the_team(repo) -> None:
    _pin(repo, {"team_id": "edison-team", "team_name": "Edison"})
    config = Config()
    assert config.team_id == "edison-team"
    assert config.api_key == "global-key"


def test_null_team_pin_selects_personal_account(repo) -> None:
    _pin(repo, {"team_id": None})
    assert Config().team_id is None


def test_context_pin_selects_saved_context(repo) -> None:
    _pin(repo, {"context": "customer", "team_id": "edison-team"})
    config = Config()
    assert config.api_key == "customer-key"
    assert config.base_url == "https://api.customer.example"
    assert config.team_id == "edison-team"


def test_prime_context_and_env_vars_outrank_directory_context(repo, monkeypatch) -> None:
    _pin(repo, {"team_id": "edison-team"})
    monkeypatch.setenv("PRIME_CONTEXT", "customer")
    assert Config().team_id == "customer-team"
    monkeypatch.delenv("PRIME_CONTEXT")
    monkeypatch.setenv("PRIME_TEAM_ID", "env-team")
    assert Config().team_id == "env-team"


def test_home_prime_directory_is_not_a_directory_context(repo, monkeypatch) -> None:
    home = repo.parent.parent
    _pin(home, {"team_id": "edison-team"})
    monkeypatch.chdir(home)
    assert Config().team_id == "global-team"


def test_unknown_pinned_context_is_an_error(repo) -> None:
    _pin(repo, {"context": "missing"})
    config = Config()
    with pytest.raises(ValueError, match="context.json"):
        config.team_id


def test_malformed_directory_context_is_an_error(repo) -> None:
    (repo / ".prime").mkdir()
    (repo / ".prime" / "context.json").write_text("[]")
    config = Config()
    with pytest.raises(ValueError, match="expected a JSON object"):
        config.api_key


def test_symlinked_directory_context_is_ignored(repo) -> None:
    home = repo.parent.parent
    (repo / ".prime").mkdir()
    (repo / ".prime" / "context.json").symlink_to(home / ".prime" / "config.json")
    config = Config()
    assert config.local_context_file is None
    assert config.team_id == "global-team"


def test_empty_team_env_means_personal_account(repo, monkeypatch) -> None:
    monkeypatch.setenv("PRIME_TEAM_ID", "")
    assert Config().team_id is None


def test_broken_directory_context_only_fails_values_read_from_it(repo, monkeypatch) -> None:
    _pin(repo, {"context": "missing"})
    monkeypatch.setenv("PRIME_API_KEY", "env-key")
    monkeypatch.setenv("PRIME_TEAM_ID", "env-team")
    config = Config()
    assert config.api_key == "env-key"
    assert config.team_id == "env-team"
    with pytest.raises(ValueError, match="context.json"):
        config.config
