import json

import pytest

from prime_sandboxes import Config


def _write_configs(tmp_path) -> None:
    config_dir = tmp_path / ".prime"
    environments_dir = config_dir / "environments"
    environments_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "api_key": "production-key",
                "base_url": "https://api.production.example",
                "team_id": "production-team",
                "user_id": "production-user",
            }
        )
    )
    (environments_dir / "dev.json").write_text(
        json.dumps(
            {
                "api_key": "dev-key",
                "base_url": "https://api.dev.example/api/v1",
                "team_id": "dev-team",
                "user_id": "dev-user",
            }
        )
    )


def test_config_loads_temporary_context(monkeypatch, tmp_path) -> None:
    _write_configs(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_CONTEXT", "dev")

    config = Config()

    assert config.api_key == "dev-key"
    assert config.base_url == "https://api.dev.example"
    assert config.team_id == "dev-team"
    assert config.user_id == "dev-user"


def test_environment_variables_override_temporary_context(monkeypatch, tmp_path) -> None:
    _write_configs(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_CONTEXT", "dev")
    monkeypatch.setenv("PRIME_API_KEY", "environment-key")
    monkeypatch.setenv("PRIME_API_BASE_URL", "https://api.environment.example/api/v1")
    monkeypatch.setenv("PRIME_TEAM_ID", "environment-team")
    monkeypatch.setenv("PRIME_USER_ID", "environment-user")

    config = Config()

    assert config.api_key == "environment-key"
    assert config.base_url == "https://api.environment.example"
    assert config.team_id == "environment-team"
    assert config.user_id == "environment-user"


def test_production_context_restores_builtin_scope(monkeypatch, tmp_path) -> None:
    _write_configs(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_CONTEXT", "production")

    config = Config()

    assert config.api_key == "production-key"
    assert config.base_url == Config.DEFAULT_BASE_URL
    assert config.team_id is None
    assert config.user_id == "production-user"


@pytest.mark.parametrize("content", ["{", "[]", '"not-an-object"'])
def test_broken_temporary_context_never_falls_back(monkeypatch, tmp_path, content) -> None:
    _write_configs(tmp_path)
    (tmp_path / ".prime" / "environments" / "dev.json").write_text(content)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_CONTEXT", "dev")

    with pytest.raises(ValueError, match="context|JSON object"):
        Config()


def test_missing_temporary_context_never_falls_back(monkeypatch, tmp_path) -> None:
    _write_configs(tmp_path)
    (tmp_path / ".prime" / "environments" / "dev.json").unlink()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_CONTEXT", "dev")

    with pytest.raises(ValueError, match="Context file not found"):
        Config()


def test_unreadable_temporary_context_never_falls_back(monkeypatch, tmp_path) -> None:
    _write_configs(tmp_path)
    environment_file = tmp_path / ".prime" / "environments" / "dev.json"
    original_read_text = type(environment_file).read_text

    def fail_for_context(path, *args, **kwargs):
        if path == environment_file:
            raise PermissionError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(environment_file), "read_text", fail_for_context)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_CONTEXT", "dev")

    with pytest.raises(ValueError, match="Failed to load context 'dev'"):
        Config()
