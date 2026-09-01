import json
import os
from pathlib import Path
from typing import Any

import pytest
from prime_cli.core import Config
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _create_dev_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.delenv("PRIME_CONTEXT", raising=False)
    config = Config()
    config.set_api_key("root-key")
    config.set_user_id("root-user")
    config.save_environment("dev")
    return config.config_file, config.environments_dir / "dev.json"


@pytest.mark.parametrize(
    "command",
    [
        ["config", "set-api-key", "replacement-key"],
        ["config", "set-team-id", ""],
        ["config", "set-base-url", "https://replacement.example"],
        ["config", "set-frontend-url", "https://frontend.example"],
        ["config", "set-inference-url", "https://inference.example"],
        ["config", "set-traces-url", "https://traces.example"],
        ["config", "set-share-resources-with-team", "true"],
        ["config", "set-ssh-key-path", "/tmp/id_test"],
        ["config", "reset", "--yes"],
        ["config", "use", "production"],
        ["config", "save", "copy"],
        ["config", "delete", "dev"],
        ["login", "--headless"],
        ["logout", "--yes"],
        ["switch", "personal"],
    ],
)
def test_temporary_context_rejects_config_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: list[str],
) -> None:
    root_file, dev_file = _create_dev_context(monkeypatch, tmp_path)
    root_before = root_file.read_text()
    dev_before = dev_file.read_text()

    result = runner.invoke(app, ["-c", "dev", *command])

    assert result.exit_code == 1, result.output
    assert "Temporary context 'dev' is read-only" in result.output
    assert root_file.read_text() == root_before
    assert dev_file.read_text() == dev_before
    assert os.getenv("PRIME_CONTEXT") is None


def test_whoami_does_not_cache_user_id_in_temporary_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root_file, dev_file = _create_dev_context(monkeypatch, tmp_path)
    root_before = json.loads(root_file.read_text())
    dev_before = json.loads(dev_file.read_text())

    def mock_get(self: Any, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        assert endpoint == "/user/whoami"
        return {
            "data": {
                "id": "fresh-user",
                "email": "dev@example.com",
                "name": "Dev User",
                "slug": "dev-user",
                "scope": {},
            }
        }

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = runner.invoke(app, ["-c", "dev", "whoami"])

    assert result.exit_code == 0, result.output
    assert "fresh-user" in result.output
    assert json.loads(root_file.read_text()) == root_before
    assert json.loads(dev_file.read_text()) == dev_before
