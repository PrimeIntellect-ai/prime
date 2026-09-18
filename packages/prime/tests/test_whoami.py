"""Tests for the whoami command's output rendering."""

from pathlib import Path
from typing import Any

import pytest
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    for variable in ("PRIME_CONTEXT", "PRIME_TEAM_ID", "PRIME_USER_ID"):
        monkeypatch.delenv(variable, raising=False)


def _mock_whoami(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        assert endpoint == "/user/whoami"
        return {
            "data": {
                "id": "user-1",
                "email": "user@example.com",
                "name": "User One",
                "slug": None,  # missing username renders "Not set"
                "scope": {},
            }
        }

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)


@pytest.mark.parametrize("plain", [False, True])
def test_missing_username_renders_without_markup_leak(monkeypatch, plain):
    _mock_whoami(monkeypatch)

    args = ["whoami", "--plain"] if plain else ["whoami"]
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "Not set" in result.output
    assert "[dim]" not in result.output
