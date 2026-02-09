import json
from typing import Any, Dict, Optional

import pytest
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def mock_secrets_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the API client for secrets endpoints."""
    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    monkeypatch.setattr("prime_cli.core.Config.team_id", None)

    sample_secrets = [
        {
            "id": "secret-id-1234567890",
            "name": "MY_SECRET",
            "description": "Test secret",
            "isFile": False,
            "userId": "user-123",
            "teamId": None,
            "createdAt": "2026-01-15T10:00:00Z",
            "updatedAt": "2026-01-15T10:00:00Z",
        },
        {
            "id": "secret-id-0987654321",
            "name": "API_KEY",
            "description": None,
            "isFile": False,
            "userId": "user-123",
            "teamId": None,
            "createdAt": "2026-01-10T08:00:00Z",
            "updatedAt": "2026-01-10T08:00:00Z",
        },
    ]

    def mock_get(
        self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if endpoint == "/secrets/" or endpoint == "/secrets":
            return {"data": sample_secrets, "totalCount": 2}
        elif endpoint.startswith("/secrets/"):
            secret_id = endpoint.split("/")[-1]
            for s in sample_secrets:
                if s["id"] == secret_id or s["id"].startswith(secret_id):
                    return {"data": s}
            return {"data": sample_secrets[0]}
        return {"data": []}

    def mock_post(
        self: Any, endpoint: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if endpoint == "/secrets/" or endpoint == "/secrets":
            return {
                "data": {
                    "id": "new-secret-id-12345",
                    "name": json.get("name") if json else "NEW_SECRET",
                    "description": json.get("description") if json else None,
                    "isFile": json.get("isFile", False) if json else False,
                    "userId": "user-123",
                    "teamId": json.get("teamId") if json else None,
                    "createdAt": "2026-02-01T12:00:00Z",
                    "updatedAt": "2026-02-01T12:00:00Z",
                }
            }
        return {"data": {}}

    def mock_patch(
        self: Any, endpoint: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if endpoint.startswith("/secrets/"):
            return {
                "data": {
                    "id": "secret-id-1234567890",
                    "name": json.get("name", "MY_SECRET") if json else "MY_SECRET",
                    "description": json.get("description") if json else None,
                    "isFile": False,
                    "userId": "user-123",
                    "teamId": None,
                    "createdAt": "2026-01-15T10:00:00Z",
                    "updatedAt": "2026-02-01T12:00:00Z",
                }
            }
        return {"data": {}}

    def mock_delete(self: Any, endpoint: str) -> None:
        pass

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)
    monkeypatch.setattr("prime_cli.core.APIClient.post", mock_post)
    monkeypatch.setattr("prime_cli.core.APIClient.patch", mock_patch)
    monkeypatch.setattr("prime_cli.core.APIClient.delete", mock_delete)


class TestSecretsList:
    """Tests for the secrets list command."""

    def test_list_secrets_table_output(self, mock_secrets_api: None) -> None:
        """Test listing secrets with table output."""
        result = runner.invoke(app, ["secret", "list"], env={"COLUMNS": "200", "LINES": "50"})

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Personal Secrets" in result.output
        assert "MY_SECRET" in result.output
        assert "API_KEY" in result.output
        assert "Test secret" in result.output

    def test_list_secrets_json_output(self, mock_secrets_api: None) -> None:
        """Test listing secrets with JSON output."""
        result = runner.invoke(app, ["secret", "list", "-o", "json"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        output = json.loads(result.output)
        assert "secrets" in output
        assert len(output["secrets"]) == 2
        assert output["secrets"][0]["name"] == "MY_SECRET"

    def test_list_empty_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test listing when no secrets exist."""
        monkeypatch.setenv("PRIME_API_KEY", "test-key")
        monkeypatch.setattr("prime_cli.core.Config.team_id", None)

        def mock_get(
            self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            return {"data": [], "totalCount": 0}

        monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

        result = runner.invoke(app, ["secret", "list"])

        assert result.exit_code == 0
        assert "No personal secrets found" in result.output


class TestSecretsCreate:
    """Tests for the secrets create command."""

    def test_create_secret_success(self, mock_secrets_api: None) -> None:
        """Test creating a secret successfully."""
        result = runner.invoke(
            app,
            ["secret", "create", "-n", "NEW_SECRET", "-v", "secret-value"],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Created personal secret 'NEW_SECRET'" in result.output
        assert "ID:" in result.output

    def test_create_secret_with_description(self, mock_secrets_api: None) -> None:
        """Test creating a secret with a description."""
        result = runner.invoke(
            app,
            [
                "secret",
                "create",
                "-n",
                "NEW_SECRET",
                "-v",
                "secret-value",
                "-d",
                "My description",
            ],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Created personal secret 'NEW_SECRET'" in result.output

    def test_create_secret_json_output(self, mock_secrets_api: None) -> None:
        """Test creating a secret with JSON output."""
        result = runner.invoke(
            app,
            ["secret", "create", "-n", "NEW_SECRET", "-v", "value", "-o", "json"],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        output = json.loads(result.output)
        assert output["name"] == "NEW_SECRET"
        assert "id" in output

    def test_create_secret_interactive_cancel_name(self, mock_secrets_api: None) -> None:
        """Test that create can be cancelled during name prompt."""
        result = runner.invoke(
            app,
            ["secret", "create"],
            input="\n",
        )
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_create_secret_interactive(self, mock_secrets_api: None) -> None:
        """Test creating a secret interactively."""
        result = runner.invoke(
            app,
            ["secret", "create"],
            input="MY_NEW_SECRET\nsecret-value\n",
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Created personal secret" in result.output


class TestSecretsUpdate:
    """Tests for the secrets update command."""

    def test_update_secret_name(self, mock_secrets_api: None) -> None:
        """Test updating a secret's name."""
        result = runner.invoke(
            app,
            ["secret", "update", "secret-id-1234567890", "-n", "RENAMED_SECRET"],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Updated secret" in result.output

    def test_update_secret_value(self, mock_secrets_api: None) -> None:
        """Test updating a secret's value."""
        result = runner.invoke(
            app,
            ["secret", "update", "secret-id-1234567890", "-v", "new-value"],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Updated secret" in result.output

    def test_update_secret_no_changes(self, mock_secrets_api: None) -> None:
        """Test that update prompts when no changes are provided."""
        result = runner.invoke(
            app,
            ["secret", "update", "secret-id-1234567890"],
            input="\n",
        )
        assert result.exit_code == 0
        assert "No changes made" in result.output

    def test_update_secret_interactive(self, mock_secrets_api: None) -> None:
        """Test updating a secret interactively."""
        result = runner.invoke(
            app,
            ["secret", "update", "secret-id-1234567890"],
            input="new-secret-value\n",
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Updated secret" in result.output

    def test_update_secret_json_output(self, mock_secrets_api: None) -> None:
        """Test updating a secret with JSON output."""
        result = runner.invoke(
            app,
            ["secret", "update", "secret-id-1234567890", "-n", "NEW_NAME", "-o", "json"],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        output = json.loads(result.output)
        assert "id" in output


class TestSecretsDelete:
    """Tests for the secrets delete command."""

    def test_delete_secret_with_confirm(self, mock_secrets_api: None) -> None:
        """Test deleting a secret with confirmation."""
        result = runner.invoke(
            app,
            ["secret", "delete", "secret-id-1234567890", "-y"],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Deleted secret" in result.output

    def test_delete_secret_cancelled(self, mock_secrets_api: None) -> None:
        """Test cancelling secret deletion."""
        result = runner.invoke(
            app,
            ["secret", "delete", "secret-id-1234567890"],
            input="n\n",
        )

        assert result.exit_code == 0
        assert "Cancelled" in result.output


class TestSecretsGet:
    """Tests for the secrets get command."""

    def test_get_secret_table_output(self, mock_secrets_api: None) -> None:
        """Test getting a secret with table output."""
        result = runner.invoke(
            app,
            ["secret", "get", "secret-id-1234567890"],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Secret Details" in result.output
        assert "MY_SECRET" in result.output

    def test_get_secret_json_output(self, mock_secrets_api: None) -> None:
        """Test getting a secret with JSON output."""
        result = runner.invoke(
            app,
            ["secret", "get", "secret-id-1234567890", "-o", "json"],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        output = json.loads(result.output)
        assert output["name"] == "MY_SECRET"
        assert output["id"] == "secret-id-1234567890"


class TestSecretsHelp:
    """Tests for help output."""

    def test_secrets_help(self) -> None:
        """Test that secrets help works."""
        result = runner.invoke(app, ["secret", "--help"])

        assert result.exit_code == 0
        assert "Manage global secrets" in result.output
        assert "list" in result.output
        assert "create" in result.output
        assert "update" in result.output
        assert "delete" in result.output
        assert "get" in result.output

    def test_secrets_list_help(self) -> None:
        """Test that secrets list help works."""
        result = runner.invoke(app, ["secret", "list", "--help"])

        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--output" in output

    def test_secrets_create_help(self) -> None:
        """Test that secrets create help works."""
        result = runner.invoke(app, ["secret", "create", "--help"])

        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--name" in output
        assert "--value" in output
        assert "--description" in output
        assert "--file" in output
