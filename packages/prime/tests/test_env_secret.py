import json
from typing import Any, Dict, List, Optional

import pytest
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def mock_env_secret_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the API client for environment secret endpoints."""
    monkeypatch.setenv("PRIME_API_KEY", "test-key")

    sample_env_detail = {
        "id": "env-uuid-12345",
        "name": "test-env",
        "owner": {"name": "testuser", "type": "user"},
    }

    sample_secrets: List[Dict[str, Any]] = [
        {
            "id": "esecret-id-001",
            "name": "DB_PASSWORD",
            "source": "environment",
            "description": "Database password",
            "createdAt": "2026-01-15T10:00:00Z",
            "updatedAt": "2026-01-15T10:00:00Z",
        },
        {
            "id": "esecret-id-002",
            "name": "OPENAI_KEY",
            "source": "global-linked",
            "description": None,
            "createdAt": "2026-01-10T08:00:00Z",
            "updatedAt": "2026-01-10T08:00:00Z",
        },
    ]

    def mock_get(
        self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if "/@latest" in endpoint:
            return {"data": sample_env_detail}
        elif "/secrets" in endpoint and "/link/" not in endpoint:
            if endpoint.endswith("/secrets"):
                return {"data": sample_secrets}
            # Individual secret
            secret_id = endpoint.split("/")[-1]
            for s in sample_secrets:
                if s["id"] == secret_id:
                    return {"data": s}
            return {"data": sample_secrets[0]}
        return {"data": {}}

    def mock_post(
        self: Any, endpoint: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if "/secrets/link/" in endpoint:
            global_id = endpoint.split("/link/")[-1]
            return {
                "data": {
                    "id": "link-id-12345",
                    "secretId": global_id,
                    "secretName": json.get("envVarName", "LINKED_SECRET")
                    if json
                    else "LINKED_SECRET",
                    "environmentId": "env-uuid-12345",
                    "createdAt": "2026-02-01T12:00:00Z",
                }
            }
        elif "/secrets" in endpoint:
            return {
                "data": {
                    "id": "new-esecret-id-001",
                    "name": json.get("name") if json else "NEW_SECRET",
                    "value": "[encrypted]",
                    "description": json.get("description") if json else None,
                    "source": "environment",
                    "createdAt": "2026-02-01T12:00:00Z",
                    "updatedAt": "2026-02-01T12:00:00Z",
                }
            }
        return {"data": {}}

    def mock_patch(
        self: Any, endpoint: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if "/secrets/" in endpoint:
            return {
                "data": {
                    "id": "esecret-id-001",
                    "name": json.get("name", "DB_PASSWORD") if json else "DB_PASSWORD",
                    "description": json.get("description") if json else None,
                    "source": "environment",
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


class TestEnvSecretList:
    """Tests for the env secret list command."""

    def test_list_secrets_table_output(self, mock_env_secret_api: None) -> None:
        """Test listing env secrets with table output."""
        result = runner.invoke(
            app,
            ["env", "secret", "list", "testuser/test-env"],
            env={"COLUMNS": "200", "LINES": "50"},
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Secrets for testuser/test-env" in result.output
        assert "DB_PASSWORD" in result.output
        assert "OPENAI_KEY" in result.output

    def test_list_secrets_json_output(self, mock_env_secret_api: None) -> None:
        """Test listing env secrets with JSON output."""
        result = runner.invoke(app, ["env", "secret", "list", "testuser/test-env", "-o", "json"])
        assert result.exit_code == 0, f"Failed: {result.output}"
        output = json.loads(result.output)
        assert "secrets" in output
        assert len(output["secrets"]) == 2
        assert output["secrets"][0]["name"] == "DB_PASSWORD"

    def test_list_secrets_shows_source(self, mock_env_secret_api: None) -> None:
        """Test that list output contains the source column."""
        result = runner.invoke(
            app,
            ["env", "secret", "list", "testuser/test-env"],
            env={"COLUMNS": "200", "LINES": "50"},
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "environment" in result.output
        assert "global-linked" in result.output

    def test_list_empty_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test listing when no env secrets exist."""
        monkeypatch.setenv("PRIME_API_KEY", "test-key")

        def mock_get(
            self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            if "/@latest" in endpoint:
                return {"data": {"id": "env-uuid-12345"}}
            return {"data": []}

        monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

        result = runner.invoke(app, ["env", "secret", "list", "testuser/test-env"])
        assert result.exit_code == 0
        assert "No secrets found" in result.output

    def test_list_secrets_invalid_output_format(self, mock_env_secret_api: None) -> None:
        """Test listing secrets with an invalid output format."""
        result = runner.invoke(app, ["env", "secret", "list", "testuser/test-env", "-o", "xml"])
        assert result.exit_code != 0
        assert "Invalid output format" in result.output


class TestEnvSecretCreate:
    """Tests for the env secret create command."""

    def test_create_secret_success(self, mock_env_secret_api: None) -> None:
        """Test creating an env secret successfully."""
        result = runner.invoke(
            app,
            [
                "env",
                "secret",
                "create",
                "testuser/test-env",
                "-n",
                "NEW_SECRET",
                "-v",
                "secret-value",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Created secret 'NEW_SECRET'" in result.output
        assert "testuser/test-env" in result.output
        assert "ID:" in result.output

    def test_create_secret_with_description(self, mock_env_secret_api: None) -> None:
        """Test creating an env secret with a description."""
        result = runner.invoke(
            app,
            [
                "env",
                "secret",
                "create",
                "testuser/test-env",
                "-n",
                "NEW_SECRET",
                "-v",
                "secret-value",
                "-d",
                "A test secret",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Created secret 'NEW_SECRET'" in result.output

    def test_create_secret_json_output(self, mock_env_secret_api: None) -> None:
        """Test creating an env secret with JSON output."""
        result = runner.invoke(
            app,
            [
                "env",
                "secret",
                "create",
                "testuser/test-env",
                "-n",
                "NEW_SECRET",
                "-v",
                "value",
                "-o",
                "json",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        output = json.loads(result.output)
        assert output["name"] == "NEW_SECRET"
        assert "id" in output

    def test_create_secret_interactive(self, mock_env_secret_api: None) -> None:
        """Test creating an env secret interactively."""
        result = runner.invoke(
            app,
            ["env", "secret", "create", "testuser/test-env"],
            input="MY_NEW_SECRET\nsecret-value\n",
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Created secret" in result.output

    def test_create_secret_interactive_cancel_name(self, mock_env_secret_api: None) -> None:
        """Test that create can be cancelled during name prompt."""
        result = runner.invoke(
            app,
            ["env", "secret", "create", "testuser/test-env"],
            input="\n",
        )
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_create_secret_interactive_cancel_value(self, mock_env_secret_api: None) -> None:
        """Test that create can be cancelled during value prompt."""
        result = runner.invoke(
            app,
            ["env", "secret", "create", "testuser/test-env", "-n", "NEW_SECRET"],
            input="\n",
        )
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_create_secret_invalid_name_lowercase(self, mock_env_secret_api: None) -> None:
        """Test that lowercase secret names are rejected."""
        result = runner.invoke(
            app,
            ["env", "secret", "create", "testuser/test-env", "-n", "my_secret", "-v", "value"],
        )
        assert result.exit_code != 0
        assert "Invalid secret name" in result.output

    def test_create_secret_invalid_name_starts_with_number(self, mock_env_secret_api: None) -> None:
        """Test that names starting with a number are rejected."""
        result = runner.invoke(
            app,
            ["env", "secret", "create", "testuser/test-env", "-n", "1BAD_NAME", "-v", "value"],
        )
        assert result.exit_code != 0
        assert "Invalid secret name" in result.output

    def test_create_secret_invalid_name_special_chars(self, mock_env_secret_api: None) -> None:
        """Test that names with special characters are rejected."""
        result = runner.invoke(
            app,
            ["env", "secret", "create", "testuser/test-env", "-n", "MY-SECRET", "-v", "value"],
        )
        assert result.exit_code != 0
        assert "Invalid secret name" in result.output


class TestEnvSecretUpdate:
    """Tests for the env secret update command."""

    def test_update_secret_with_id_and_name(self, mock_env_secret_api: None) -> None:
        """Test updating a secret name by ID."""
        result = runner.invoke(
            app,
            [
                "env",
                "secret",
                "update",
                "testuser/test-env",
                "--id",
                "esecret-id-001",
                "-n",
                "RENAMED_SECRET",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Updated secret" in result.output

    def test_update_secret_with_value(self, mock_env_secret_api: None) -> None:
        """Test updating a secret value by ID."""
        result = runner.invoke(
            app,
            [
                "env",
                "secret",
                "update",
                "testuser/test-env",
                "--id",
                "esecret-id-001",
                "-v",
                "new-secret-value",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Updated secret" in result.output

    def test_update_secret_with_description(self, mock_env_secret_api: None) -> None:
        """Test updating a secret description."""
        result = runner.invoke(
            app,
            [
                "env",
                "secret",
                "update",
                "testuser/test-env",
                "--id",
                "esecret-id-001",
                "-d",
                "Updated description",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Updated secret" in result.output

    def test_update_secret_json_output(self, mock_env_secret_api: None) -> None:
        """Test updating a secret with JSON output."""
        result = runner.invoke(
            app,
            [
                "env",
                "secret",
                "update",
                "testuser/test-env",
                "--id",
                "esecret-id-001",
                "-n",
                "NEW_NAME",
                "-o",
                "json",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        output = json.loads(result.output)
        assert "id" in output

    def test_update_secret_no_changes_cancel(self, mock_env_secret_api: None) -> None:
        """Test that update with no changes and empty interactive input cancels."""
        result = runner.invoke(
            app,
            ["env", "secret", "update", "testuser/test-env", "--id", "esecret-id-001"],
            input="\n",
        )
        assert result.exit_code == 0
        assert "No changes made" in result.output

    def test_update_secret_interactive_select(self, mock_env_secret_api: None) -> None:
        """Test interactive secret selection for update."""
        result = runner.invoke(
            app,
            ["env", "secret", "update", "testuser/test-env"],
            # Select item 1, then provide new value
            input="1\nnew-secret-value\n",
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Updated secret" in result.output

    def test_update_secret_interactive_cancel_selection(self, mock_env_secret_api: None) -> None:
        """Test cancelling interactive selection for update."""
        result = runner.invoke(
            app,
            ["env", "secret", "update", "testuser/test-env"],
            input="\n",
        )
        assert result.exit_code == 0
        assert "Cancelled" in result.output


class TestEnvSecretDelete:
    """Tests for the env secret delete command."""

    def test_delete_secret_with_confirm(self, mock_env_secret_api: None) -> None:
        """Test deleting an env secret with --yes flag."""
        result = runner.invoke(
            app,
            ["env", "secret", "delete", "testuser/test-env", "--id", "esecret-id-001", "-y"],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Deleted secret" in result.output
        assert "testuser/test-env" in result.output

    def test_delete_secret_cancelled(self, mock_env_secret_api: None) -> None:
        """Test cancelling env secret deletion."""
        result = runner.invoke(
            app,
            ["env", "secret", "delete", "testuser/test-env", "--id", "esecret-id-001"],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_delete_secret_interactive_select(self, mock_env_secret_api: None) -> None:
        """Test interactive secret selection for delete."""
        result = runner.invoke(
            app,
            ["env", "secret", "delete", "testuser/test-env"],
            # Select item 1, then confirm
            input="1\ny\n",
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Deleted secret" in result.output

    def test_delete_secret_interactive_cancel_selection(self, mock_env_secret_api: None) -> None:
        """Test cancelling interactive selection for delete."""
        result = runner.invoke(
            app,
            ["env", "secret", "delete", "testuser/test-env"],
            input="\n",
        )
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_delete_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test delete when no secrets exist."""
        monkeypatch.setenv("PRIME_API_KEY", "test-key")

        def mock_get(
            self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            if "/@latest" in endpoint:
                return {"data": {"id": "env-uuid-12345"}}
            return {"data": []}

        monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

        result = runner.invoke(app, ["env", "secret", "delete", "testuser/test-env"])
        assert result.exit_code == 0
        assert "No secrets to delete" in result.output


# ---------------------------------------------------------------------------
# Link
# ---------------------------------------------------------------------------


class TestEnvSecretLink:
    """Tests for the env secret link command."""

    def test_link_secret_success(self, mock_env_secret_api: None) -> None:
        """Test linking a global secret to an environment."""
        result = runner.invoke(
            app,
            ["env", "secret", "link", "global-secret-id-123", "testuser/test-env"],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Linked global secret" in result.output
        assert "testuser/test-env" in result.output

    def test_link_secret_json_output(self, mock_env_secret_api: None) -> None:
        """Test linking a secret with JSON output."""
        result = runner.invoke(
            app,
            [
                "env",
                "secret",
                "link",
                "global-secret-id-123",
                "testuser/test-env",
                "-o",
                "json",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        output = json.loads(result.output)
        assert "secretId" in output or "id" in output


class TestEnvSecretUnlink:
    """Tests for the env secret unlink command."""

    def test_unlink_secret_success(self, mock_env_secret_api: None) -> None:
        """Test unlinking a global secret from an environment."""
        result = runner.invoke(
            app,
            ["env", "secret", "unlink", "global-secret-id-123", "testuser/test-env", "-y"],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Unlinked global secret" in result.output
        assert "testuser/test-env" in result.output

    def test_unlink_secret_cancelled(self, mock_env_secret_api: None) -> None:
        """Test cancelling unlink confirmation."""
        result = runner.invoke(
            app,
            ["env", "secret", "unlink", "global-secret-id-123", "testuser/test-env"],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_unlink_secret_confirmed_interactively(self, mock_env_secret_api: None) -> None:
        """Test confirming unlink interactively."""
        result = runner.invoke(
            app,
            ["env", "secret", "unlink", "global-secret-id-123", "testuser/test-env"],
            input="y\n",
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Unlinked global secret" in result.output


class TestEnvSecretHelp:
    """Tests for help output."""

    def test_env_secret_help(self) -> None:
        """Test that env secret help lists all subcommands."""
        result = runner.invoke(app, ["env", "secret", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "list" in output
        assert "create" in output
        assert "update" in output
        assert "delete" in output
        assert "link" in output
        assert "unlink" in output

    def test_env_secret_list_help(self) -> None:
        """Test that env secret list help shows options."""
        result = runner.invoke(app, ["env", "secret", "list", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--output" in output

    def test_env_secret_create_help(self) -> None:
        """Test that env secret create help shows options."""
        result = runner.invoke(app, ["env", "secret", "create", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--name" in output
        assert "--value" in output
        assert "--description" in output

    def test_env_secret_link_help(self) -> None:
        """Test that env secret link help shows arguments."""
        result = runner.invoke(app, ["env", "secret", "link", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "GLOBAL_SECRET_ID" in output

    def test_env_secret_unlink_help(self) -> None:
        """Test that env secret unlink help shows arguments."""
        result = runner.invoke(app, ["env", "secret", "unlink", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "GLOBAL_SECRET_ID" in output
