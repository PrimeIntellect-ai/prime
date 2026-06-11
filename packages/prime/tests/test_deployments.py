from types import SimpleNamespace
from typing import Any

from prime_cli.api.deployments import DeploymentsClient
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()


def test_deployments_create_prints_chat_and_api_key_commands(monkeypatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    model = SimpleNamespace(
        id="adapter-123",
        display_name="Adapter",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        status="READY",
        deployment_status="NOT_DEPLOYED",
    )
    updated_model = SimpleNamespace(deployment_status="DEPLOYING")

    class DummyDeploymentsClient:
        def __init__(self, api_client: Any) -> None:
            self.api_client = api_client

        def get_adapter(self, model_id: str) -> Any:
            assert model_id == "adapter-123"
            return model

        def get_deployable_models(self) -> list[str]:
            return [model.base_model]

        def deploy_adapter(self, model_id: str) -> Any:
            assert model_id == "adapter-123"
            return updated_model

    monkeypatch.setattr("prime_cli.commands.deployments.APIClient", lambda: object())
    monkeypatch.setattr(
        "prime_cli.commands.deployments.DeploymentsClient",
        DummyDeploymentsClient,
    )

    result = runner.invoke(app, ["deployments", "create", "adapter-123", "--yes"])
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "prime inference chat" in output
    assert '"meta-llama/Llama-3.1-8B-Instruct:adapter-123"' in output
    assert '"Hello" --max-tokens 100' in output
    assert "For scripts or API clients" in output
    assert "prime login" not in output
    assert "https://docs.primeintellect.ai/api-reference/api-keys" in output
    assert "export PRIME_API_KEY=<insert_key_here>" in output
    assert "PRIME_API_KEY" in output
    assert "curl -X POST" in output


def test_update_adapter_project_sends_backend_payload_shape() -> None:
    class FakeAPIClient:
        def __init__(self) -> None:
            self.requests: list[tuple[str, str, dict[str, Any] | None]] = []

        def request(
            self,
            method: str,
            endpoint: str,
            json: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            self.requests.append((method, endpoint, json))
            return {
                "adapter": {
                    "id": "adapter-123",
                    "userId": "user-1",
                    "projectId": json.get("projectId") if json else None,
                    "rftRunId": "run-123",
                    "baseModel": "Qwen/Qwen3.5-0.8B",
                    "status": "READY",
                    "deploymentStatus": "NOT_DEPLOYED",
                    "createdAt": "2026-05-17T00:00:00Z",
                    "updatedAt": "2026-05-17T00:00:00Z",
                }
            }

    api_client = FakeAPIClient()
    client = DeploymentsClient(api_client)  # type: ignore[arg-type]

    adapter = client.update_adapter_project(
        "adapter-123",
        "project-123",
        operation="add",
    )

    assert api_client.requests == [
        (
            "PATCH",
            "/rft/adapters/adapter-123/project",
            {"projectId": "project-123", "operation": "add"},
        )
    ]
    assert adapter.project_id == "project-123"
