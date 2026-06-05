from types import SimpleNamespace
from typing import Any

from prime_cli.api.deployments import DeploymentsClient
from prime_cli.client import APIError
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {"PRIME_API_KEY": "dummy", "PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}


def _adapter_response(
    *,
    adapter_id: str = "adapter-123",
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    deployment_status: str = "DEPLOYING",
) -> dict[str, Any]:
    return {
        "adapter": {
            "id": adapter_id,
            "displayName": "Checkpoint Adapter",
            "userId": "user-123",
            "teamId": None,
            "rftRunId": "run-123",
            "baseModel": base_model,
            "step": 20,
            "status": "READY",
            "deploymentStatus": deployment_status,
            "deployedAt": None,
            "deploymentError": None,
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
        },
        "message": "Checkpoint adapter deployment started",
    }


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


def test_deployments_client_deploy_checkpoint_posts_endpoint() -> None:
    captured: dict[str, Any] = {}

    class DummyAPIClient:
        def post(self, endpoint: str, json: dict[str, Any] | None = None) -> dict:
            captured["endpoint"] = endpoint
            captured["json"] = json
            return _adapter_response()

    adapter = DeploymentsClient(DummyAPIClient()).deploy_checkpoint("ckpt-123")

    assert captured["endpoint"] == "/rft/checkpoints/ckpt-123/deploy"
    assert captured["json"] is None
    assert adapter.id == "adapter-123"


def test_deployments_create_checkpoint_prints_adapter_result(monkeypatch) -> None:
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    adapter = SimpleNamespace(
        id="adapter-456",
        base_model="Qwen/Qwen3.5-0.8B",
        deployment_status="DEPLOYING",
    )

    class DummyDeploymentsClient:
        def __init__(self, api_client: Any) -> None:
            self.api_client = api_client

        def deploy_checkpoint(self, checkpoint_id: str) -> Any:
            assert checkpoint_id == "ckpt-456"
            return adapter

    monkeypatch.setattr("prime_cli.commands.deployments.APIClient", lambda: object())
    monkeypatch.setattr(
        "prime_cli.commands.deployments.DeploymentsClient",
        DummyDeploymentsClient,
    )

    result = runner.invoke(
        app,
        ["deployments", "create", "--checkpoint-id", "ckpt-456", "--yes"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "Deploying checkpoint:" in output
    assert "Checkpoint ID: ckpt-456" in output
    assert "Deployment initiated successfully!" in output
    assert "Adapter ID: adapter-456" in output
    assert "Status: DEPLOYING" in output
    assert '"Qwen/Qwen3.5-0.8B:adapter-456"' in output
    assert "prime deployments list" in output


def test_deployments_create_checkpoint_surfaces_conflict_errors(monkeypatch) -> None:
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    class DummyDeploymentsClient:
        def __init__(self, api_client: Any) -> None:
            self.api_client = api_client

        def deploy_checkpoint(self, checkpoint_id: str) -> Any:
            assert checkpoint_id == "ckpt-busy"
            raise APIError("HTTP 409: Checkpoint adapter preparation is already in progress")

    monkeypatch.setattr("prime_cli.commands.deployments.APIClient", lambda: object())
    monkeypatch.setattr(
        "prime_cli.commands.deployments.DeploymentsClient",
        DummyDeploymentsClient,
    )

    result = runner.invoke(
        app,
        ["deployments", "create", "--checkpoint-id", "ckpt-busy", "--yes"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 1
    assert "Error: HTTP 409" in output
    assert "Checkpoint adapter preparation is already in progress" in output
