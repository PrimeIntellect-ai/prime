from types import SimpleNamespace
from typing import Any

from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()


def test_deployments_create_prints_login_aware_chat_command(monkeypatch) -> None:
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
    assert "prime login" in output
    assert "PRIME_API_KEY" in output
    assert "curl -X POST" not in output
