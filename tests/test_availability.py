import json
import os
from typing import Any, Dict

import pytest
from prime_cli.api.client import APIClient
from prime_cli.main import app
from typer.testing import CliRunner


@pytest.fixture
def mock_api_client(monkeypatch: pytest.MonkeyPatch) -> APIClient:
    # Get the absolute path to the test data file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(test_dir, "data", "availability_response.json")

    with open(data_file, "r") as f:
        api_response_json: Dict[str, Any] = json.load(f)

    # Create a mock client that returns the test data
    mock_client = APIClient("dummy_url")
    # Patch the APIClient class itself, not just an instance
    monkeypatch.setattr(
        "prime_cli.api.client.APIClient.get", lambda *args, **kwargs: api_response_json
    )
    return mock_client


def test_availability_list(mock_api_client: APIClient, capsys: pytest.CaptureFixture[str]) -> None:
    runner = CliRunner()

    # Invoke the CLI command
    result = runner.invoke(app, ["availability", "list"])

    # Check the exit code is 0 (success)
    assert result.exit_code == 0, f"Failed: {result.exit_code}\n{result.output}"

    # Verify table headers are present
    assert "Available GPU Resources" in result.output
    assert "ID" in result.output
    assert "GPU Type" in result.output
    assert "GPUs" in result.output
    assert "Socket" in result.output
    assert "Provider" in result.output
    assert "Location" in result.output
    assert "Stock" in result.output
    assert "Price/Hr" in result.output
    assert "Memory (GB)" in result.output
    assert "Security" in result.output
    assert "vCPUs" in result.output
    assert "RAM (GB)" in result.output

    # Verify some expected data points
    assert "A40_48GB" in result.output
    assert "A100_80GB" in result.output
    assert "PCIe" in result.output
    assert "SXM4" in result.output
    assert "runpod" in result.output
    assert "tensordock" in result.output
    assert "secure_cloud" in result.output
    assert "community_cloud" in result.output

    # Verify deployment instructions are present
    assert "To deploy a pod with one of these configurations:" in result.output
