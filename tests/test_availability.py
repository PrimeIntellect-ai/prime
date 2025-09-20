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

    # Verify table headers are present (accounting for possible truncation)
    assert "Available GPU Resources" in result.output
    assert "ID" in result.output
    assert "GPU" in result.output  # May be truncated to "GPU Ty..."
    assert "vC" in result.output  # vCPUs may be truncated to "vC..."
    assert "RAM" in result.output  # May be truncated
    assert "Di" in result.output  # Disk may be truncated to "Di..."

    # Verify some expected data points (accounting for possible truncation)
    assert "A40" in result.output
    assert "A100" in result.output or "A1" in result.output  # May be truncated
    assert "PC" in result.output or "PCIe" in result.output  # May be truncated
    assert "SX" in result.output or "SXM" in result.output  # May be truncated
    assert "run" in result.output  # runpod may be truncated
    assert "dat" in result.output or "datacenter" in result.output  # May be truncated
    assert "com" in result.output or "community" in result.output  # May be truncated

    # Verify deployment instructions are present
    assert "To deploy a pod with one of these configurations:" in result.output


def test_availability_gpu_types(
    mock_api_client: APIClient, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = CliRunner()

    # Invoke the CLI command
    result = runner.invoke(app, ["availability", "gpu-types"])

    # Check the exit code is 0 (success)
    assert result.exit_code == 0, f"Failed: {result.exit_code}\n{result.output}"

    # Verify table headers are present
    assert "GPU Type" in result.output

    # Verify some expected data points
    assert "A100 80GB" in result.output
    assert "A40 48GB" in result.output
