import json
import os
from typing import Any, Dict

import pytest
from prime_cli.main import app
from prime_core import APIClient
from typer.testing import CliRunner


@pytest.fixture
def mock_api_client(monkeypatch: pytest.MonkeyPatch) -> APIClient:
    # Set the environment variable
    monkeypatch.setenv("PRIME_API_KEY", "dummy")

    # Get the absolute path to the test data files
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(test_dir, "data", "availability_response.json")
    summary_file = os.path.join(test_dir, "data", "availability_summary.json")
    disks_file = os.path.join(test_dir, "data", "availability_disks.json")

    with open(data_file, "r") as f:
        api_response_json: Dict[str, Any] = json.load(f)

    with open(summary_file, "r") as f:
        summary_response_json: Dict[str, Any] = json.load(f)

    with open(disks_file, "r") as f:
        disks_response_json: Dict[str, Any] = json.load(f)

    # Create a mock get function that returns different data based on endpoint
    def mock_get(self: Any, endpoint: str, params: Any = None) -> Dict[str, Any]:
        if endpoint == "/availability" or endpoint == "availability":
            return api_response_json
        elif endpoint == "/availability/gpu-summary" or endpoint == "availability/gpu-summary":
            return summary_response_json
        elif endpoint == "/availability/disks" or endpoint == "availability/disks":
            return disks_response_json
        else:
            # Default fallback
            return api_response_json

    # Create a mock client that returns the test data
    mock_client = APIClient("dummy_url")
    # Patch the APIClient.get method to use our mock function
    monkeypatch.setattr("prime_core.APIClient.get", mock_get)
    return mock_client


def test_availability_list(mock_api_client: APIClient, capsys: pytest.CaptureFixture[str]) -> None:
    runner = CliRunner()

    # Invoke the CLI command with wide terminal to prevent truncation
    result = runner.invoke(app, ["availability", "list"], env={"COLUMNS": "200"})

    # Check the exit code is 0 (success)
    assert result.exit_code == 0, f"Failed: {result.exit_code}\n{result.output}"

    # Verify table headers are present (accounting for possible truncation)
    assert "Available GPU Resources" in result.output
    assert "ID" in result.output
    assert "GPU Type" in result.output
    assert "vCPU" in result.output
    assert "RAM" in result.output
    assert "Disk" in result.output

    # Verify deployment instructions are present
    assert "To deploy a pod with one of these configurations:" in result.output


def test_availability_gpu_types(
    mock_api_client: APIClient, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = CliRunner()

    # Invoke the CLI command with wide terminal to prevent truncation
    result = runner.invoke(app, ["availability", "gpu-types"], env={"COLUMNS": "200"})

    # Check the exit code is 0 (success)
    assert result.exit_code == 0, f"Failed: {result.exit_code}\n{result.output}"

    # Verify table headers are present
    assert "GPU Type" in result.output

    # Verify some expected data points (GPU types use underscores)
    assert "A100_80GB" in result.output
    assert "H100_80GB" in result.output


def test_availability_disks(mock_api_client: APIClient, capsys: pytest.CaptureFixture[str]) -> None:
    runner = CliRunner()

    # Invoke the CLI command with wide terminal to prevent truncation
    # Rich respects COLUMNS and TERM environment variables
    result = runner.invoke(app, ["availability", "disks"], env={"COLUMNS": "250", "LINES": "50"})

    # Check the exit code is 0 (success)
    assert result.exit_code == 0, f"Failed: {result.exit_code}\n{result.output}"

    # Verify table headers are present (may be truncated depending on terminal width)
    assert "Available Disks" in result.output
    assert "ID" in result.output
    assert "Provider" in result.output
    assert "Location" in result.output
    assert "Stock" in result.output
    assert "Price" in result.output  # May be "Price/Hr/GB" or "Price/H…"
    assert "Max Size" in result.output
    assert "Multino" in result.output  # May be "Is Multinode" or "Multino…"

    # Verify some expected data points from the test data
    # (check for partial strings to handle potential truncation)
    assert "runpod" in result.output
    assert "hyperstack" in result.output  # "hyperstack" may be truncated
    assert "crusoecloud" in result.output  # "crusoecloud" may be truncated
    assert "dc_roan" in result.output
    assert "US" in result.output
    assert "NO" in result.output
    assert "IS" in result.output
    assert "Available" in result.output

    # Verify instructions are present
    assert "To create a disk with one of these configurations:" in result.output
