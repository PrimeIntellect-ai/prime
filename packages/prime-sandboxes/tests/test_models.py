"""Tests for Pydantic models"""

from prime_sandboxes.models import (
    CreateSandboxRequest,
    Sandbox,
    SandboxStatus,
)


def test_create_sandbox_request_defaults():
    """Test default values for CreateSandboxRequest"""
    request = CreateSandboxRequest(
        name="test-sandbox",
        docker_image="python:3.11-slim",
    )

    assert request.name == "test-sandbox"
    assert request.docker_image == "python:3.11-slim"
    assert request.cpu_cores == 1
    assert request.memory_gb == 2
    assert request.disk_size_gb == 5
    assert request.gpu_count == 0
    assert request.timeout_minutes == 60
    assert request.labels == []


def test_sandbox_status_enum():
    """Test SandboxStatus enum values"""
    assert SandboxStatus.PENDING == "PENDING"
    assert SandboxStatus.RUNNING == "RUNNING"
    assert SandboxStatus.TERMINATED == "TERMINATED"


def test_sandbox_model_with_alias():
    """Test Sandbox model handles API field aliases"""
    data = {
        "id": "test-123",
        "name": "test-sandbox",
        "dockerImage": "python:3.11-slim",
        "cpuCores": 2,
        "memoryGB": 4,
        "diskSizeGB": 10,
        "diskMountPath": "/workspace",
        "gpuCount": 0,
        "status": "RUNNING",
        "timeoutMinutes": 120,
        "labels": ["test"],
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
    }

    sandbox = Sandbox.model_validate(data)

    assert sandbox.id == "test-123"
    assert sandbox.name == "test-sandbox"
    assert sandbox.cpu_cores == 2
    assert sandbox.memory_gb == 4
    assert sandbox.disk_size_gb == 10
    assert sandbox.status == "RUNNING"


def test_create_sandbox_request_float_disk_size():
    """Test that disk_size_gb accepts float values"""
    request = CreateSandboxRequest(
        name="test-sandbox",
        docker_image="python:3.11-slim",
        disk_size_gb=0.5,
    )
    assert request.disk_size_gb == 0.5


def test_create_sandbox_request_fractional_disk_sizes():
    """Test various fractional disk size values"""
    for size in [0.5, 1.5, 2.25, 10.5]:
        request = CreateSandboxRequest(
            name="test-sandbox",
            docker_image="python:3.11-slim",
            disk_size_gb=size,
        )
        assert request.disk_size_gb == size


def test_sandbox_model_float_disk_size():
    """Test Sandbox model handles float diskSizeGB from API"""
    data = {
        "id": "test-123",
        "name": "test-sandbox",
        "dockerImage": "python:3.11-slim",
        "cpuCores": 2,
        "memoryGB": 4,
        "diskSizeGB": 0.5,
        "diskMountPath": "/workspace",
        "gpuCount": 0,
        "status": "RUNNING",
        "timeoutMinutes": 120,
        "labels": ["test"],
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
    }

    sandbox = Sandbox.model_validate(data)
    assert sandbox.disk_size_gb == 0.5
