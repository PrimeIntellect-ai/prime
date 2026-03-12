"""Tests for Pydantic models"""

import pytest
from pydantic import ValidationError

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
    assert request.gpu_type is None
    assert request.timeout_minutes == 60
    assert request.labels == []


def test_create_sandbox_request_requires_gpu_type_for_gpu_count():
    """Test gpu_type is required when gpu_count > 0"""
    with pytest.raises(ValidationError):
        CreateSandboxRequest(
            name="gpu-sandbox",
            docker_image="python:3.11-slim",
            gpu_count=1,
        )


def test_create_sandbox_request_accepts_gpu_type_for_gpu_count():
    """Test gpu_type is accepted for GPU sandbox requests"""
    request = CreateSandboxRequest(
        name="gpu-sandbox",
        docker_image="python:3.11-slim",
        gpu_count=1,
        gpu_type="H100_80GB",
    )

    assert request.gpu_count == 1
    assert request.gpu_type == "H100_80GB"


def test_create_sandbox_request_gpu_type_none_matches_default():
    """Test explicit gpu_type=None behaves like omitting gpu_type"""
    request_default = CreateSandboxRequest(
        name="cpu-sandbox-default",
        docker_image="python:3.11-slim",
    )
    request_none = CreateSandboxRequest(
        name="cpu-sandbox-none",
        docker_image="python:3.11-slim",
        gpu_type=None,
    )

    assert request_default.gpu_type is None
    assert request_none.gpu_type is None


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
        "gpuCount": 1,
        "gpuType": "H100_80GB",
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
    assert sandbox.status == "RUNNING"
    assert sandbox.gpu_type == "H100_80GB"
