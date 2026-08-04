"""Tests for Pydantic models"""

import pytest
from pydantic import ValidationError

from prime_sandboxes.models import (
    CreateSandboxRequest,
    Sandbox,
    SandboxStatus,
    StartCommand,
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
    assert request.memory_gb == 1
    assert request.disk_size_gb == 5
    assert request.gpu_count == 0
    assert request.gpu_type is None
    # Unset vm defers the runtime choice to the server (platform default: VM).
    assert request.vm is None
    assert request.timeout_minutes == 60
    assert request.region is None
    assert request.labels == []
    assert request.start_command == "tail -f /dev/null"


def test_unset_vm_is_omitted_from_payload():
    """Unset vm must be excluded from the API payload so the server default applies."""
    request = CreateSandboxRequest(
        name="test-sandbox",
        docker_image="python:3.11-slim",
    )

    assert "vm" not in request.model_dump(exclude_none=True)


def test_explicit_vm_false_is_serialized():
    """The explicit container opt-out must survive exclude_none serialization."""
    request = CreateSandboxRequest(
        name="test-sandbox",
        docker_image="python:3.11-slim",
        vm=False,
    )

    assert request.model_dump(exclude_none=True)["vm"] is False


def test_unset_vm_rejects_explicit_string_start_command():
    """Explicit string start commands are container-only; require vm=False."""
    with pytest.raises(ValidationError, match="container-only"):
        CreateSandboxRequest(
            name="test-sandbox",
            docker_image="python:3.11-slim",
            start_command="sleep infinity",
        )


def test_container_opt_out_keeps_explicit_string_start_command():
    request = CreateSandboxRequest(
        name="test-sandbox",
        docker_image="python:3.11-slim",
        vm=False,
        start_command="sleep infinity",
    )

    assert request.start_command == "sleep infinity"


def test_vm_start_command_preserves_argv():
    request = CreateSandboxRequest(
        name="vm-workload",
        docker_image="team/image:v1",
        vm=True,
        start_command=StartCommand(
            executable="/worker",
            args=["--platform", "linux/amd64", "value with spaces"],
        ),
    )

    assert request.model_dump(exclude_none=True)["start_command"] == {
        "executable": "/worker",
        "args": ["--platform", "linux/amd64", "value with spaces"],
    }


def test_vm_without_start_command_does_not_inherit_container_default():
    request = CreateSandboxRequest(
        name="interactive-vm",
        docker_image="team/image:v1",
        vm=True,
    )

    assert request.start_command is None


def test_vm_rejects_legacy_string_start_command():
    with pytest.raises(ValidationError):
        CreateSandboxRequest(
            name="vm-workload",
            docker_image="team/image:v1",
            vm=True,
            start_command="/worker --platform linux/amd64",
        )


def test_create_sandbox_request_accepts_region():
    """Test region is accepted for multi-cluster sandbox creation"""
    request = CreateSandboxRequest(
        name="regional-sandbox",
        docker_image="python:3.11-slim",
        region="eu-west",
    )

    assert request.region == "eu-west"
    assert request.model_dump(exclude_none=True)["region"] == "eu-west"


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
        vm=True,
    )

    assert request.gpu_count == 1
    assert request.gpu_type == "H100_80GB"
    assert request.vm is True


def test_create_sandbox_request_rejects_gpu_with_container_opt_out():
    """GPUs conflict with an explicit container opt-out (vm=False)"""
    with pytest.raises(ValidationError):
        CreateSandboxRequest(
            name="gpu-sandbox",
            docker_image="python:3.11-slim",
            gpu_count=1,
            gpu_type="H100_80GB",
            vm=False,
        )


def test_create_sandbox_request_allows_gpu_with_unset_vm():
    """Unset vm defers to the platform default (VM), which supports GPUs"""
    request = CreateSandboxRequest(
        name="gpu-sandbox",
        docker_image="python:3.11-slim",
        gpu_count=1,
        gpu_type="H100_80GB",
    )

    assert request.vm is None
    assert request.gpu_count == 1


def test_create_sandbox_request_rejects_gpu_type_without_gpu_count():
    """Test gpu_type is rejected when gpu_count is zero"""
    with pytest.raises(ValidationError):
        CreateSandboxRequest(
            name="cpu-sandbox",
            docker_image="python:3.11-slim",
            gpu_type="H100_80GB",
            vm=True,
        )


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


def test_guaranteed_requires_container_opt_out():
    """guaranteed is container-only; unset vm resolves to VM on the server"""
    with pytest.raises(ValidationError, match="vm=False"):
        CreateSandboxRequest(
            name="guaranteed-sandbox",
            docker_image="python:3.11-slim",
            guaranteed=True,
        )

    request = CreateSandboxRequest(
        name="guaranteed-sandbox",
        docker_image="python:3.11-slim",
        guaranteed=True,
        vm=False,
    )
    assert request.guaranteed is True


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
        "vm": True,
        "status": "RUNNING",
        "timeoutMinutes": 120,
        "labels": ["test"],
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
        "region": "eu-west",
    }

    sandbox = Sandbox.model_validate(data)

    assert sandbox.id == "test-123"
    assert sandbox.name == "test-sandbox"
    assert sandbox.cpu_cores == 2
    assert sandbox.memory_gb == 4
    assert sandbox.status == "RUNNING"
    assert sandbox.gpu_type == "H100_80GB"
    assert sandbox.vm is True
    assert sandbox.region == "eu-west"


def test_image_update_source_forms_are_mutually_exclusive():
    import pytest

    from prime_sandboxes import ImageUpdateSource, PersonalImageOwner

    with pytest.raises(ValueError):
        ImageUpdateSource(owner=PersonalImageOwner(), name="app", tag="v1", reference="app:v1")
    with pytest.raises(ValueError):
        ImageUpdateSource(owner=PersonalImageOwner(), name="app")
    assert ImageUpdateSource(reference="prime/alice/app:v1").reference


def test_image_update_patch_requires_a_change():
    import pytest

    from prime_sandboxes import ImageUpdatePatch

    with pytest.raises(ValueError):
        ImageUpdatePatch()


def test_image_update_patch_rejects_private_platform():
    import pytest

    from prime_sandboxes import ImageUpdatePatch, ImageVisibility, PlatformImageOwner

    with pytest.raises(ValueError):
        ImageUpdatePatch(owner=PlatformImageOwner(), visibility=ImageVisibility.PRIVATE)


def test_update_images_request_serializes_camel_case_aliases():
    from prime_sandboxes import (
        ImageUpdateItem,
        ImageUpdatePatch,
        ImageUpdateSource,
        TeamImageOwner,
        UpdateImagesRequest,
    )

    request = UpdateImagesRequest(
        dry_run=True,
        updates=[
            ImageUpdateItem(
                source=ImageUpdateSource(
                    owner=TeamImageOwner(team_id="team1"), name="app", tag="v1"
                ),
                set=ImageUpdatePatch(name="renamed"),
            )
        ],
    )
    payload = request.model_dump(by_alias=True, exclude_none=True)
    assert payload["dryRun"] is True
    assert payload["updates"][0]["source"]["owner"] == {
        "type": "team",
        "teamId": "team1",
    }


def test_update_images_response_parses_owner_union():
    from prime_sandboxes import PlatformImageOwner, UpdateImagesResponse

    response = UpdateImagesResponse.model_validate(
        {
            "success": True,
            "dryRun": False,
            "results": [
                {
                    "source": {"owner": {"type": "personal"}, "name": "a", "tag": "b"},
                    "success": True,
                    "after": {
                        "owner": {"type": "platform"},
                        "name": "a",
                        "tag": "b",
                        "visibility": "PUBLIC",
                    },
                }
            ],
        }
    )
    assert isinstance(response.results[0].after.owner, PlatformImageOwner)
