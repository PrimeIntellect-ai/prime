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
    assert request.vm is False
    assert request.timeout_minutes == 60
    assert request.region is None
    assert request.labels == []


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


def test_create_sandbox_request_requires_vm_for_gpu_count():
    """Test vm is required when gpu_count > 0"""
    with pytest.raises(ValidationError):
        CreateSandboxRequest(
            name="gpu-sandbox",
            docker_image="python:3.11-slim",
            gpu_count=1,
            gpu_type="H100_80GB",
        )


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


def test_vm_sandbox_accepts_environment_variables_and_secrets():
    request = CreateSandboxRequest(
        name="vm-sandbox",
        docker_image="python:3.11-slim",
        vm=True,
        environment_vars={"PUBLIC_VALUE": "hello"},
        secrets={"API_TOKEN": "secret"},
    )

    assert request.environment_vars == {"PUBLIC_VALUE": "hello"}
    assert request.secrets == {"API_TOKEN": "secret"}


@pytest.mark.parametrize(
    ("environment_vars", "secrets", "message"),
    [
        ({"1INVALID": "value"}, None, "must match"),
        ({"SANDBOX_ID": "value"}, None, "reserved"),
        (None, {"TOKEN": "before\x00after"}, "NUL"),
        ({"TOKEN": "public"}, {"TOKEN": "secret"}, "both"),
        (None, {"TOKEN": "x" * (16 * 1024 + 1)}, "16384-byte limit"),
    ],
)
def test_vm_sandbox_rejects_invalid_environment_contract(environment_vars, secrets, message):
    with pytest.raises(ValidationError, match=message):
        CreateSandboxRequest(
            name="vm-sandbox",
            docker_image="python:3.11-slim",
            vm=True,
            environment_vars=environment_vars,
            secrets=secrets,
        )


def test_vm_sandbox_validation_error_hides_secret_input():
    secret = "super-sensitive-secret-value"

    with pytest.raises(ValidationError, match="reserved") as exc_info:
        CreateSandboxRequest(
            name="vm-sandbox",
            docker_image="python:3.11-slim",
            vm=True,
            secrets={"SANDBOX_ID": secret},
        )

    assert secret not in str(exc_info.value)


def test_vm_sandbox_rejects_too_many_combined_variables():
    with pytest.raises(ValidationError, match="257 variables"):
        CreateSandboxRequest(
            name="vm-sandbox",
            docker_image="python:3.11-slim",
            vm=True,
            environment_vars={f"PUBLIC_{index}": "x" for index in range(128)},
            secrets={f"SECRET_{index}": "x" for index in range(129)},
        )


def test_vm_sandbox_rejects_oversized_serialized_secret_payload():
    # The key/value total is below 32 KiB, but JSON framing pushes the KMS
    # plaintext beyond its independent 32 KiB limit.
    secrets = {f"K{index}": "x" * 155 for index in range(200)}

    with pytest.raises(ValidationError, match="serialized VM secrets"):
        CreateSandboxRequest(
            name="vm-sandbox",
            docker_image="python:3.11-slim",
            vm=True,
            secrets=secrets,
        )


def test_container_sandbox_keeps_existing_environment_contract():
    request = CreateSandboxRequest(
        name="container-sandbox",
        docker_image="python:3.11-slim",
        environment_vars={"container.style.key": "value"},
    )

    assert request.environment_vars == {"container.style.key": "value"}


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
