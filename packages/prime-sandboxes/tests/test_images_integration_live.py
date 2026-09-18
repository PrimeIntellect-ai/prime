"""Live integration tests for image builds and the newer sandbox surfaces.

Opt-in via PRIME_LIVE_VM_SMOKE=1 (the convention in
test_live_process_idempotency_live.py); plain pytest runs never touch a real
backend. The backend is selected by the standard env overrides
(PRIME_API_KEY / PRIME_API_BASE_URL / PRIME_TEAM_ID, or PRIME_CONTEXT when
the config home is not patched), so the same suite runs against dev and prod.

Source-transfer tests exercise the pre-ENG-5865 backend wire contract: the
legacy bulk shape ``{sourceImage, success, buildId, fullImagePath}`` plus a
``failed[]`` mirror is normalized by ``SourceImageBuildResult``'s
``accept_legacy_result`` validator, and the single-source response carries
``buildId`` (accepted via the SDK's alias choices). The assertions on the
normalized objects prove the dual-accept path ran against the live legacy
wire, complementing the hermetic shape tests in test_images_client.py.
"""

import io
import os
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from prime_sandboxes import (
    APIError,
    BuildImageRequest,
    BuildImageResponse,
    BulkBuildImageResponse,
    CreateSandboxRequest,
    ImageClient,
    StartCommand,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("PRIME_LIVE_VM_SMOKE") != "1",
    reason="Live VM smoke tests are opt-in.",
)

# Tiny public images: the registry pull/push cost stays CPU-minutes scale.
SOURCE_IMAGE_OK = os.environ.get("PRIME_LIVE_SOURCE_IMAGE", "registry.k8s.io/pause:3.9")
SOURCE_IMAGE_OK_ALT = "registry.k8s.io/pause:3.10"

BUILD_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}
BUILD_POLL_SECONDS = 900
POLL_INTERVAL_SECONDS = 10


@pytest.fixture(scope="module")
def image_client():
    # Home-dir isolation, matching the platform VM E2E conftest: these tests
    # are driven by PRIME_API_KEY / PRIME_API_BASE_URL / PRIME_TEAM_ID, and an
    # unpatched config file would otherwise leak an unrelated team scope into
    # the image list/build requests.
    with patch("pathlib.Path.home", return_value=Path(tempfile.mkdtemp())):
        yield ImageClient()


def _team_id():
    """Team context for image operations: only an explicit PRIME_TEAM_ID.

    An empty PRIME_TEAM_ID means personal, not a team id of "".
    """
    return os.environ.get("PRIME_TEAM_ID") or None


def _poll_build_terminal(image_client, build_id):
    """Poll a build group until it reaches a terminal status."""
    deadline = time.monotonic() + BUILD_POLL_SECONDS
    status = None
    while time.monotonic() < deadline:
        status = image_client.get_build_status(build_id)
        if status["status"] in BUILD_TERMINAL_STATUSES:
            return status
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(
        f"build {build_id} did not reach a terminal status within "
        f"{BUILD_POLL_SECONDS}s (last: {status})"
    )


def _delete_image(image_client, image_name, image_tag, team_id):
    """Best-effort cleanup of an image this run created."""
    params = {"teamId": team_id} if team_id else None
    try:
        image_client.client.request("DELETE", f"/images/{image_name}/{image_tag}", params=params)
        print(f"[CLEANUP] deleted image {image_name}:{image_tag}")
    except APIError as exc:
        print(f"[CLEANUP] failed to delete {image_name}:{image_tag}: {exc}")


def _listed_rows(image_client, image_name, image_tag, team_id):
    listed = image_client.list(search=image_name, team_id=team_id)
    return [
        row for row in listed.data if row.image_name == image_name and row.image_tag == image_tag
    ]


def test_single_source_transfer_live(image_client):
    """A single-source transfer returns one build shaped like a dockerfile build."""
    team_id = _team_id()
    response = image_client.transfer_image(SOURCE_IMAGE_OK, team_id=team_id)
    assert isinstance(response, BuildImageResponse)
    # The legacy single-source wire uses the buildId key; the SDK accepts both.
    assert response.build_id
    assert response.build_ids and response.build_id in response.build_ids
    assert response.full_image_path

    # Non-Docker-Hub sources keep the source repository's last path segment
    # and its tag as the personal/team destination.
    image_name = SOURCE_IMAGE_OK.rsplit("/", 1)[-1].split(":")[0]
    image_tag = SOURCE_IMAGE_OK.rsplit(":", 1)[-1].split("@")[0]
    try:
        status = _poll_build_terminal(image_client, response.build_id)
        assert status["status"] == "COMPLETED", status
        assert _listed_rows(image_client, image_name, image_tag, team_id)
    finally:
        _delete_image(image_client, image_name, image_tag, team_id)


def test_bulk_source_transfer_partial_failure_live(image_client):
    """Comma-separated sources: ordered best-effort results, legacy shape normalized.

    The partial failure is a duplicate destination: a repeated source
    resolves to the same name:tag and the backend rejects the second copy
    per-source, without failing the first. (An unsupported registry source
    would be rejected for the whole request before any source is processed.)
    """
    team_id = _team_id()
    sources = f"{SOURCE_IMAGE_OK_ALT},{SOURCE_IMAGE_OK_ALT}"
    response = image_client.transfer_image(sources, team_id=team_id)
    assert isinstance(response, BulkBuildImageResponse)
    assert [result.source_image for result in response.results] == [
        SOURCE_IMAGE_OK_ALT,
        SOURCE_IMAGE_OK_ALT,
    ]
    ok, bad = response.results
    # Legacy rows carry flat success/buildId fields; normalization nests them
    # under build and leaves only the new shape on serialization.
    assert ok.build is not None
    assert ok.build.build_id
    assert ok.error is None
    assert bad.build is None
    assert "duplicate" in bad.error.lower()
    serialized = response.model_dump(by_alias=True)
    assert set(serialized) == {"results"}
    assert set(serialized["results"][0]) == {
        "sourceImage",
        "build",
        "error",
        "retryable",
    }

    try:
        status = _poll_build_terminal(image_client, ok.build.build_id)
        assert status["status"] == "COMPLETED", status
        assert _listed_rows(image_client, "pause", "3.10", team_id)
    finally:
        _delete_image(image_client, "pause", "3.10", team_id)


def test_dockerfile_build_full_flow_live(image_client):
    """Dockerfile build: initiate -> upload context -> start -> poll -> list."""
    team_id = _team_id()
    image_name = f"it-live-df-{uuid.uuid4().hex[:8]}"
    image_tag = "smoke"
    request = BuildImageRequest(image_name=image_name, image_tag=image_tag, team_id=team_id)
    response = image_client.initiate_build(request)
    assert isinstance(response, BuildImageResponse)
    assert response.build_id
    # requires_upload context: a dockerfile build must return upload metadata.
    assert response.upload_url
    assert response.expires_in

    try:
        dockerfile = f"FROM {SOURCE_IMAGE_OK}\n"
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            data = dockerfile.encode()
            info = tarfile.TarInfo(name="Dockerfile")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        upload = httpx.put(
            response.upload_url,
            content=buffer.getvalue(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=600,
        )
        assert upload.status_code == 200, upload.text

        image_client.start_build(response.build_id)
        status = _poll_build_terminal(image_client, response.build_id)
        assert status["status"] == "COMPLETED", status
        assert _listed_rows(image_client, image_name, image_tag, team_id)
    finally:
        _delete_image(image_client, image_name, image_tag, team_id)


@pytest.fixture(scope="module")
def egress_vm(sandbox_client):
    """One VM created with an egress allowlist and a structured start command."""
    sandbox = sandbox_client.create(
        CreateSandboxRequest(
            name=f"it-live-vm-{uuid.uuid4().hex[:8]}",
            docker_image="python:3.11-slim",
            start_command=StartCommand(
                executable="/bin/sh",
                args=["-c", "printf start-ok > /tmp/it-live-started; exec sleep 600"],
            ),
            network_allowlist=["api.primeintellect.ai"],
            cpu_cores=1,
            memory_gb=2,
            disk_size_gb=10,
            timeout_minutes=30,
            labels=["it-live"],
        )
    )
    try:
        sandbox_client.wait_for_creation(sandbox.id, max_attempts=120)
        yield sandbox
    finally:
        sandbox_client.delete(sandbox.id)


def test_structured_start_command_live(sandbox_client, egress_vm):
    """A structured start_command round-trips and its process actually runs."""
    start_command = egress_vm.start_command
    assert isinstance(start_command, StartCommand)
    assert start_command.executable == "/bin/sh"
    assert start_command.args[0] == "-c"
    # The start process may still be a moment behind the sandbox reaching
    # RUNNING, so poll for the marker instead of racing the boot.
    deadline = time.monotonic() + 30
    stdout = ""
    while time.monotonic() < deadline:
        marker = sandbox_client.execute_command(egress_vm.id, "cat /tmp/it-live-started")
        stdout = marker.stdout.strip()
        if marker.exit_code == 0 and stdout == "start-ok":
            break
        time.sleep(2)
    assert stdout == "start-ok"


def test_background_job_live(sandbox_client, egress_vm):
    """start_background_job returns a handle whose status polls to completion."""
    job = sandbox_client.start_background_job(egress_vm.id, "echo bg-job-ok", working_dir="/tmp")
    deadline = time.monotonic() + 120
    status = None
    while time.monotonic() < deadline:
        status = sandbox_client.get_background_job(egress_vm.id, job)
        if status.completed:
            break
        time.sleep(2)
    assert status is not None and status.completed, status
    assert status.exit_code == 0
    assert status.stdout and status.stdout.strip() == "bg-job-ok"


def test_egress_policy_live(sandbox_client, egress_vm):
    """Egress rules survive create, and set_network replaces them atomically."""
    status = sandbox_client.get_network(egress_vm.id)
    assert status.policy.allowlist == ["api.primeintellect.ai"]
    assert status.policy.denylist is None
    # The create-time policy counts as generation 0 and is applied at boot.
    assert status.applied

    # Deny-all is represented as an empty allowlist with no denylist.
    replaced = sandbox_client.set_network(egress_vm.id, deny=["*"])
    assert replaced.policy.allowlist == []
    assert replaced.policy.denylist is None
    assert replaced.generation > status.generation

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        current = sandbox_client.get_network(egress_vm.id)
        if current.applied:
            break
        time.sleep(2)
    assert current.applied, current
    assert current.policy.allowlist == []
    assert current.policy.denylist is None
    assert current.applied_generation == replaced.generation
