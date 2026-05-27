from types import SimpleNamespace
from typing import Any

import pytest
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()


def test_sandbox_create_with_gpu_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    captured: dict[str, Any] = {}

    def mock_create(self: Any, request: Any) -> Any:
        captured["request"] = request
        return SimpleNamespace(id="sbx-gpu-123")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.create", mock_create)

    result = runner.invoke(
        app,
        [
            "sandbox",
            "create",
            "team-1/gpu-runtime:v1",
            "--vm",
            "--gpu-count",
            "1",
            "--gpu-type",
            "H100_80GB",
            "--yes",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Successfully created sandbox sbx-gpu-123" in output
    assert "VM: Enabled" in output
    assert "GPUs: H100_80GB x1" in output
    assert "Docker Image: team-1/gpu-runtime:v1" in output
    assert captured["request"].docker_image == "team-1/gpu-runtime:v1"
    assert captured["request"].gpu_count == 1
    assert captured["request"].gpu_type == "H100_80GB"
    assert captured["request"].vm is True


def test_sandbox_create_gpu_without_docker_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    captured: dict[str, Any] = {}

    def mock_create(self: Any, request: Any) -> Any:
        captured["request"] = request
        return SimpleNamespace(id="sbx-gpu-default-image")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.create", mock_create)

    result = runner.invoke(
        app,
        [
            "sandbox",
            "create",
            "--gpu-count",
            "1",
            "--gpu-type",
            "RTX_PRO_6000",
            "--yes",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 1
    assert "GPUs require VM sandboxes." in output
    assert "Successfully created sandbox" not in output
    assert "request" not in captured


def test_sandbox_create_accepts_docker_image_for_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    captured: dict[str, Any] = {}

    def mock_create(self: Any, request: Any) -> Any:
        captured["request"] = request
        return SimpleNamespace(id="sbx-gpu-with-image")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.create", mock_create)

    result = runner.invoke(
        app,
        [
            "sandbox",
            "create",
            "python:3.11-slim",
            "--vm",
            "--gpu-count",
            "1",
            "--gpu-type",
            "H100_80GB",
            "--yes",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Successfully created sandbox sbx-gpu-with-image" in output
    assert captured["request"].docker_image == "python:3.11-slim"
    assert captured["request"].gpu_count == 1
    assert captured["request"].gpu_type == "H100_80GB"
    assert captured["request"].vm is True


def test_sandbox_create_accepts_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    captured: dict[str, Any] = {}

    def mock_create(self: Any, request: Any) -> Any:
        captured["request"] = request
        return SimpleNamespace(id="sbx-eu-west")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.create", mock_create)

    result = runner.invoke(
        app,
        [
            "sandbox",
            "create",
            "python:3.11-slim",
            "--region",
            "eu-west",
            "--yes",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Successfully created sandbox sbx-eu-west" in output
    assert "Region: eu-west" in output
    assert captured["request"].region == "eu-west"


def test_sandbox_create_requires_gpu_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    called = False

    def mock_create(self: Any, request: Any) -> Any:
        nonlocal called
        called = True
        return SimpleNamespace(id="sbx-should-not-create")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.create", mock_create)

    result = runner.invoke(
        app,
        ["sandbox", "create", "--gpu-count", "1", "--yes"],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 1
    assert "GPU type is required when requesting GPUs." in output
    assert called is False


def test_sandbox_create_requires_vm_for_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    called = False

    def mock_create(self: Any, request: Any) -> Any:
        nonlocal called
        called = True
        return SimpleNamespace(id="sbx-should-not-create")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.create", mock_create)

    result = runner.invoke(
        app,
        [
            "sandbox",
            "create",
            "python:3.11-slim",
            "--gpu-count",
            "1",
            "--gpu-type",
            "H100_80GB",
            "--yes",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 1
    assert "GPUs require VM sandboxes." in output
    assert called is False


def test_sandbox_create_rejects_gpu_type_without_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    called = False

    def mock_create(self: Any, request: Any) -> Any:
        nonlocal called
        called = True
        return SimpleNamespace(id="sbx-should-not-create")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.create", mock_create)

    result = runner.invoke(
        app,
        ["sandbox", "create", "--gpu-type", "H100_80GB", "--yes"],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 1
    assert "GPU type provided without GPUs." in output
    assert called is False


def test_sandbox_create_requires_docker_image_for_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    called = False

    def mock_create(self: Any, request: Any) -> Any:
        nonlocal called
        called = True
        return SimpleNamespace(id="sbx-should-not-create")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.create", mock_create)

    result = runner.invoke(app, ["sandbox", "create", "--yes"])

    output = strip_ansi(result.output)
    assert result.exit_code == 1
    assert "Docker image is required." in output
    assert called is False


def test_sandbox_create_vm_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    captured: dict[str, Any] = {}

    def mock_create(self: Any, request: Any) -> Any:
        captured["request"] = request
        return SimpleNamespace(id="sbx-vm-123")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.create", mock_create)

    result = runner.invoke(
        app,
        ["sandbox", "create", "user-1/vm-image:latest", "--vm", "--yes"],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Successfully created sandbox sbx-vm-123" in output
    assert captured["request"].vm is True
    assert captured["request"].gpu_count == 0


def test_sandbox_delete_by_label_scopes_to_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default --label delete scopes to the caller's user_id server-side.

    The CLI now makes a single ``list(per_page=1)`` call to preview the count,
    then a single ``bulk_delete`` with the scope/labels — the server does the
    filtering, so no pagination and no client-side user_id filter.
    """
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setenv("PRIME_USER_ID", "user-1")

    captured: dict[str, Any] = {}

    def mock_list(self: Any, **kwargs: Any) -> Any:
        captured["list_kwargs"] = kwargs
        return SimpleNamespace(
            sandboxes=[SimpleNamespace(id="sbx-owned", user_id="user-1")],
            total=1,
            page=1,
            per_page=1,
            has_next=False,
        )

    def mock_bulk_delete(self: Any, **kwargs: Any) -> Any:
        captured["bulk_delete_kwargs"] = kwargs
        return SimpleNamespace(
            succeeded=["sbx-owned"],
            failed=[],
            message="deleted",
        )

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.list", mock_list)
    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.bulk_delete", mock_bulk_delete)

    result = runner.invoke(app, ["sandbox", "delete", "--label", "keep", "--yes"])

    output = strip_ansi(result.output)
    assert result.exit_code == 0, f"Failed: {result.output}"

    # Preview call: scoped to caller + labels, only active sandboxes
    list_kwargs = captured["list_kwargs"]
    assert list_kwargs["labels"] == ["keep"]
    assert list_kwargs["user_id"] == "user-1"
    assert list_kwargs["exclude_terminated"] is True
    assert list_kwargs["per_page"] == 1

    # Delete call: server-side scope, no client-side ID list
    bulk_kwargs = captured["bulk_delete_kwargs"]
    assert bulk_kwargs["labels"] == ["keep"]
    assert bulk_kwargs["user_id"] == "user-1"
    assert bulk_kwargs["all_users"] is False
    assert bulk_kwargs.get("sandbox_ids") is None

    assert "Successfully deleted 1 sandbox(es):" in output


def test_sandbox_delete_by_label_all_users_passes_admin_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--all-users flips user_id scoping to all_users=True (server admin-gated)."""
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setenv("PRIME_USER_ID", "user-1")

    captured: dict[str, Any] = {}

    def mock_list(self: Any, **kwargs: Any) -> Any:
        captured["list_kwargs"] = kwargs
        return SimpleNamespace(
            sandboxes=[SimpleNamespace(id="sbx-active", user_id="user-1")],
            total=1,
            page=1,
            per_page=1,
            has_next=False,
        )

    def mock_bulk_delete(self: Any, **kwargs: Any) -> Any:
        captured["bulk_delete_kwargs"] = kwargs
        return SimpleNamespace(
            succeeded=["sbx-active"],
            failed=[],
            message="deleted",
        )

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.list", mock_list)
    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.bulk_delete", mock_bulk_delete)

    result = runner.invoke(app, ["sandbox", "delete", "--label", "archive", "--all-users", "--yes"])

    output = strip_ansi(result.output)
    assert result.exit_code == 0, f"Failed: {result.output}"

    list_kwargs = captured["list_kwargs"]
    assert list_kwargs["labels"] == ["archive"]
    assert list_kwargs["exclude_terminated"] is True
    assert list_kwargs["user_id"] is None

    bulk_kwargs = captured["bulk_delete_kwargs"]
    assert bulk_kwargs["labels"] == ["archive"]
    assert bulk_kwargs["all_users"] is True
    assert bulk_kwargs["user_id"] is None

    assert "Processed 1 sandbox(es)" in output


def test_sandbox_ssh_no_id_picks_running_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """`prime sandbox ssh` with no ID lists running, non-VM sandboxes to pick from.

    Selecting one feeds its ID into the rest of the flow; we stop the flow right
    after by returning a non-RUNNING sandbox from ``get``.
    """
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setattr("prime_cli.commands.sandbox.shutil.which", lambda _: "/usr/bin/ssh")

    captured: dict[str, Any] = {}

    def mock_list(self: Any, **kwargs: Any) -> Any:
        captured["list_kwargs"] = kwargs
        return SimpleNamespace(
            sandboxes=[
                SimpleNamespace(
                    id="sbx-container",
                    name="builder",
                    docker_image="python:3.12",
                    vm=False,
                    created_at="2026-05-01T00:00:00Z",
                ),
                SimpleNamespace(
                    id="sbx-vm",
                    name="gpu-box",
                    docker_image="cuda:12",
                    vm=True,
                    created_at="2026-05-02T00:00:00Z",
                ),
            ],
            total=2,
            page=1,
            per_page=100,
            has_next=False,
        )

    def mock_get(self: Any, sandbox_id: str) -> Any:
        captured["get_id"] = sandbox_id
        return SimpleNamespace(id=sandbox_id, vm=False, status="STOPPED")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.list", mock_list)
    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.get", mock_get)

    result = runner.invoke(app, ["sandbox", "ssh"], input="1\n")

    output = strip_ansi(result.output)
    # Only RUNNING sandboxes are requested, and the VM one is filtered out of the picker.
    assert captured["list_kwargs"]["status"] == "RUNNING"
    assert "sbx-container" in output
    assert "sbx-vm" not in output
    # The chosen sandbox flows into the rest of the SSH flow.
    assert captured["get_id"] == "sbx-container"
    assert "not running" in output
    assert result.exit_code == 1


def test_sandbox_ssh_no_id_no_running_sandboxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no SSH-able sandboxes, the picker reports it and exits cleanly."""
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setattr("prime_cli.commands.sandbox.shutil.which", lambda _: "/usr/bin/ssh")

    def mock_list(self: Any, **kwargs: Any) -> Any:
        # Only a VM sandbox exists; it is not SSH-able, so the picker is empty.
        return SimpleNamespace(
            sandboxes=[
                SimpleNamespace(
                    id="sbx-vm",
                    name="gpu-box",
                    docker_image="cuda:12",
                    vm=True,
                    created_at="2026-05-02T00:00:00Z",
                ),
            ],
            total=1,
            page=1,
            per_page=100,
            has_next=False,
        )

    def mock_get(self: Any, sandbox_id: str) -> Any:
        raise AssertionError("get should not be called when the picker is empty")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.list", mock_list)
    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.get", mock_get)

    result = runner.invoke(app, ["sandbox", "ssh"])

    output = strip_ansi(result.output)
    assert "No running sandboxes available to SSH into." in output
    assert result.exit_code == 0
