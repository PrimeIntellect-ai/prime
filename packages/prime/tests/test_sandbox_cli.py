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


def test_sandbox_delete_by_label_only_deletes_owned_sandboxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setenv("PRIME_USER_ID", "user-1")

    captured: dict[str, Any] = {}

    def mock_list(self: Any, **kwargs: Any) -> Any:
        captured["list_kwargs"] = kwargs
        return SimpleNamespace(
            sandboxes=[
                SimpleNamespace(id="sbx-owned", user_id="user-1"),
                SimpleNamespace(id="sbx-other", user_id="user-2"),
            ],
            has_next=False,
        )

    def mock_bulk_delete(self: Any, sandbox_ids: list[str] | None = None, **_: Any) -> Any:
        captured["sandbox_ids"] = sandbox_ids
        return SimpleNamespace(
            succeeded=sandbox_ids or [],
            failed=[],
            message="deleted",
        )

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.list", mock_list)
    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.bulk_delete", mock_bulk_delete)

    result = runner.invoke(app, ["sandbox", "delete", "--label", "keep", "--yes"])

    output = strip_ansi(result.output)
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert captured["list_kwargs"]["labels"] == ["keep"]
    assert captured["list_kwargs"]["exclude_terminated"] is True
    assert captured["sandbox_ids"] == ["sbx-owned"]
    assert "Successfully deleted 1 sandbox(es):" in output


def test_sandbox_delete_by_label_aligns_with_all_active_only_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setenv("PRIME_USER_ID", "user-1")

    def mock_list(self: Any, **kwargs: Any) -> Any:
        assert kwargs["labels"] == ["archive"]
        assert kwargs["exclude_terminated"] is True
        return SimpleNamespace(
            sandboxes=[SimpleNamespace(id="sbx-active", user_id="user-1")],
            has_next=False,
        )

    def mock_bulk_delete(self: Any, sandbox_ids: list[str] | None = None, **_: Any) -> Any:
        assert sandbox_ids == ["sbx-active"]
        return SimpleNamespace(
            succeeded=["sbx-active"],
            failed=[],
            message="deleted",
        )

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.list", mock_list)
    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.bulk_delete", mock_bulk_delete)

    result = runner.invoke(app, ["sandbox", "delete", "--label", "archive", "--yes"])

    output = strip_ansi(result.output)
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Processed 1 sandbox(es)" in output
