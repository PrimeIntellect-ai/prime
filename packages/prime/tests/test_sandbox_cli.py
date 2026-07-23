import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from prime_cli.commands.sandbox import _format_sandbox_expiry
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()


def _fake_sandbox(**overrides: Any) -> SimpleNamespace:
    """A sandbox stand-in with every field the list formatter reads."""
    now = datetime.now(timezone.utc)
    base: dict[str, Any] = dict(
        id="sbx-1",
        name="box",
        docker_image="python:3.12",
        status="RUNNING",
        cpu_cores=1.0,
        memory_gb=2.0,
        gpu_count=0,
        region="us",
        labels=[],
        created_at=now,
        started_at=now - timedelta(minutes=10),
        timeout_minutes=60,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_detailed_sandbox(**overrides: Any) -> SimpleNamespace:
    advanced_configs = SimpleNamespace(
        model_dump=lambda: {
            "vmEgressPolicy": {
                "accepted": True,
                "allowlist": ["example.com", "google.com"],
                "denylist": None,
                "generation": 3,
            },
            "customSetting": True,
        }
    )
    return _fake_sandbox(
        start_command="tail -f /dev/null",
        disk_size_gb=10.0,
        disk_mount_path="/sandbox-workspace",
        vm=True,
        network_allowlist=["example.com", "google.com"],
        network_denylist=None,
        idle_timeout_minutes=None,
        termination_reason=None,
        terminated_at=None,
        exit_code=None,
        environment_vars=None,
        secrets=None,
        advanced_configs=advanced_configs,
        user_id="user-1",
        team_id=None,
        registry_credentials_id=None,
        **overrides,
    )


def _network_status(
    *,
    allowlist: list[str] | None = None,
    denylist: list[str] | None = None,
    applied: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        policy=SimpleNamespace(allowlist=allowlist, denylist=denylist),
        generation=2,
        applied_generation=2 if applied else -1,
        applied=applied,
    )


def _configure_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")


def test_sandbox_network_without_flags_shows_current_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_cli(monkeypatch)
    calls: list[str] = []

    def mock_get_network(self: Any, sandbox_id: str) -> SimpleNamespace:
        calls.append(sandbox_id)
        return _network_status(denylist=[])

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.get_network", mock_get_network)

    result = runner.invoke(app, ["sandbox", "network", "sbx-1"])

    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert calls == ["sbx-1"]
    assert "Network access for sbx-1" in output
    assert "Internet access: unrestricted" in output
    assert "generation" not in output.lower()


def test_sandbox_network_warns_when_rules_are_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_cli(monkeypatch)

    def mock_get_network(self: Any, sandbox_id: str) -> SimpleNamespace:
        return _network_status(denylist=[], applied=False)

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.get_network", mock_get_network)

    result = runner.invoke(app, ["sandbox", "network", "sbx-1"])

    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "not active yet" in output
    assert "Retrying automatically" in output
    assert "generation" not in output.lower()


@pytest.mark.parametrize(
    ("allowlist", "denylist", "expected"),
    [
        (["example.com", "google.com"], None, "Access limited to: example.com, google.com"),
        ([], None, "Internet access: blocked"),
        (None, ["example.com", "google.com"], "Blocked: example.com, google.com"),
        (None, [], "Internet access: unrestricted"),
    ],
)
def test_sandbox_network_describes_effective_policy(
    monkeypatch: pytest.MonkeyPatch,
    allowlist: list[str] | None,
    denylist: list[str] | None,
    expected: str,
) -> None:
    _configure_cli(monkeypatch)

    def mock_get_network(self: Any, sandbox_id: str) -> SimpleNamespace:
        return _network_status(allowlist=allowlist, denylist=denylist)

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.get_network", mock_get_network)

    result = runner.invoke(app, ["sandbox", "network", "sbx-1"])

    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert expected in output
    assert "generation" not in output.lower()


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            ["--allow", "api.example.com, 10.0.0.0/8"],
            {
                "allow": ["api.example.com", "10.0.0.0/8"],
                "deny": None,
            },
        ),
        (
            ["--deny", "ads.example.com,192.0.2.0/24"],
            {
                "allow": None,
                "deny": ["ads.example.com", "192.0.2.0/24"],
            },
        ),
        (
            ["--allow", "*"],
            {
                "allow": ["*"],
                "deny": None,
            },
        ),
        (
            ["--deny", "*"],
            {
                "allow": None,
                "deny": ["*"],
            },
        ),
    ],
)
def test_sandbox_network_flags_replace_rules(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    expected: dict[str, Any],
) -> None:
    _configure_cli(monkeypatch)
    captured: dict[str, Any] = {}

    def mock_set_network(self: Any, sandbox_id: str, **kwargs: Any) -> SimpleNamespace:
        captured["sandbox_id"] = sandbox_id
        captured.update(kwargs)
        return _network_status(denylist=[])

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.set_network", mock_set_network)

    result = runner.invoke(app, ["sandbox", "network", "sbx-1", *arguments])

    assert result.exit_code == 0, result.output
    assert captured == {"sandbox_id": "sbx-1", **expected}


def test_sandbox_network_rejects_multiple_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_cli(monkeypatch)

    result = runner.invoke(
        app,
        [
            "sandbox",
            "network",
            "sbx-1",
            "--allow",
            "api.example.com",
            "--deny",
            "ads.example.com",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 1
    assert "provide at most one of" in output


def test_sandbox_network_replaces_instead_of_accumulating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_cli(monkeypatch)
    calls: list[dict[str, Any]] = []

    def mock_set_network(self: Any, sandbox_id: str, **kwargs: Any) -> SimpleNamespace:
        calls.append({"sandbox_id": sandbox_id, **kwargs})
        return _network_status(allowlist=kwargs["allow"])

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.set_network", mock_set_network)

    first = runner.invoke(
        app,
        ["sandbox", "network", "sbx-1", "--allow", "example.com"],
    )
    second = runner.invoke(
        app,
        ["sandbox", "network", "sbx-1", "--allow", "google.com"],
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "✓ Network access updated" in strip_ansi(first.output)
    assert "Access limited to: example.com" in strip_ansi(first.output)
    assert "generation" not in strip_ansi(first.output).lower()
    assert "Access limited to: google.com" in strip_ansi(second.output)
    assert calls == [
        {"sandbox_id": "sbx-1", "allow": ["example.com"], "deny": None},
        {"sandbox_id": "sbx-1", "allow": ["google.com"], "deny": None},
    ]


def test_sandbox_network_rejects_empty_comma_separated_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_cli(monkeypatch)

    result = runner.invoke(
        app,
        ["sandbox", "network", "sbx-1", "--allow", "example.com,,google.com"],
    )

    assert result.exit_code == 1
    assert "comma-separated list" in strip_ansi(result.output)


def test_sandbox_get_renders_network_as_first_class_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_cli(monkeypatch)
    sandbox = _fake_detailed_sandbox()
    monkeypatch.setattr(
        "prime_cli.commands.sandbox.SandboxClient.get",
        lambda self, sandbox_id: sandbox,
    )

    table_result = runner.invoke(app, ["sandbox", "get", "sbx-1"])

    table_output = strip_ansi(table_result.output)
    assert table_result.exit_code == 0, table_result.output
    assert "Network Access" in table_output
    assert "Limited to: example.com, google.com" in table_output
    assert "Egress Allowlist" not in table_output
    assert "vmEgressPolicy" not in table_output
    assert "generation" not in table_output
    assert "customSetting" in table_output

    json_result = runner.invoke(app, ["sandbox", "get", "sbx-1", "--output", "json"])

    assert json_result.exit_code == 0, json_result.output
    json_output = json.loads(json_result.output)
    assert json_output["network_allowlist"] == ["example.com", "google.com"]
    assert json_output["network_denylist"] is None
    assert json_output["advanced_configs"] == {"customSetting": True}


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
    assert captured["request"].cpu_cores == 1.0
    assert captured["request"].memory_gb == 1.0


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


def test_sandbox_ssh_no_id_pages_through_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """The picker pages past page 1, even when page 1 holds only VMs.

    Guards against reporting "no running sandboxes" when the only SSH-able
    container lives on a later page.
    """
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setattr("prime_cli.commands.sandbox.shutil.which", lambda _: "/usr/bin/ssh")

    captured: dict[str, Any] = {}
    pages = {
        1: SimpleNamespace(
            sandboxes=[
                SimpleNamespace(
                    id="sbx-vm",
                    name="gpu-box",
                    docker_image="cuda:12",
                    vm=True,
                    created_at="2026-05-01T00:00:00Z",
                )
            ],
            total=2,
            page=1,
            per_page=100,
            has_next=True,
        ),
        2: SimpleNamespace(
            sandboxes=[
                SimpleNamespace(
                    id="sbx-container",
                    name="builder",
                    docker_image="python:3.12",
                    vm=False,
                    created_at="2026-05-02T00:00:00Z",
                )
            ],
            total=2,
            page=2,
            per_page=100,
            has_next=False,
        ),
    }

    def mock_list(self: Any, **kwargs: Any) -> Any:
        captured.setdefault("pages_requested", []).append(kwargs["page"])
        return pages[kwargs["page"]]

    def mock_get(self: Any, sandbox_id: str) -> Any:
        captured["get_id"] = sandbox_id
        return SimpleNamespace(id=sandbox_id, vm=False, status="STOPPED")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.list", mock_list)
    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.get", mock_get)

    result = runner.invoke(app, ["sandbox", "ssh"], input="1\n")

    output = strip_ansi(result.output)
    assert captured["pages_requested"] == [1, 2]
    assert "sbx-container" in output
    assert captured["get_id"] == "sbx-container"
    assert result.exit_code == 1


def test_sandbox_ssh_no_id_picker_paginates_display(monkeypatch: pytest.MonkeyPatch) -> None:
    """With >50 SSH-able sandboxes the picker shows 50 per page with next/prev nav."""
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setattr("prime_cli.commands.sandbox.shutil.which", lambda _: "/usr/bin/ssh")

    captured: dict[str, Any] = {}

    def mock_list(self: Any, **kwargs: Any) -> Any:
        # 60 running containers on a single API page; display pages them 50 at a time.
        sandboxes = [
            SimpleNamespace(
                id=f"sbx-{i:03d}",
                name=f"box-{i:03d}",
                docker_image="python:3.12",
                vm=False,
                created_at=f"2026-05-01T00:{i:02d}:00Z",
            )
            for i in range(60)
        ]
        return SimpleNamespace(sandboxes=sandboxes, total=60, page=1, per_page=100, has_next=False)

    def mock_get(self: Any, sandbox_id: str) -> Any:
        captured["get_id"] = sandbox_id
        return SimpleNamespace(id=sandbox_id, vm=False, status="STOPPED")

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.list", mock_list)
    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.get", mock_get)

    # Advance to page 2, then select item 51 (global numbering -> sbx-050).
    result = runner.invoke(app, ["sandbox", "ssh"], input="n\n51\n")

    output = strip_ansi(result.output)
    assert "page 1/2" in output
    assert "page 2/2" in output
    assert captured["get_id"] == "sbx-050"
    assert result.exit_code == 1


def test_format_sandbox_expiry_running_shows_time_left() -> None:
    now = datetime.now(timezone.utc)
    sb = _fake_sandbox(status="RUNNING", started_at=now - timedelta(minutes=10), timeout_minutes=60)
    result = _format_sandbox_expiry(sb)
    # ~50 minutes left; assert the shape, not the exact minute (avoids clock flakiness).
    assert result.endswith("left")
    assert "timeout" not in result


def test_format_sandbox_expiry_not_started_shows_timeout() -> None:
    # No started_at yet -> the clock hasn't begun, so show the configured budget.
    pending = _fake_sandbox(status="PENDING", started_at=None, timeout_minutes=60)
    provisioning = _fake_sandbox(status="PROVISIONING", started_at=None, timeout_minutes=30)
    assert _format_sandbox_expiry(pending) == "60m timeout"
    assert _format_sandbox_expiry(provisioning) == "30m timeout"


def test_format_sandbox_expiry_past_deadline_is_expiring() -> None:
    now = datetime.now(timezone.utc)
    sb = _fake_sandbox(
        status="RUNNING", started_at=now - timedelta(minutes=120), timeout_minutes=60
    )
    assert _format_sandbox_expiry(sb) == "expiring"


@pytest.mark.parametrize("status", ["TERMINATED", "TIMEOUT", "ERROR"])
def test_format_sandbox_expiry_terminal_has_no_time_left(status: str) -> None:
    assert _format_sandbox_expiry(_fake_sandbox(status=status)) == "-"


def test_sandbox_list_table_has_expires_column(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setenv("COLUMNS", "200")  # keep the wide table from wrapping the header

    now = datetime.now(timezone.utc)

    def mock_list(self: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            sandboxes=[
                _fake_sandbox(
                    id="sbx-run",
                    status="RUNNING",
                    started_at=now - timedelta(minutes=10),
                    timeout_minutes=60,
                ),
                _fake_sandbox(
                    id="sbx-pending", status="PENDING", started_at=None, timeout_minutes=45
                ),
            ],
            total=2,
            page=1,
            per_page=50,
            has_next=False,
        )

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.list", mock_list)

    result = runner.invoke(app, ["sandbox", "list"])

    output = strip_ansi(result.output)
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Expires" in output
    assert "left" in output  # running sandbox shows time remaining
    assert "45m timeout" in output  # pending sandbox shows its configured budget


def test_sandbox_list_json_includes_expiry_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")

    now = datetime.now(timezone.utc)

    def mock_list(self: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            sandboxes=[
                _fake_sandbox(
                    id="sbx-run",
                    status="RUNNING",
                    started_at=now - timedelta(minutes=10),
                    timeout_minutes=60,
                ),
                _fake_sandbox(
                    id="sbx-pending", status="PENDING", started_at=None, timeout_minutes=45
                ),
            ],
            total=2,
            page=1,
            per_page=50,
            has_next=False,
        )

    monkeypatch.setattr("prime_cli.commands.sandbox.SandboxClient.list", mock_list)

    result = runner.invoke(app, ["sandbox", "list", "--output", "json"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    import json as _json

    data = _json.loads(result.output)
    by_id = {s["id"]: s for s in data["sandboxes"]}
    # Running sandbox has a concrete deadline; pending one has none yet.
    assert by_id["sbx-run"]["timeout_minutes"] == 60
    assert by_id["sbx-run"]["expires_at"] is not None
    assert by_id["sbx-pending"]["timeout_minutes"] == 45
    assert by_id["sbx-pending"]["expires_at"] is None
