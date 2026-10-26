import json
import subprocess
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from prime_sandboxes import (
    APIError,
    PaymentRequiredError,
    SandboxNotRunningError,
    Volume,
    VolumeListResponse,
    VolumeMount,
)
from typer.testing import CliRunner

runner = CliRunner()


def _configure_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
    # Empty HOME: Config() loads no profile, so base_url is the production
    # default and the importer image channel resolves to "prod".
    monkeypatch.setenv("HOME", str(tmp_path))


# --- D2: prime sandbox volumes ---


def test_volume_create(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _configure_cli(monkeypatch, tmp_path)
    captured: dict[str, Any] = {}

    def fake_create(self: Any, name: str, *, team_id: Any = None) -> Volume:
        captured["create_args"] = (name, team_id)
        return Volume(id="cvol-1", name="data", created_at=datetime.now(timezone.utc))

    monkeypatch.setattr("prime_sandboxes.volumes.Volumes.create", fake_create)

    result = runner.invoke(app, ["sandbox", "volumes", "create", "data", "--team-id", "team-1"])

    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert "Created volume data" in output
    assert "cvol-1" in output
    # A fresh volume has no occupied bytes; an older server sends no
    # account fields, so no account-storage line either.
    assert "Size: 0 bytes" in output
    assert "Account storage" not in output
    assert captured["create_args"] == ("data", "team-1")


def test_volume_list_table_and_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _configure_cli(monkeypatch, tmp_path)
    created = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)
    response = VolumeListResponse(
        volumes=[
            Volume(
                id="cvol-1",
                name="data",
                occupied_bytes=1536 * 1024**2,
                created_at=created,
            ),
            Volume(
                id="cvol-2",
                name="shared",
                team_id="team-1",
                occupied_bytes=512 * 1024**2,
                created_at=created,
            ),
        ],
        total=2,
        page=1,
        per_page=50,
        has_next=False,
        storage_used_bytes=2 * 1024**3,
        storage_limit_gb=1024,
    )
    captured: dict[str, Any] = {}

    def fake_list(
        self: Any, team_id: Any = None, page: int = 1, per_page: int = 50, **kwargs: Any
    ) -> VolumeListResponse:
        captured["team_id"] = team_id
        captured["page"] = page
        captured["per_page"] = per_page
        return response

    monkeypatch.setattr("prime_sandboxes.volumes.Volumes.list", fake_list)

    table = runner.invoke(app, ["sandbox", "volumes", "list", "--team-id", "team-1"])
    assert table.exit_code == 0, table.output
    output = strip_ansi(table.output)
    assert "data" in output and "shared" in output and "cvol-1" in output
    assert "Size" in output and "1.5 GB" in output and "512.0 MB" in output
    assert "Account storage: 2.0 GB used of 1024 GiB limit" in output
    assert captured["team_id"] == "team-1"
    assert captured == {"team_id": "team-1", "page": 1, "per_page": 50}

    paged = runner.invoke(app, ["sandbox", "volumes", "list", "-p", "3", "-n", "7"])
    assert paged.exit_code == 0, paged.output
    assert captured == {"team_id": None, "page": 3, "per_page": 7}

    as_json = runner.invoke(app, ["sandbox", "volumes", "list", "--json"])
    assert as_json.exit_code == 0, as_json.output
    payload = json.loads(as_json.output)
    assert payload["total"] == 2 and payload["has_next"] is False
    assert [v["name"] for v in payload["volumes"]] == ["data", "shared"]
    assert [v["occupied_bytes"] for v in payload["volumes"]] == [
        1536 * 1024**2,
        512 * 1024**2,
    ]
    assert payload["storage_used_bytes"] == 2 * 1024**3
    assert payload["storage_limit_gb"] == 1024


@pytest.mark.parametrize(
    ("storage_used_bytes", "storage_limit_gb", "expected"),
    [
        # Explicit 0 used with a limit still shows both.
        pytest.param(0, 1024, "Account storage: 0 bytes used of 1024 GiB limit", id="zero"),
        # A missing limit drops only the limit part.
        pytest.param(2048, None, "Account storage: 2.0 KB used", id="no-limit"),
        # An older server omits usage entirely; absence is not a known 0 used.
        pytest.param(None, None, None, id="absent"),
    ],
)
def test_volume_list_empty_page_account_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    storage_used_bytes: Any,
    storage_limit_gb: Any,
    expected: Any,
) -> None:
    _configure_cli(monkeypatch, tmp_path)

    def fake_list(
        self: Any, team_id: Any = None, page: int = 1, per_page: int = 50, **kwargs: Any
    ) -> VolumeListResponse:
        return VolumeListResponse(
            volumes=[],
            total=0,
            page=1,
            per_page=50,
            has_next=False,
            storage_used_bytes=storage_used_bytes,
            storage_limit_gb=storage_limit_gb,
        )

    monkeypatch.setattr("prime_sandboxes.volumes.Volumes.list", fake_list)

    result = runner.invoke(app, ["sandbox", "volumes", "list"])

    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert "No volumes found." in output
    if expected is None:
        assert "Account storage" not in output
    else:
        assert expected in output
        if storage_limit_gb is None:
            assert "limit" not in output


def test_volume_get(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _configure_cli(monkeypatch, tmp_path)
    captured: dict[str, Any] = {}

    def fake_get(self: Any, name: str, team_id: Any = None) -> Volume:
        captured["args"] = (name, team_id)
        return Volume(
            id="cvol-1",
            name="data",
            user_id="user-1",
            head_checkpoint_id="chk-9",
            occupied_bytes=1536 * 1024**2,
            account_storage_used_bytes=2 * 1024**3,
            account_storage_limit_gb=1024,
            created_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr("prime_sandboxes.volumes.Volumes.get", fake_get)

    result = runner.invoke(app, ["sandbox", "volumes", "get", "data", "--team-id", "team-1"])

    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert "data" in output and "cvol-1" in output and "Personal" in output
    assert "Head checkpoint: chk-9" in output
    assert "Size: 1.5 GB" in output
    assert "Account storage: 2.0 GB used of 1024 GiB limit" in output
    assert captured["args"] == ("data", "team-1")


def test_volume_delete_and_in_use_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _configure_cli(monkeypatch, tmp_path)
    deleted: list[Any] = []
    monkeypatch.setattr(
        "prime_sandboxes.volumes.Volumes.delete",
        lambda self, name, team_id=None: deleted.append((name, team_id)),
    )

    cancelled = runner.invoke(app, ["sandbox", "volumes", "delete", "data"], input="n")
    assert cancelled.exit_code == 0, cancelled.output
    assert "Cancelled" in strip_ansi(cancelled.output)
    assert deleted == []

    result = runner.invoke(
        app, ["sandbox", "volumes", "delete", "data", "--team-id", "team-1", "-y"]
    )
    assert result.exit_code == 0, result.output
    assert deleted == [("data", "team-1")]

    def in_use(self: Any, name: str, team_id: Any = None) -> None:
        raise APIError("HTTP 409 Volume is attached to a running sandbox")

    monkeypatch.setattr("prime_sandboxes.volumes.Volumes.delete", in_use)
    failing = runner.invoke(app, ["sandbox", "volumes", "delete", "data", "-y"])
    assert failing.exit_code == 1, failing.output
    assert "attached" in strip_ansi(failing.output)


# --- D2: prime sandbox volumes publish ---


def _fake_publish(captured: dict[str, Any], checkpoint: Any = None) -> None:
    def fake_publish(self: Any, sandbox_id: str, name: str, *, wait: bool = True) -> Any:
        captured["args"] = (sandbox_id, name, wait)
        return checkpoint

    return fake_publish


def _fake_wait(captured: dict[str, Any], checkpoint: Any = None) -> None:
    def fake_wait(self: Any, checkpoint_id: str, timeout_seconds: float = 300) -> Any:
        captured["wait"] = (checkpoint_id, timeout_seconds)
        return checkpoint

    return fake_wait


# --- D2: prime sandbox volumes publish ---


def test_volume_publish_waits_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _configure_cli(monkeypatch, tmp_path)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "prime_sandboxes.volumes.Volumes.publish",
        _fake_publish(captured, SimpleNamespace(id="chk-1", state="PENDING", merge=None)),
    )
    monkeypatch.setattr(
        "prime_sandboxes.sandbox.SandboxClient.wait_for_checkpoint",
        _fake_wait(captured, SimpleNamespace(id="chk-1", state="DURABLE", merge=None)),
    )

    result = runner.invoke(app, ["sandbox", "volumes", "publish", "data", "--sandbox", "sbx-1"])

    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert captured["args"] == ("sbx-1", "data", False)
    assert captured["wait"] == ("chk-1", 300)
    assert "Checkpoint ID: chk-1" in output
    assert "State: DURABLE" in output
    assert "Published volume data" in output
    assert "not yet DURABLE" not in output


def test_volume_publish_no_wait_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _configure_cli(monkeypatch, tmp_path)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "prime_sandboxes.volumes.Volumes.publish",
        _fake_publish(captured, SimpleNamespace(id="chk-1", state="PENDING", merge=None)),
    )

    result = runner.invoke(
        app,
        ["sandbox", "volumes", "publish", "data", "--sandbox", "sbx-1", "--no-wait"],
    )

    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert captured["args"] == ("sbx-1", "data", False)
    assert "wait" not in captured
    assert "Checkpoint ID: chk-1" in output
    assert "State: PENDING" in output
    # No success claim before DURABLE, plus an honest check command.
    assert "Published volume data" not in output
    assert "not yet DURABLE" in output
    assert "prime sandbox checkpoint get chk-1" in " ".join(output.split())


def test_volume_publish_wait_timeout_passthrough(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "prime_sandboxes.volumes.Volumes.publish",
        _fake_publish(captured, SimpleNamespace(id="chk-1", state="PENDING", merge=None)),
    )
    monkeypatch.setattr(
        "prime_sandboxes.sandbox.SandboxClient.wait_for_checkpoint",
        _fake_wait(captured, SimpleNamespace(id="chk-1", state="DURABLE", merge=None)),
    )

    result = runner.invoke(
        app,
        ["sandbox", "volumes", "publish", "data", "--sandbox", "sbx-1", "--wait-timeout", "42"],
    )

    assert result.exit_code == 0, result.output
    assert captured["wait"] == ("chk-1", 42)


def test_volume_publish_wait_timeout_rejects_nonpositive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "prime_sandboxes.volumes.Volumes.publish",
        _fake_publish(captured, SimpleNamespace(id="chk-1", state="PENDING", merge=None)),
    )

    def invalid_wait(self: Any, checkpoint_id: str, timeout_seconds: float = 300) -> Any:
        raise ValueError("timeout_seconds must be finite and greater than zero")

    monkeypatch.setattr("prime_sandboxes.sandbox.SandboxClient.wait_for_checkpoint", invalid_wait)

    result = runner.invoke(
        app,
        ["sandbox", "volumes", "publish", "data", "--sandbox", "sbx-1", "--wait-timeout", "0"],
    )

    assert result.exit_code == 1, result.output
    assert "must be finite and greater than zero" in strip_ansi(result.output)


def test_volume_publish_surfaces_api_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _configure_cli(monkeypatch, tmp_path)
    for message in (
        "HTTP 400: Volume is mounted read-only",
        "HTTP 404: Volume is not attached to this sandbox",
        "HTTP 409: The volume's head moved since attach",
    ):

        def raise_api_error(
            self: Any,
            sandbox_id: str,
            name: str,
            *,
            wait: bool = True,
            message: str = message,
        ) -> Any:
            raise APIError(message)

        monkeypatch.setattr("prime_sandboxes.volumes.Volumes.publish", raise_api_error)
        result = runner.invoke(app, ["sandbox", "volumes", "publish", "data", "--sandbox", "sbx-1"])
        assert result.exit_code == 1, result.output
        assert message in strip_ansi(result.output)


def test_volume_publish_409_conflict(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _configure_cli(monkeypatch, tmp_path)
    captured: dict[str, Any] = {}

    def conflict(self: Any, sandbox_id: str, name: str, *, wait: bool = True) -> Any:
        captured["args"] = (sandbox_id, name, wait)
        raise APIError("HTTP 409: Another sandbox's publish is in flight")

    monkeypatch.setattr("prime_sandboxes.volumes.Volumes.publish", conflict)

    result = runner.invoke(app, ["sandbox", "volumes", "publish", "data", "--sandbox", "sbx-1"])

    assert result.exit_code == 1, result.output
    output = strip_ansi(result.output)
    assert captured["args"] == ("sbx-1", "data", False)
    assert "HTTP 409" in output
    assert "Another sandbox" in output


def test_volume_publish_wait_timeout_reports_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "prime_sandboxes.volumes.Volumes.publish",
        _fake_publish({}, SimpleNamespace(id="chk-1", state="PENDING", merge="pending")),
    )

    def wait_timeout(self: Any, checkpoint_id: str, timeout_seconds: float = 300) -> Any:
        raise TimeoutError(
            "Checkpoint chk-1 did not finish its volume merge within 300s (merge=pending)"
        )

    monkeypatch.setattr("prime_sandboxes.sandbox.SandboxClient.wait_for_checkpoint", wait_timeout)

    result = runner.invoke(app, ["sandbox", "volumes", "publish", "data", "--sandbox", "sbx-1"])

    assert result.exit_code == 1, result.output
    output = " ".join(strip_ansi(result.output).split())
    assert "did not finish its volume merge within 300s" in output
    assert "Stopped waiting; the publication may still complete." in output
    assert "prime sandbox checkpoint get chk-1" in output


def test_volume_publish_wait_failed_exits_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "prime_sandboxes.volumes.Volumes.publish",
        _fake_publish({}, SimpleNamespace(id="chk-1", state="PENDING", merge=None)),
    )

    def wait_failed(self: Any, checkpoint_id: str, timeout_seconds: float = 300) -> Any:
        raise RuntimeError("Checkpoint chk-1's volume merge failed: writer output rejected")

    monkeypatch.setattr("prime_sandboxes.sandbox.SandboxClient.wait_for_checkpoint", wait_failed)

    result = runner.invoke(app, ["sandbox", "volumes", "publish", "data", "--sandbox", "sbx-1"])

    assert result.exit_code == 1, result.output
    output = " ".join(strip_ansi(result.output).split())
    assert "volume merge failed" in output
    assert "prime sandbox checkpoint get chk-1" in output


def test_volume_publish_requires_sandbox_option(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)

    result = runner.invoke(app, ["sandbox", "volumes", "publish", "data"])

    assert result.exit_code == 2, result.output
    assert "--sandbox" in strip_ansi(result.output)


def test_volume_publish_has_no_team_option(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _configure_cli(monkeypatch, tmp_path)

    result = runner.invoke(
        app, ["sandbox", "volumes", "publish", "data", "--sandbox", "sbx-1", "--team-id", "t-1"]
    )

    assert result.exit_code == 2, result.output


# --- D2: prime sandbox create --volume ---


def test_sandbox_create_volume_option(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _configure_cli(monkeypatch, tmp_path)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "prime_cli.commands.sandbox.SandboxClient.create",
        lambda self, request: captured.__setitem__("request", request)
        or SimpleNamespace(id="sbx-1"),
    )

    result = runner.invoke(
        app,
        [
            "sandbox",
            "create",
            "--volume",
            "data:/data",
            "--volume",
            "ro-data:/mnt:ro",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert "Volumes: data:/data, ro-data:/mnt (ro)" in output
    assert captured["request"].volumes == [
        VolumeMount(name="data", mount_path="/data", read_only=False),
        VolumeMount(name="ro-data", mount_path="/mnt", read_only=True),
    ]


def test_sandbox_create_volume_option_rejects_malformed_syntax(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    created: list[Any] = []
    monkeypatch.setattr(
        "prime_cli.commands.sandbox.SandboxClient.create",
        lambda self, request: created.append(request),
    )

    for spec in ("data", "data:", ":/data"):
        result = runner.invoke(app, ["sandbox", "create", "--volume", spec, "--yes"])
        assert result.exit_code == 1, result.output
        assert "expected NAME:/mount/path[:ro]" in strip_ansi(result.output)

    result = runner.invoke(app, ["sandbox", "create", "--volume", "data::ro", "--yes"])
    assert result.exit_code == 1, result.output
    assert "empty mount path" in strip_ansi(result.output)

    assert created == []


def test_sandbox_create_without_volume_keeps_default_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "prime_cli.commands.sandbox.SandboxClient.create",
        lambda self, request: captured.__setitem__("request", request)
        or SimpleNamespace(id="sbx-1"),
    )

    result = runner.invoke(app, ["sandbox", "create", "--yes"])

    assert result.exit_code == 0, result.output
    assert captured["request"].volumes is None
    assert "Volumes:" not in strip_ansi(result.output)


# --- D3: prime sandbox volumes create --from ---


class _ProdConfig:
    """Config stub resolving to the production API base URL."""

    DEFAULT_BASE_URL = "https://api.primeintellect.ai"

    def __init__(self) -> None:
        self.base_url = self.DEFAULT_BASE_URL
        self.team_id = None


class _DevConfig:
    """Import-flow tests run against a non-production API base URL."""

    DEFAULT_BASE_URL = "https://api.primeintellect.ai"

    def __init__(self) -> None:
        self.base_url = "https://api.dev.example"
        self.team_id = None


def _configure_import_mocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, rsync_returncode: int = 0
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "rsync_calls": [],
        "deleted": [],
        "closed_sessions": [],
        "order": [],
        "publish_calls": [],
        "checkpoint_waits": [],
        "api_clients": set(),
    }
    monkeypatch.setattr("prime_cli.commands.sandbox_volumes.Config", _DevConfig)

    def fake_volume_create(self: Any, name: str, *, team_id: Any = None) -> Volume:
        state["order"].append("volume_create")
        state["api_clients"].add(id(self._parent.client))
        state["volume_creates"] = state.get("volume_creates", [])
        state["volume_creates"].append((name, team_id))
        return Volume(id="cvol-1", name=name, created_at=datetime.now(timezone.utc))

    def fake_sandbox_create(self: Any, request: Any) -> Any:
        state["order"].append("sandbox_create")
        state["api_clients"].add(id(self.client))
        state["sandbox_request"] = request
        return SimpleNamespace(id="sbx-import-1")

    def fake_wait(self: Any, sandbox_id: str, **kwargs: Any) -> None:
        state["order"].append("wait_creation")
        state["waited"] = sandbox_id

    def fake_get(self: Any, sandbox_id: str) -> Any:
        return SimpleNamespace(status="RUNNING")

    def fake_ssh_session(self: Any, sandbox_id: str, public_key: str) -> Any:
        assert sandbox_id == "sbx-import-1"
        state["public_key"] = public_key
        return SimpleNamespace(session_id="sess-1", host="gw.example.com", port=2022)

    def fake_close_session(self: Any, sandbox_id: str, session_id: str) -> None:
        state["order"].append("close_session")
        state["closed_sessions"].append((sandbox_id, session_id))

    def fake_rsync(command: Any, **kwargs: Any) -> Any:
        state["order"].append("rsync")
        state["rsync_calls"].append(list(command))
        state["rsync_kwargs"] = kwargs
        return SimpleNamespace(returncode=rsync_returncode)

    def fake_delete(self: Any, sandbox_id: str, **kwargs: Any) -> Any:
        state["order"].append("delete")
        state["deleted"].append(sandbox_id)
        return {}

    def fake_publish(self: Any, sandbox_id: str, name: str, wait: bool = True) -> Any:
        state["order"].append("publish")
        state["api_clients"].add(id(self._parent.client))
        state["publish_calls"].append((sandbox_id, name, wait))
        return SimpleNamespace(id="chk-1", state="PENDING")

    def fake_wait_checkpoint(self: Any, checkpoint_id: str, **kwargs: Any) -> Any:
        state["order"].append("wait_checkpoint")
        state["checkpoint_waits"].append(checkpoint_id)
        return SimpleNamespace(id=checkpoint_id, state="DURABLE")

    monkeypatch.setattr("prime_sandboxes.volumes.Volumes.create", fake_volume_create)
    monkeypatch.setattr("prime_sandboxes.volumes.Volumes.publish", fake_publish)
    monkeypatch.setattr(
        "prime_cli.commands.sandbox_volumes.SandboxClient.create", fake_sandbox_create
    )
    monkeypatch.setattr(
        "prime_cli.commands.sandbox_volumes.SandboxClient.wait_for_creation", fake_wait
    )
    monkeypatch.setattr(
        "prime_cli.commands.sandbox_volumes.SandboxClient.wait_for_checkpoint",
        fake_wait_checkpoint,
    )
    monkeypatch.setattr("prime_cli.commands.sandbox_ssh.SandboxClient.get", fake_get)
    monkeypatch.setattr(
        "prime_cli.commands.sandbox_ssh.SandboxClient.create_ssh_session", fake_ssh_session
    )
    monkeypatch.setattr(
        "prime_cli.commands.sandbox_ssh.SandboxClient.close_ssh_session", fake_close_session
    )
    monkeypatch.setattr("prime_cli.commands.sandbox_ssh.SandboxClient.delete", fake_delete)
    monkeypatch.setattr("prime_cli.commands.sandbox_volumes.shutil.which", lambda name: True)
    monkeypatch.setattr("prime_cli.commands.sandbox_volumes.subprocess.run", fake_rsync)
    return state


def test_volume_import_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(app, ["sandbox", "volumes", "create", "data", "--from", str(data_dir)])

    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert "Created volume data" in output
    # Two generic phase lines in order, no implementation-detail chatter.
    assert (
        output.index("Importing files into volume...")
        < output.index("Publishing volume...")
        < output.index("✓ Published volume data")
    )
    assert "DURABLE" in output
    assert "Waiting for importer sandbox" not in output
    assert "rsync" not in output
    assert "Copied data into volume" not in output
    assert "Publish requested" not in output
    # Success-path rsync progress is suppressed; stderr stays attached.
    assert state["rsync_kwargs"] == {"stdout": subprocess.DEVNULL}

    assert state["volume_creates"] == [("data", None)]
    sandbox_request = state["sandbox_request"]
    assert sandbox_request.name.startswith("volume-import-data-")
    assert sandbox_request.docker_image == "volume-import:dev"
    assert sandbox_request.cpu_cores == 1.0
    assert sandbox_request.memory_gb == 1.0
    assert sandbox_request.disk_size_gb == 5.0
    assert sandbox_request.timeout_minutes == -1
    assert sandbox_request.idle_timeout_minutes == 30
    assert sandbox_request.labels == ["prime:volume-import"]
    assert sandbox_request.volumes == [
        VolumeMount(name="data", mount_path="/volume", read_only=False)
    ]

    assert state["waited"] == "sbx-import-1"
    assert state["closed_sessions"] == [("sbx-import-1", "sess-1")]

    (command,) = state["rsync_calls"]
    assert command[:4] == ["rsync", "-az", "--partial", "-e"]
    assert command[5] == f"{data_dir}/"
    assert command[6] == "sess-1@gw.example.com:/volume/"
    remote_shell = command[4]
    assert remote_shell.startswith("ssh -p 2022 -i ")
    assert "ProxyCommand=" in remote_shell
    assert "prime_cli.ssh_proxy gw.example.com 2022 sess-1" in remote_shell
    assert "-o StrictHostKeyChecking=no" in remote_shell
    assert "-o UserKnownHostsFile=/dev/null" in remote_shell

    assert state["deleted"] == ["sbx-import-1"]
    # Publish and durability checks sit strictly between rsync and cleanup,
    # and the SSH session is closed before the publish is requested.
    assert state["order"] == [
        "volume_create",
        "sandbox_create",
        "wait_creation",
        "rsync",
        "close_session",
        "publish",
        "wait_checkpoint",
        "delete",
    ]
    assert state["publish_calls"] == [("sbx-import-1", "data", False)]
    assert state["checkpoint_waits"] == ["chk-1"]
    # One sandbox client runs the whole flow: one API transport and waiter.
    assert len(state["api_clients"]) == 1


def test_volume_import_uses_prod_channel_on_default_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # HOME is empty, so Config().base_url is the production default.
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("prime_cli.commands.sandbox_volumes.Config", _ProdConfig)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(app, ["sandbox", "volumes", "create", "data", "--from", str(data_dir)])

    assert result.exit_code == 0, result.output
    assert state["sandbox_request"].docker_image == "volume-import:prod"


def test_volume_import_rsync_failure_keeps_volume_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path, rsync_returncode=23)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(app, ["sandbox", "volumes", "create", "data", "--from", str(data_dir)])

    assert result.exit_code == 1, result.output
    output = " ".join(strip_ansi(result.output).split())
    assert "rsync failed" in output
    assert "Nothing was published by this attempt" in output
    assert "keeps its previously published contents" in output
    assert "resumes incrementally" not in output
    # Nothing was published or waited on; the importer sandbox is still
    # cleaned up, the volume record is not deleted.
    assert state["publish_calls"] == []
    assert state["checkpoint_waits"] == []
    assert state["deleted"] == ["sbx-import-1"]
    assert state["closed_sessions"] == [("sbx-import-1", "sess-1")]


def test_volume_import_rejects_missing_rsync_before_any_api_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("prime_cli.commands.sandbox_volumes.shutil.which", lambda name: False)

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(app, ["sandbox", "volumes", "create", "data", "--from", str(data_dir)])

    assert result.exit_code == 1, result.output
    assert "rsync is required" in strip_ansi(result.output)
    # Rejected before any API mutation.
    assert state.get("volume_creates", []) == []
    assert state["rsync_calls"] == []
    assert "sandbox_request" not in state


def test_volume_import_reuses_existing_volume_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # First run: volume created, transfer fails partway.
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path, rsync_returncode=23)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    first = runner.invoke(
        app,
        ["sandbox", "volumes", "create", "data", "--team-id", "team-1", "--from", str(data_dir)],
    )
    assert first.exit_code == 1, first.output
    assert state["volume_creates"] == [("data", "team-1")]

    # Second run: create now conflicts (same scope); the importer attaches at
    # the volume's durable head for a fresh copy.
    def taken(self: Any, name: str, *, team_id: Any = None) -> Volume:
        state["volume_creates"].append((name, team_id))
        raise APIError("HTTP 409: Volume name already taken")

    monkeypatch.setattr("prime_sandboxes.volumes.Volumes.create", taken)

    def succeeding_rsync(command: Any, **kwargs: Any) -> Any:
        state["rsync_calls"].append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_cli.commands.sandbox_volumes.subprocess.run", succeeding_rsync)
    state["rsync_calls"].clear()

    second = runner.invoke(
        app,
        ["sandbox", "volumes", "create", "data", "--team-id", "team-1", "--from", str(data_dir)],
    )

    assert second.exit_code == 0, second.output
    output = " ".join(strip_ansi(second.output).split())
    assert "Volume data already exists in this scope" in output
    assert "copying the local directory into its current published contents" in output
    assert "resumes incrementally" not in output
    assert state["publish_calls"] == [("sbx-import-1", "data", False)]
    assert state["volume_creates"] == [("data", "team-1"), ("data", "team-1")]
    # Same scope carried into the reused importer sandbox.
    assert state["sandbox_request"].team_id == "team-1"
    assert len(state["rsync_calls"]) == 1
    assert state["deleted"] == ["sbx-import-1", "sbx-import-1"]

    # A body merely mentioning "HTTP 409" is a server error, not a conflict.
    def server_error(self: Any, name: str, *, team_id: Any = None) -> Volume:
        raise APIError("HTTP 500: detail body mentioning HTTP 409")

    monkeypatch.setattr("prime_sandboxes.volumes.Volumes.create", server_error)
    third = runner.invoke(
        app,
        ["sandbox", "volumes", "create", "data", "--team-id", "team-1", "--from", str(data_dir)],
    )

    assert third.exit_code == 1, third.output
    assert len(state["rsync_calls"]) == 1


def test_volume_import_wait_failure_cleans_up_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path)

    def fail_wait(self: Any, sandbox_id: str, **kwargs: Any) -> None:
        raise SandboxNotRunningError(sandbox_id, message="never reached RUNNING")

    monkeypatch.setattr(
        "prime_cli.commands.sandbox_volumes.SandboxClient.wait_for_creation", fail_wait
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(app, ["sandbox", "volumes", "create", "data", "--from", str(data_dir)])

    assert result.exit_code == 1, result.output
    assert "never reached RUNNING" in strip_ansi(result.output)
    assert state["deleted"] == ["sbx-import-1"]
    assert state["rsync_calls"] == []


@pytest.mark.parametrize(
    ("raise_in_create", "expected_exit", "expected_line"),
    [
        # Regression: a failure while creating the importer sandbox must
        # propagate, never exit 0 with a silent swallow.
        pytest.param(
            PaymentRequiredError("HTTP 402: Payment required"), 1, "Payment Required", id="api-402"
        ),
        pytest.param(KeyboardInterrupt(), 130, "Interrupted before publishing", id="interrupt"),
    ],
)
def test_volume_import_importer_create_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    raise_in_create: BaseException,
    expected_exit: int,
    expected_line: str,
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path)

    def failing_create(self: Any, request: Any) -> Any:
        raise raise_in_create

    monkeypatch.setattr("prime_cli.commands.sandbox_volumes.SandboxClient.create", failing_create)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(app, ["sandbox", "volumes", "create", "data", "--from", str(data_dir)])

    assert result.exit_code == expected_exit, result.output
    output = " ".join(strip_ansi(result.output).split())
    assert expected_line in output
    assert "Importing files into volume..." in output
    assert "Publishing volume..." not in output
    assert state["rsync_calls"] == []
    assert state["publish_calls"] == []
    assert state["checkpoint_waits"] == []
    assert state["deleted"] == []


def test_volume_import_sandbox_delete_failure_after_durable_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path)

    def failing_delete(self: Any, sandbox_id: str, **kwargs: Any) -> Any:
        state["order"].append("delete")
        state["deleted"].append(sandbox_id)
        raise RuntimeError("connection reset during delete")

    monkeypatch.setattr("prime_cli.commands.sandbox_ssh.SandboxClient.delete", failing_delete)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(app, ["sandbox", "volumes", "create", "data", "--from", str(data_dir)])

    # The publish reached DURABLE; only the cleanup warning is surfaced.
    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert "Published volume data" in output
    assert "DURABLE" in output
    assert "failed to delete importer sandbox sbx-import-1" in output
    assert "prime sandbox delete sbx-import-1" in output
    # Durable data is published, so the warning has no unpublished-data note.
    assert "unpublished import data" not in output


def test_volume_import_interrupt_before_publish_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path)

    def interrupted_run(command: Any, **kwargs: Any) -> Any:
        state["rsync_calls"].append(list(command))
        raise KeyboardInterrupt()

    monkeypatch.setattr("prime_cli.commands.sandbox_volumes.subprocess.run", interrupted_run)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(app, ["sandbox", "volumes", "create", "data", "--from", str(data_dir)])

    assert result.exit_code == 130, result.output
    output = " ".join(strip_ansi(result.output).split())
    assert "Interrupted before publishing" in output
    assert "nothing was published by this attempt" in output
    assert "resumes incrementally" not in output
    assert state["publish_calls"] == []
    assert state["deleted"] == ["sbx-import-1"]
    assert state["closed_sessions"] == [("sbx-import-1", "sess-1")]


def test_volume_import_publish_conflict_retains_importer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path)

    def conflict(self: Any, sandbox_id: str, name: str, wait: bool = True) -> Any:
        state["publish_calls"].append((sandbox_id, name, wait))
        raise APIError("HTTP 409: Another sandbox's publish is in flight")

    monkeypatch.setattr("prime_sandboxes.volumes.Volumes.publish", conflict)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(app, ["sandbox", "volumes", "create", "data", "--from", str(data_dir)])

    assert result.exit_code == 1, result.output
    output = " ".join(strip_ansi(result.output).split())
    assert "HTTP 409" in output
    assert "Another sandbox" in output
    # The importer is kept: no delete call, and recovery output names it.
    assert state["deleted"] == []
    assert "kept" in output
    assert "prime sandbox volumes get data" in output
    assert "may be older or from another publisher" in output
    assert "does not by itself confirm this publish" in output
    assert "prime sandbox checkpoint get" in output
    assert "prime sandbox delete sbx-import-1" in output
    assert "30 minutes without transfer activity" in output


def test_volume_import_retention_hint_carries_team_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path)

    def conflict(self: Any, sandbox_id: str, name: str, wait: bool = True) -> Any:
        state["publish_calls"].append((sandbox_id, name, wait))
        raise APIError("HTTP 409: Another sandbox's publish is in flight")

    monkeypatch.setattr("prime_sandboxes.volumes.Volumes.publish", conflict)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "sandbox",
            "volumes",
            "create",
            "weird name",
            "--team-id",
            "team 1",
            "--from",
            str(data_dir),
        ],
    )

    assert result.exit_code == 1, result.output
    output = " ".join(strip_ansi(result.output).split())
    assert state["deleted"] == []
    # The recovery hint carries the effective team scope and quoted arguments.
    assert "prime sandbox volumes get 'weird name' --team-id 'team 1'" in output
    assert "prime sandbox delete sbx-import-1" in output


def test_volume_import_publish_timeout_retains_importer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path)

    def wait_timeout(self: Any, checkpoint_id: str, **kwargs: Any) -> Any:
        state["checkpoint_waits"].append(checkpoint_id)
        raise TimeoutError(f"Checkpoint {checkpoint_id} did not become DURABLE within 300s")

    monkeypatch.setattr(
        "prime_cli.commands.sandbox_volumes.SandboxClient.wait_for_checkpoint", wait_timeout
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(app, ["sandbox", "volumes", "create", "data", "--from", str(data_dir)])

    assert result.exit_code == 1, result.output
    output = " ".join(strip_ansi(result.output).split())
    assert state["publish_calls"] == [("sbx-import-1", "data", False)]
    assert "did not become DURABLE within 300s" in output
    assert "The publish may still complete" in output
    # Unknown publish state: importer kept, both ids disclosed, no
    # head-unchanged claim.
    assert state["deleted"] == []
    assert "kept" in output
    assert "prime sandbox checkpoint get chk-1" in output
    assert "30 minutes without transfer activity" in output
    assert "unchanged" not in output


def test_volume_import_publish_failed_retains_importer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path)

    def wait_failed(self: Any, checkpoint_id: str, **kwargs: Any) -> Any:
        state["checkpoint_waits"].append(checkpoint_id)
        raise RuntimeError(
            f"Checkpoint {checkpoint_id} cannot become durable (state=FAILED): no error details"
        )

    monkeypatch.setattr(
        "prime_cli.commands.sandbox_volumes.SandboxClient.wait_for_checkpoint", wait_failed
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(app, ["sandbox", "volumes", "create", "data", "--from", str(data_dir)])

    assert result.exit_code == 1, result.output
    output = " ".join(strip_ansi(result.output).split())
    assert "state=FAILED" in output
    assert "The publish failed" in output
    assert state["deleted"] == []
    assert "prime sandbox checkpoint get chk-1" in output
    assert "prime sandbox delete sbx-import-1" in output


def test_volume_import_interrupt_during_publish_retains_importer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    state = _configure_import_mocks(monkeypatch, tmp_path)

    def wait_interrupt(self: Any, checkpoint_id: str, **kwargs: Any) -> Any:
        state["checkpoint_waits"].append(checkpoint_id)
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        "prime_cli.commands.sandbox_volumes.SandboxClient.wait_for_checkpoint", wait_interrupt
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(app, ["sandbox", "volumes", "create", "data", "--from", str(data_dir)])

    assert result.exit_code == 130, result.output
    output = " ".join(strip_ansi(result.output).split())
    assert "Interrupted while publishing" in output
    assert "unknown" in output
    assert state["deleted"] == []
    assert "prime sandbox checkpoint get chk-1" in output
    assert "prime sandbox delete sbx-import-1" in output
    assert "30 minutes without transfer activity" in output


def test_volume_import_rejects_missing_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    created: list[Any] = []
    monkeypatch.setattr(
        "prime_sandboxes.volumes.Volumes.create",
        lambda self, request: created.append(request),
    )

    result = runner.invoke(
        app, ["sandbox", "volumes", "create", "data", "--from", str(tmp_path / "missing")]
    )

    assert result.exit_code == 1, result.output
    assert "not a local directory" in strip_ansi(result.output)
    assert created == []


# --- ssh_connect regression: transport refactor ---


def test_ssh_connect_builds_command_and_closes_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    closed: list[Any] = []
    commands: list[Any] = []

    monkeypatch.setattr(
        "prime_cli.commands.sandbox.SandboxClient.get",
        lambda self, sandbox_id: SimpleNamespace(status="RUNNING"),
    )
    monkeypatch.setattr(
        "prime_cli.commands.sandbox.SandboxClient.create_ssh_session",
        lambda self, sandbox_id, public_key: SimpleNamespace(
            session_id="sess-1", host="gw.example.com", port=2022
        ),
    )
    monkeypatch.setattr(
        "prime_cli.commands.sandbox.SandboxClient.close_ssh_session",
        lambda self, sandbox_id, session_id: closed.append((sandbox_id, session_id)),
    )
    monkeypatch.setattr(
        "prime_cli.commands.sandbox.subprocess.run",
        lambda command, **kwargs: commands.append(command) or SimpleNamespace(returncode=7),
    )

    result = runner.invoke(app, ["sandbox", "ssh", "sbx-1", "--", "-v"])

    assert result.exit_code == 7, result.output
    (command,) = commands
    assert command[0] == "ssh"
    assert command[1] == "sess-1@gw.example.com"
    assert "-p" in command and "2022" in command
    assert command[-1] == "-v"
    assert closed == [("sbx-1", "sess-1")]


def test_ssh_connect_rejects_not_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _configure_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "prime_cli.commands.sandbox.SandboxClient.get",
        lambda self, sandbox_id: SimpleNamespace(status="PENDING"),
    )

    result = runner.invoke(app, ["sandbox", "ssh", "sbx-1"])

    assert result.exit_code == 1, result.output
    assert "not running" in strip_ansi(result.output)
