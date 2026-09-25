from types import SimpleNamespace

import pytest
from prime_cli.api.training import HostedTrainingClient
from prime_cli.commands import volumes
from prime_cli.main import app
from typer.testing import CliRunner


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], True),
        (
            [
                "--read",
            ],
            True,
        ),
        (["--read-only"], True),
        (["--write"], False),
        (["--read-write"], False),
    ],
)
def test_shell_modes_and_shared_ssh_endpoint(tmp_path, monkeypatch, flags, expected):
    key = tmp_path / "key"
    key.write_text("test")
    monkeypatch.setattr(volumes.Config, "ssh_key_path", property(lambda self: str(key)))
    monkeypatch.setattr(
        volumes,
        "_client",
        lambda: (
            SimpleNamespace(
                create_volume_session=lambda *a, **kw: (
                    captured.append(kw)
                    or SimpleNamespace(
                        id="s1",
                        status="RUNNING",
                        read_only=kw["read_only"],
                        ssh_connection="research@host.tailnet.ts.net -p 22",
                    )
                )
            ),
            "t1",
        ),
    )
    monkeypatch.setattr(
        volumes.subprocess,
        "run",
        lambda cmd, **kw: (commands.append(cmd) or SimpleNamespace(returncode=0)),
    )
    captured, commands = [], []
    result = CliRunner().invoke(
        app, ["volumes", "shell", "data", *flags], env={"PRIME_DISABLE_VERSION_CHECK": "1"}
    )
    assert result.exit_code == 0, result.output
    assert captured[0] == {"read_only": expected, "team_id": "t1"}
    assert commands[0] == ["ssh", "-i", str(key), "-p", "22", "research@host.tailnet.ts.net"]
    assert "sftp" in result.output and "rsync" in result.output


def test_shell_rejects_conflicting_flags_without_api_call(tmp_path, monkeypatch):
    key = tmp_path / "key"
    key.write_text("test")
    monkeypatch.setattr(volumes.Config, "ssh_key_path", property(lambda self: str(key)))
    result = CliRunner().invoke(
        app,
        ["volumes", "shell", "data", "--read", "--write"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == 2


def test_client_session_wire_contract():
    requests = []

    class FakeAPI:
        def post(self, path, json):
            requests.append(("POST", path, json))
            return {"id": "s1", "volumeName": "data", "status": "PENDING", "readOnly": False}

        def get(self, path, params):
            requests.append(("GET", path, params))
            return {"id": "s1", "volumeName": "data", "status": "RUNNING", "readOnly": False}

        def delete(self, path, params):
            requests.append(("DELETE", path, params))

    client = HostedTrainingClient(FakeAPI())
    assert not client.create_volume_session("data", read_only=False, team_id="t1").read_only
    assert client.get_volume_session("data", "s1", team_id="t1").status == "RUNNING"
    client.stop_volume_session("data", "s1", team_id="t1")
    assert requests == [
        ("POST", "/training/volumes/data/sessions", {"readOnly": False, "teamId": "t1"}),
        ("GET", "/training/volumes/data/sessions/s1", {"teamId": "t1"}),
        ("DELETE", "/training/volumes/data/sessions/s1", {"teamId": "t1"}),
    ]


def test_shell_pins_session_host_key(monkeypatch, tmp_path):
    key = tmp_path / "key"
    key.write_text("test")
    monkeypatch.setattr(volumes.Config, "ssh_key_path", property(lambda self: str(key)))

    session = SimpleNamespace(
        id="s1",
        status="RUNNING",
        read_only=True,
        ssh_connection="ubuntu@vol-shell-1.corp.ts.net",
        host_public_key="ssh-rsa AAAHOSTKEY",
    )
    monkeypatch.setattr(
        volumes,
        "_client",
        lambda: (
            SimpleNamespace(
                create_volume_session=lambda *a, **kw: session,
                get_volume_session=lambda *a, **kw: session,
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        volumes.subprocess,
        "run",
        lambda cmd, **kw: (commands.append(cmd) or SimpleNamespace(returncode=0)),
    )
    commands = []
    result = CliRunner().invoke(
        app, ["volumes", "shell", "data"], env={"PRIME_DISABLE_VERSION_CHECK": "1"}
    )
    assert result.exit_code == 0, result.output
    cmd = commands[0]
    assert cmd[0] == "ssh"
    assert any(c == "-o StrictHostKeyChecking=yes" for c in cmd)
    kh = next(
        c.removeprefix("-o UserKnownHostsFile=")
        for c in cmd
        if c.startswith("-o UserKnownHostsFile=")
    )
    assert "vol-shell-1.corp.ts.net ssh-rsa AAAHOSTKEY" in open(kh).read()
    # Printed transfer examples pin the same options.
    assert "-o StrictHostKeyChecking=yes" in result.output


def test_shell_without_host_key_uses_user_known_hosts(monkeypatch, tmp_path):
    key = tmp_path / "key"
    key.write_text("test")
    monkeypatch.setattr(volumes.Config, "ssh_key_path", property(lambda self: str(key)))
    monkeypatch.setattr(
        volumes,
        "_client",
        lambda: (
            SimpleNamespace(
                create_volume_session=lambda *a, **kw: SimpleNamespace(
                    id="s1", status="RUNNING", read_only=True,
                    ssh_connection="ubuntu@h.corp.ts.net", host_public_key=None,
                )
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        volumes.subprocess,
        "run",
        lambda cmd, **kw: (commands.append(cmd) or SimpleNamespace(returncode=0)),
    )
    commands = []
    result = CliRunner().invoke(
        app, ["volumes", "shell", "data"], env={"PRIME_DISABLE_VERSION_CHECK": "1"}
    )
    assert result.exit_code == 0, result.output
    assert not any("UserKnownHostsFile" in c for c in commands[0])
