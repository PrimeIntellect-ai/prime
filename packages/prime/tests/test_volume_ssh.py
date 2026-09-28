import shlex
from types import SimpleNamespace

import pytest
from prime_cli.api.training import HostedTrainingClient
from prime_cli.commands import volumes
from prime_cli.main import app
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def _session_dir(tmp_path, monkeypatch):
    """Keep the ssh config and known_hosts out of the real ~/.prime."""
    folder = tmp_path / "volume-ssh"
    folder.mkdir()
    monkeypatch.setattr(volumes, "_session_dir", lambda: folder)
    return folder


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
def test_shell_modes_and_shared_ssh_endpoint(tmp_path, monkeypatch, _session_dir, flags, expected):
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
        app, ["volumes", "ssh", "data", *flags], env={"PRIME_DISABLE_VERSION_CHECK": "1"}
    )
    assert result.exit_code == 0, result.output
    assert captured[0] == {"read_only": expected, "team_id": "t1"}
    config = _session_dir / "config"
    # The CLI's own ssh: only the config file and the short alias.
    assert commands[0] == ["ssh", "-F", str(config), "host"]
    text = config.read_text()
    assert "Host host\n" in text and "HostName host.tailnet.ts.net" in text
    assert "User research" in text and "Port 22" in text
    assert f'IdentityFile "{key}"' in text and "IdentitiesOnly yes" in text
    # The printed examples are short: no inline -o options.
    assert "-o " not in result.output and "sftp works too" in result.output
    rsync_line = next(ln.strip() for ln in result.output.splitlines() if "rsync " in ln)
    rsync_argv = shlex.split(rsync_line)
    # The rsync -e string never contains the host (it would be run remotely).
    assert "host" not in rsync_argv[rsync_argv.index("-e") + 1].split()
    if expected:
        # Read-only sessions print download-direction examples.
        assert rsync_argv[-2:] == ["host:/volume/FILE", "."]
    else:
        assert rsync_argv[-2:] == ["FILE", "host:/volume/"]


def test_shell_rejects_conflicting_flags_without_api_call(tmp_path, monkeypatch):
    key = tmp_path / "key"
    key.write_text("test")
    monkeypatch.setattr(volumes.Config, "ssh_key_path", property(lambda self: str(key)))
    result = CliRunner().invoke(
        app,
        ["volumes", "ssh", "data", "--read", "--write"],
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


def test_shell_pins_session_host_key(monkeypatch, tmp_path, _session_dir):
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
        app, ["volumes", "ssh", "data"], env={"PRIME_DISABLE_VERSION_CHECK": "1"}
    )
    assert result.exit_code == 0, result.output
    # Pinned in the CLI-owned known_hosts; checking is strict, never disabled.
    known_hosts = _session_dir / "known_hosts"
    assert known_hosts.read_text() == "vol-shell-1.corp.ts.net ssh-rsa AAAHOSTKEY\n"
    text = (_session_dir / "config").read_text()
    assert f'UserKnownHostsFile "{known_hosts}"' in text
    assert "StrictHostKeyChecking yes" in text
    assert commands[0] == ["ssh", "-F", str(_session_dir / "config"), "vol-shell-1"]
    # A second session for another host keeps the first one's block and pin,
    # and re-running for the same host replaces (never duplicates) them.
    CliRunner().invoke(app, ["volumes", "ssh", "data"], env={"PRIME_DISABLE_VERSION_CHECK": "1"})
    assert text.count("Host vol-shell-1") == 1
    assert (_session_dir / "config").read_text().count("Host vol-shell-1") == 1
    assert known_hosts.read_text().count("vol-shell-1.corp.ts.net") == 1


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
                    id="s1",
                    status="RUNNING",
                    read_only=True,
                    ssh_connection="ubuntu@h.corp.ts.net",
                    host_public_key=None,
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
        app, ["volumes", "ssh", "data"], env={"PRIME_DISABLE_VERSION_CHECK": "1"}
    )
    assert result.exit_code == 0, result.output
    assert not any("UserKnownHostsFile" in c for c in commands[0])
    assert "Using SSH key" in result.output


@pytest.mark.parametrize(
    "error_message,expected",
    [("pod crashed [x]", "Session is FAILED: pod crashed [x]"), (None, "Session is FAILED.")],
)
def test_shell_reports_terminal_session_error(monkeypatch, tmp_path, error_message, expected):
    key = tmp_path / "key"
    key.write_text("test")
    monkeypatch.setattr(volumes.Config, "ssh_key_path", property(lambda self: str(key)))
    session = SimpleNamespace(
        id="s1",
        status="FAILED",
        read_only=True,
        ssh_connection=None,
        host_public_key=None,
        error_message=error_message,
    )
    monkeypatch.setattr(
        volumes,
        "_client",
        lambda: (SimpleNamespace(create_volume_session=lambda *a, **kw: session), None),
    )
    result = CliRunner().invoke(
        app, ["volumes", "ssh", "data"], env={"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}
    )
    assert result.exit_code == 1
    assert expected in result.output


def _poll_client(monkeypatch, tmp_path, polls, stopped):
    """A fake client: create returns a DEPLOYING session, then each poll
    pops the next item from `polls` (an exception to raise or a session)."""
    key = tmp_path / "key"
    key.write_text("test")
    monkeypatch.setattr(volumes.Config, "ssh_key_path", property(lambda self: str(key)))
    monkeypatch.setattr(volumes.time, "sleep", lambda s: None)

    def get(*a, **kw):
        item = polls.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(
        volumes,
        "_client",
        lambda: (
            SimpleNamespace(
                create_volume_session=lambda *a, **kw: SimpleNamespace(
                    id="s1",
                    status="DEPLOYING",
                    read_only=True,
                    ssh_connection=None,
                    host_public_key=None,
                    error_message=None,
                ),
                get_volume_session=get,
                stop_volume_session=lambda name, sid, **kw: stopped.append((name, sid)),
            ),
            None,
        ),
    )
    monkeypatch.setattr(volumes.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=0))


def test_shell_retries_a_failed_status_poll(monkeypatch, tmp_path):
    """A single failed poll (network blip, 5xx) is retried, not fatal."""
    from prime_cli.core import APIError

    running = SimpleNamespace(
        id="s1",
        status="RUNNING",
        read_only=True,
        ssh_connection="prime@h.corp.ts.net",
        host_public_key=None,
        error_message=None,
    )
    stopped = []
    _poll_client(monkeypatch, tmp_path, [APIError("502 Bad Gateway"), running], stopped)
    result = CliRunner().invoke(
        app, ["volumes", "ssh", "data"], env={"PRIME_DISABLE_VERSION_CHECK": "1"}
    )
    assert result.exit_code == 0, result.output
    assert stopped == []


def test_shell_stops_a_session_it_never_connected_to(monkeypatch, tmp_path):
    """Polls that keep failing end the command, and the session it just
    created is stopped instead of being left behind."""
    from prime_cli.core import APIError

    stopped = []
    _poll_client(
        monkeypatch, tmp_path, [APIError("502") for _ in range(volumes._MAX_POLL_ERRORS)], stopped
    )
    result = CliRunner().invoke(
        app, ["volumes", "ssh", "data"], env={"PRIME_DISABLE_VERSION_CHECK": "1"}
    )
    assert result.exit_code == 1
    assert stopped == [("data", "s1")]
    assert "Stopped session s1" in result.output


@pytest.mark.parametrize("output", ["table", "json"])
def test_list_never_shows_the_namespace(monkeypatch, output):
    from prime_cli.api.training import Volume

    vol = Volume(
        name="ckpts",
        size="10Gi",
        status="RUNNING",
        clusterId="c1",
        namespace="prime-team-secret-ns",  # an old backend still sends it
        pvcName="vol-ckpts",
    )
    monkeypatch.setattr(
        volumes, "_client", lambda: (SimpleNamespace(list_volumes=lambda **kw: [vol]), None)
    )
    result = CliRunner().invoke(
        app,
        ["volumes", "list", "-o", output],
        env={"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    assert "ckpts" in result.output
    assert "prime-team-secret-ns" not in result.output
    assert "amespace" not in result.output


def _setup(monkeypatch, tmp_path, which, run_code=0, stuck=False):
    key = tmp_path / "key"
    key.write_text("test")
    monkeypatch.setattr(volumes.Config, "ssh_key_path", property(lambda self: str(key)))
    created, stopped, commands = [], [], []
    session = SimpleNamespace(
        id="s1",
        status="RUNNING",
        read_only=True,
        error_message=None,
        ssh_connection=None if stuck else "u@host.tailnet.ts.net",
    )

    def create(*a, **kw):
        created.append(kw)
        session.read_only = kw["read_only"]
        return session

    client = SimpleNamespace(
        create_volume_session=create,
        stop_volume_session=lambda *a, **kw: stopped.append(a),
    )
    monkeypatch.setattr(volumes, "_client", lambda: (client, "t1"))
    monkeypatch.setattr(volumes.shutil, "which", lambda n: f"/bin/{n}" if n in which else None)
    monkeypatch.setattr(
        volumes.subprocess,
        "run",
        lambda cmd, **kw: commands.append(cmd) or SimpleNamespace(returncode=run_code),
    )
    return created, stopped, commands


def _run(*args):
    return CliRunner().invoke(app, ["volumes", *args], env={"PRIME_DISABLE_VERSION_CHECK": "1"})


def test_get_rsync(monkeypatch, tmp_path, _session_dir):
    created, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync", "scp"})
    result = _run("get", "data", "/runs/a", "out")
    assert result.exit_code == 0, result.output
    assert created == [{"read_only": True, "team_id": "t1"}]
    ssh_e = shlex.join(["ssh", "-F", str(_session_dir / "config")])
    assert commands == [
        ["/bin/rsync", "-a", "-v", "--partial", "-e", ssh_e, "host:/volume/runs/a", "out"]
    ]
    assert "prime volumes stop data s1" in result.output


def test_put_rsync(monkeypatch, tmp_path, _session_dir):
    created, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"})
    result = _run("put", "data", "f.txt", "dir/")
    assert result.exit_code == 0, result.output
    assert created == [{"read_only": False, "team_id": "t1"}]
    assert commands[0][-2:] == ["f.txt", "host:/volume/dir/"]
    assert commands[0][:5] == ["/bin/rsync", "-a", "-v", "--partial", "-e"]


def test_scp_fallback(monkeypatch, tmp_path, _session_dir):
    _, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "scp"})
    result = _run("put", "data", "f.txt")
    assert result.exit_code == 0, result.output
    assert commands == [["scp", "-r", "-F", str(_session_dir / "config"), "f.txt", "host:/volume/"]]
    assert "rsync not found, using scp (full copy; install rsync for incremental transfers)" in (
        result.output
    )


def test_no_ssh_tools(monkeypatch, tmp_path):
    created, _, commands = _setup(monkeypatch, tmp_path, set())
    assert _run("get", "data", "x").exit_code == 1
    assert not created and not commands


@pytest.mark.parametrize("bad", ["../x", "a/../b", "a//b", "//a"])
def test_remote_path_rejected(monkeypatch, tmp_path, bad):
    created, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"})
    assert _run("get", "data", bad).exit_code == 2
    assert not created and not commands


def test_remote_path_normalized():
    assert volumes._remote_path("/") == "/volume/"
    assert volumes._remote_path("a/b") == "/volume/a/b"
    assert volumes._remote_path("/a/b/") == "/volume/a/b/"


def test_failed_transfer_exit_code(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, {"ssh", "rsync"}, run_code=23)
    result = _run("get", "data", "x")
    assert result.exit_code == 23
    assert "tailnet" in result.output


def test_wait_failure_stops_session(monkeypatch, tmp_path):
    _, stopped, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"}, stuck=True)
    monkeypatch.setattr(volumes.time, "sleep", lambda s: None)
    monkeypatch.setattr(volumes.time, "monotonic", iter([0, 1000]).__next__)
    result = _run("get", "data", "x")
    assert result.exit_code == 1
    assert stopped == [("data", "s1")] and not commands
