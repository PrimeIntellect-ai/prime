import hashlib
import json
import os
import select
import shlex
import socket
import ssl
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from prime_cli.api.training import HostedTrainingClient
from prime_cli.commands import volumes
from prime_cli.main import app
from prime_cli.volume_gateway import GatewayError, relay
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
        ([], False),
        (["--read"], True),
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
    assert captured[0] == {"read_only": expected, "allow_writable": False, "team_id": "t1"}
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
    client.create_volume_session("data", read_only=True, allow_writable=True, team_id="t1")
    assert client.get_volume_session("data", "s1", team_id="t1").status == "RUNNING"
    client.stop_volume_session("data", "s1", team_id="t1")
    assert requests == [
        ("POST", "/training/volumes/data/sessions", {"readOnly": False, "teamId": "t1"}),
        (
            "POST",
            "/training/volumes/data/sessions",
            {"readOnly": True, "allowWritable": True, "teamId": "t1"},
        ),
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


def test_list_shows_the_cluster_name(monkeypatch):
    from prime_cli.api.training import Volume

    vols = [
        Volume(name="a", status="RUNNING", clusterId="c1", cluster="gpu-east", pvcName="vol-a"),
        Volume(name="b", status="RUNNING", clusterId="c2", pvcName="vol-b"),  # older backend
    ]
    monkeypatch.setattr(
        volumes, "_client", lambda: (SimpleNamespace(list_volumes=lambda **kw: vols), None)
    )
    env = {"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}

    table = CliRunner().invoke(app, ["volumes", "list"], env=env)
    assert table.exit_code == 0, table.output
    assert "Cluster" in table.output and "gpu-east" in table.output

    as_json = CliRunner().invoke(app, ["volumes", "list", "-o", "json"], env=env)
    assert [v["cluster"] for v in json.loads(as_json.output)] == ["gpu-east", None]


def test_create_passes_the_cluster_through(monkeypatch):
    from prime_cli.api.training import Volume

    calls = []

    def create_volume(name, size, team_id=None, cluster=None):
        calls.append((name, size, team_id, cluster))
        return Volume(
            name=name, size=size, status="PENDING", clusterId="c1", cluster="gpu-east", pvcName="v"
        )

    monkeypatch.setattr(
        volumes, "_client", lambda: (SimpleNamespace(create_volume=create_volume), "t1")
    )
    env = {"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}

    result = CliRunner().invoke(
        app, ["volumes", "create", "ckpts", "--cluster", "gpu-east"], env=env
    )
    assert result.exit_code == 0, result.output
    assert "on gpu-east" in result.output
    as_json = CliRunner().invoke(app, ["volumes", "create", "ckpts", "-o", "json"], env=env)
    assert json.loads(as_json.output)["cluster"] == "gpu-east"
    assert calls == [("ckpts", "1Ti", "t1", "gpu-east"), ("ckpts", "1Ti", "t1", None)]


def test_client_sends_cluster_only_when_set():
    posted = []
    body = {"name": "v", "status": "PENDING", "clusterId": "c1", "pvcName": "vol-v"}
    api = SimpleNamespace(post=lambda path, json=None: posted.append(json) or body)
    client = HostedTrainingClient(api)
    client.create_volume("v", "1Ti", cluster="gpu-east")
    client.create_volume("v", "1Ti")
    assert posted[0]["cluster"] == "gpu-east"
    assert "cluster" not in posted[1]


def _setup(monkeypatch, tmp_path, which, run_code=0, stuck=False, find_stdout=""):
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

    def run(cmd, **kw):
        commands.append(cmd)
        # _transfer's "ssh" argvs: the scp fallback's symlink check and get's listing.
        return SimpleNamespace(returncode=run_code, stdout=find_stdout if cmd[0] == "ssh" else "")

    monkeypatch.setattr(volumes.subprocess, "run", run)
    return created, stopped, commands


def _run(*args):
    return CliRunner().invoke(app, ["volumes", *args], env={"PRIME_DISABLE_VERSION_CHECK": "1"})


def test_get_rsync(monkeypatch, tmp_path, _session_dir):
    created, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync", "scp"})
    result = _run("get", "data", "/runs/a", "out")
    assert result.exit_code == 0, result.output
    assert created == [{"read_only": True, "allow_writable": True, "team_id": "t1"}]
    ssh_e = shlex.join(["ssh", "-F", str(_session_dir / "config")])
    # The remote listing comes back empty (not a directory), so one rsync runs.
    assert commands[0][:4] == ["ssh", "-F", str(_session_dir / "config"), "host"]
    assert commands[1:] == [
        [
            "/bin/rsync",
            "-a",
            "-v",
            "--partial-dir=.rsync-partial",
            "-e",
            ssh_e,
            "host:/volume/runs/a",
            "out",
        ]
    ]
    assert "prime volumes stop data s1" in result.output


def test_put_rsync(monkeypatch, tmp_path, _session_dir):
    created, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"})
    result = _run("put", "data", "f.txt", "dir/")
    assert result.exit_code == 0, result.output
    assert created == [{"read_only": False, "allow_writable": False, "team_id": "t1"}]
    assert commands[0][-2:] == ["f.txt", "host:/volume/dir/"]
    assert commands[0][:5] == ["/bin/rsync", "-a", "-v", "--partial-dir=.rsync-partial", "-e"]


def test_get_says_when_it_reuses_a_read_write_session(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, {"ssh", "rsync"})
    reused = SimpleNamespace(
        id="s9",
        status="RUNNING",
        read_only=False,
        error_message=None,
        ssh_connection="u@host.tailnet.ts.net",
    )
    monkeypatch.setattr(
        volumes,
        "_client",
        lambda: (SimpleNamespace(create_volume_session=lambda *a, **kw: reused), "t1"),
    )
    result = _run("get", "data", "x")
    assert result.exit_code == 0, result.output
    assert "Reusing session s9 (read-write)" in result.output


@pytest.mark.parametrize("tools", [{"ssh", "rsync"}, {"ssh", "scp"}])
def test_local_path_starting_with_dash_is_not_an_option(monkeypatch, tmp_path, tools):
    """A local path like "--delete" must never reach rsync/scp as an option."""
    _, _, commands = _setup(monkeypatch, tmp_path, tools)
    assert _run("get", "data", "x", "--", "--delete").exit_code == 0
    assert _run("put", "data", "--", "-f.txt").exit_code == 0
    transfers = [c for c in commands if c[0] != "ssh"]  # scp's symlink check
    assert transfers[0][-1] == "./--delete"
    assert transfers[1][-2] == "./-f.txt"


@pytest.mark.parametrize("tools", [{"ssh", "rsync"}, {"ssh", "scp"}])
def test_local_path_with_colon_is_not_a_remote_operand(monkeypatch, tmp_path, tools):
    """A local name like "checkpoint:final" must stay local for rsync and scp."""
    _, _, commands = _setup(monkeypatch, tmp_path, tools)
    assert _run("put", "data", "checkpoint:final", "/").exit_code == 0
    assert _run("get", "data", "x", "out:1").exit_code == 0
    assert _run("get", "data", "x", "/abs/out:1").exit_code == 0
    transfers = [c for c in commands if c[0] != "ssh"]  # scp's symlink check
    assert transfers[0][-2] == "./checkpoint:final"
    assert transfers[1][-1] == "./out:1"
    assert transfers[2][-1] == "/abs/out:1"


def test_scp_keeps_rsync_trailing_slash_layout(monkeypatch, tmp_path):
    """A source ending in "/" copies its contents with scp too ("dir/.")."""
    _, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "scp"})
    assert _run("put", "data", "dir/", "x/").exit_code == 0
    assert _run("get", "data", "runs/a/", "out").exit_code == 0
    assert _run("put", "data", "dir", "x/").exit_code == 0
    scp = [c for c in commands if c[0] != "ssh"]  # the get's symlink check
    assert scp[0][-2:] == ["dir/.", "host:/volume/x/"]
    assert scp[1][-2:] == ["host:/volume/runs/a/.", "out"]
    assert scp[2][-2:] == ["dir", "host:/volume/x/"]


def test_scp_fallback(monkeypatch, tmp_path, _session_dir):
    _, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "scp"})
    result = _run("put", "data", "f.txt")
    assert result.exit_code == 0, result.output
    assert commands == [["scp", "-r", "-F", str(_session_dir / "config"), "f.txt", "host:/volume/"]]
    assert "rsync not found, using scp (full copy; install rsync for incremental transfers)" in (
        result.output
    )


def test_scp_put_refuses_a_symlink_tree(monkeypatch, tmp_path):
    """scp -r follows links (rsync -a copies them as links), so the scp
    fallback refuses a source tree containing one, before copying."""
    _, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "scp"})
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "f.txt").write_text("x")
    (tree / "link").symlink_to(tmp_path / "key")
    result = _run("put", "data", str(tree), "/")
    assert result.exit_code == 1
    assert "The source contains symbolic links, which scp would follow" in result.output
    assert "Install rsync" in result.output
    assert commands == []


def test_scp_put_copies_a_tree_without_links(monkeypatch, tmp_path, _session_dir):
    _, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "scp"})
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "f.txt").write_text("x")
    result = _run("put", "data", str(tree), "/")
    assert result.exit_code == 0, result.output
    config = str(_session_dir / "config")
    assert commands == [["scp", "-r", "-F", config, str(tree), "host:/volume/"]]


def test_scp_get_refuses_a_remote_symlink(monkeypatch, tmp_path, _session_dir):
    """The scp fallback checks the remote source for links over ssh first
    and refuses (scp would follow them) before any scp call."""
    _, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "scp"}, find_stdout="/volume/x/link\n")
    result = _run("get", "data", "x", "out")
    assert result.exit_code == 1
    assert "The source contains symbolic links, which scp would follow" in result.output
    assert commands == [
        ["ssh", "-F", str(_session_dir / "config"), "host", "find /volume/x -type l | head -n 1"]
    ]


def test_scp_get_copies_when_remote_tree_is_clean(monkeypatch, tmp_path, _session_dir):
    _, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "scp"})
    result = _run("get", "data", "x", "out")
    assert result.exit_code == 0, result.output
    config = str(_session_dir / "config")
    assert commands == [
        ["ssh", "-F", config, "host", "find /volume/x -type l | head -n 1"],
        ["scp", "-r", "-F", config, "host:/volume/x", "out"],
    ]


def test_scp_get_failed_symlink_check_hints_tailnet(monkeypatch, tmp_path):
    """An ssh check that cannot run is reported like a failed transfer."""
    _, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "scp"}, run_code=255)
    result = _run("get", "data", "x", "out")
    assert result.exit_code == 255
    assert "tailnet" in result.output
    assert [c[0] for c in commands] == ["ssh"]


def test_rsync_unaffected_by_the_symlink_refusal(monkeypatch, tmp_path):
    """rsync -a copies links as links: no refusal. (The only ssh call is
    get's remote listing for the parallel split.)"""
    _, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"})
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "link").symlink_to(tmp_path / "key")
    assert _run("put", "data", str(tree), "/").exit_code == 0
    assert _run("get", "data", "x", "out").exit_code == 0
    assert [c[0] for c in commands] == ["/bin/rsync", "ssh", "/bin/rsync"]


def test_no_ssh_tools(monkeypatch, tmp_path):
    created, _, commands = _setup(monkeypatch, tmp_path, set())
    assert _run("get", "data", "x").exit_code == 1
    assert not created and not commands


@pytest.mark.parametrize(
    "bad", ["../x", "a/../b", "a//b", "//a", "my file", "a/*.pt", "x;rm", "$HOME", "it's"]
)
def test_remote_path_rejected(monkeypatch, tmp_path, bad):
    created, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"})
    assert _run("get", "data", bad).exit_code == 2
    assert not created and not commands


def test_remote_path_normalized():
    assert volumes._remote_path("/") == "/volume/"
    assert volumes._remote_path("a/b") == "/volume/a/b"
    assert volumes._remote_path("/a/b/") == "/volume/a/b/"
    assert volumes._remote_path("runs/step_100/model-00001.safetensors") == (
        "/volume/runs/step_100/model-00001.safetensors"
    )


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


SHA = "ab" * 32
GATEWAY = SimpleNamespace(host="gw.example.com", port=443, cert_sha256=SHA.upper())


def _proxy_argv(config_text):
    line = next(ln for ln in config_text.splitlines() if ln.strip().startswith("ProxyCommand "))
    return shlex.split(line.split("ProxyCommand ", 1)[1].replace("%%", "%"))


def _gateway_ssh(monkeypatch, tmp_path, gateway=GATEWAY):
    key = tmp_path / "key"
    key.write_text("test")
    monkeypatch.setattr(volumes.Config, "ssh_key_path", property(lambda self: str(key)))
    session = SimpleNamespace(
        id="s1",
        status="RUNNING",
        read_only=True,
        error_message=None,
        ssh_connection="u@vol-ssh-0123.tailnet.ts.net",
        gateway=gateway,
    )
    client = SimpleNamespace(create_volume_session=lambda *a, **kw: session)
    monkeypatch.setattr(volumes, "_client", lambda: (client, "t1"))
    commands = []
    monkeypatch.setattr(
        volumes.shutil, "which", lambda n: f"/bin/{n}" if n in {"ssh", "rsync"} else None
    )
    monkeypatch.setattr(
        volumes.subprocess,
        "run",
        lambda cmd, **kw: (commands.append(cmd) or SimpleNamespace(returncode=0, stdout="")),
    )
    return commands


def test_gateway_adds_a_proxy_command_with_the_short_name_as_sni(
    monkeypatch, tmp_path, _session_dir
):
    commands = _gateway_ssh(monkeypatch, tmp_path)
    result = _run("ssh", "data")
    assert result.exit_code == 0, result.output
    config = _session_dir / "config"
    text = config.read_text()
    # HostName stays the tailnet name (the known_hosts pin), the SNI is the alias.
    assert "HostName vol-ssh-0123.tailnet.ts.net" in text
    assert _proxy_argv(text) == [
        sys.executable,
        "-m",
        "prime_cli.main",
        "volumes",
        "proxy",
        "--gateway",
        "gw.example.com:443",
        "--cert-sha256",
        SHA,
        "vol-ssh-0123",
    ]
    assert commands == [["ssh", "-F", str(config), "vol-ssh-0123"]]


def test_direct_and_missing_gateway_write_no_proxy_command(monkeypatch, tmp_path, _session_dir):
    _gateway_ssh(monkeypatch, tmp_path)
    direct = _run("ssh", "data", "--direct")
    assert direct.exit_code == 0
    assert "prime volumes get data FILE . --direct" in direct.output
    assert "ProxyCommand" not in (_session_dir / "config").read_text()
    _gateway_ssh(monkeypatch, tmp_path, gateway=None)
    plain = _run("ssh", "data")
    assert plain.exit_code == 0
    assert "--direct" not in plain.output
    assert "ProxyCommand" not in (_session_dir / "config").read_text()


@pytest.mark.parametrize(
    "gateway",
    [
        SimpleNamespace(host="gw\n  ProxyCommand x", port=443, cert_sha256=SHA),
        SimpleNamespace(host="gw.example.com", port=0, cert_sha256=SHA),
        SimpleNamespace(host="gw.example.com", port=443, cert_sha256="abc"),
    ],
)
def test_invalid_gateway_is_ignored(monkeypatch, tmp_path, _session_dir, gateway):
    _gateway_ssh(monkeypatch, tmp_path, gateway=gateway)
    result = _run("ssh", "data")
    assert result.exit_code == 0, result.output
    assert "ProxyCommand" not in (_session_dir / "config").read_text()


def test_proxy_command_escapes_percent_for_ssh(monkeypatch):
    monkeypatch.setattr(sys, "executable", "/opt/100%/py thon")
    argv = shlex.split(volumes._proxy_command(("gw:443", SHA), "vol-ssh-0123").replace("%%", "%"))
    assert argv[0] == "/opt/100%/py thon"


@pytest.mark.parametrize("command", ["get", "put"])
def test_transfers_inherit_the_proxy_through_the_config(
    monkeypatch, tmp_path, _session_dir, command
):
    commands = _gateway_ssh(monkeypatch, tmp_path)
    args = ["get", "data", "x", "out"] if command == "get" else ["put", "data", "f", "/"]
    assert _run(*args).exit_code == 0
    config = _session_dir / "config"
    assert "ProxyCommand" in config.read_text()
    # argv is the same as without a gateway: no -o ProxyCommand on the command line.
    assert commands[-1][:5] == ["/bin/rsync", "-a", "-v", "--partial-dir=.rsync-partial", "-e"]
    assert commands[-1][5] == shlex.join(["ssh", "-F", str(config)])
    assert "ProxyCommand" not in " ".join(commands[-1])


def test_failed_transfer_hint_names_the_gateway_or_the_tailnet(monkeypatch, tmp_path):
    _gateway_ssh(monkeypatch, tmp_path)
    monkeypatch.setattr(
        volumes.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=23, stdout="")
    )
    via = _run("get", "data", "x", "out")
    assert via.exit_code == 23 and "gateway" in via.output and "tailnet" not in via.output
    direct = _run("get", "data", "x", "out", "--direct")
    assert direct.exit_code == 23 and "gateway" not in direct.output and "tailnet" in direct.output


def _self_signed():
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "vol-bastion")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return pem, key_pem, hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()


@pytest.fixture
def tls_echo_gateway(tmp_path):
    """A local TLS server: reads one request, echoes it, closes. Records the SNI."""
    cert_pem, key_pem, fingerprint = _self_signed()
    (tmp_path / "cert.pem").write_bytes(cert_pem)
    (tmp_path / "key.pem").write_bytes(key_pem)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(tmp_path / "cert.pem", tmp_path / "key.pem")
    seen = []
    context.sni_callback = lambda sock, name, ctx: seen.append(name)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(60)

    def serve():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            conn.settimeout(10)
            try:
                with context.wrap_socket(conn, server_side=True) as tls:
                    tls.sendall(tls.recv(4096))
            except (OSError, ssl.SSLError):
                pass

    threading.Thread(target=serve, daemon=True).start()
    yield f"127.0.0.1:{listener.getsockname()[1]}", fingerprint, seen
    listener.close()


def test_proxy_relays_when_the_fingerprint_matches(tls_echo_gateway):
    gateway, fingerprint, seen = tls_echo_gateway
    in_r, in_w = os.pipe()
    out_r, out_w = os.pipe()
    os.write(in_w, b"SSH-2.0-hello")
    os.close(in_w)
    # Upper-case with colons is accepted, like `openssl x509 -fingerprint`.
    pretty = ":".join(fingerprint[i : i + 2].upper() for i in range(0, 64, 2))
    done = threading.Thread(
        target=relay, args=(gateway, pretty, "vol-ssh-0123", in_r, out_w), daemon=True
    )
    done.start()
    assert select.select([out_r], [], [], 10)[0]
    assert os.read(out_r, 64) == b"SSH-2.0-hello"
    done.join(10)
    assert not done.is_alive()
    assert seen == ["vol-ssh-0123"]
    for fd in (in_r, out_r, out_w):
        os.close(fd)


def test_proxy_refuses_a_different_certificate(tls_echo_gateway):
    gateway, fingerprint, _ = tls_echo_gateway
    wrong = "0" * 64
    with pytest.raises(GatewayError, match="unexpected certificate"):
        relay(gateway, wrong, "vol-ssh-0123", 0, 1)


def _start_proxy_process(gateway, fingerprint):
    return subprocess.Popen(
        [
            sys.executable, "-m", "prime_cli.main", "volumes", "proxy",
            "--gateway", gateway, "--cert-sha256", fingerprint, "vol-ssh-0123",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PRIME_DISABLE_VERSION_CHECK": "1"},
    )  # fmt: skip


def test_proxy_process_exits_nonzero_on_a_mismatch(tls_echo_gateway):
    gateway, _, _ = tls_echo_gateway
    process = _start_proxy_process(gateway, "0" * 64)
    out, err = process.communicate(timeout=10)
    assert process.returncode == 1
    assert b"unexpected certificate" in err
    assert out == b""


def test_proxy_process_pipes_stdin_to_stdout(tls_echo_gateway):
    gateway, fingerprint, _ = tls_echo_gateway
    process = _start_proxy_process(gateway, fingerprint)
    out, err = process.communicate(b"ping", timeout=10)
    assert process.returncode == 0, err
    assert out == b"ping"


def test_proxy_relays_full_duplex_while_the_peer_is_not_reading(tmp_path):
    """The gateway sends a payload without reading; the client sends its own.
    Both exceed the socket buffers, so a blocking TLS write in the relay would
    deadlock: neither side would ever read."""
    cert_pem, key_pem, fingerprint = _self_signed()
    (tmp_path / "cert.pem").write_bytes(cert_pem)
    (tmp_path / "key.pem").write_bytes(key_pem)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(tmp_path / "cert.pem", tmp_path / "key.pem")
    size = 8 * 1024 * 1024
    to_client, from_client = os.urandom(size), os.urandom(size)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    listener.listen()
    listener.settimeout(60)
    received = bytearray()
    sent = threading.Event()

    def serve():
        conn, _ = listener.accept()
        conn.settimeout(60)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        with context.wrap_socket(conn, server_side=True) as tls:
            tls.sendall(to_client)  # completes only once the client drains it
            sent.set()
            while chunk := tls.recv(64 * 1024):
                received.extend(chunk)

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    process = subprocess.Popen(
        [
            sys.executable, "-m", "prime_cli.main", "volumes", "proxy",
            "--gateway", f"127.0.0.1:{listener.getsockname()[1]}",
            "--cert-sha256", fingerprint, "vol-ssh-0123",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PRIME_DISABLE_VERSION_CHECK": "1"},
    )  # fmt: skip
    try:
        out, err = process.communicate(from_client, timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        pytest.fail("relay deadlocked")
    finally:
        listener.close()
    server.join(10)
    assert process.returncode == 0, err
    assert out == to_client
    assert bytes(received) == from_client


def test_proxy_reports_an_unreachable_gateway():
    with pytest.raises(GatewayError, match="cannot connect"):
        relay("127.0.0.1:1", SHA, "vol-ssh-0123", 0, 1)


def test_proxy_command_is_hidden():
    result = CliRunner().invoke(app, ["volumes", "--help"])
    assert "proxy" not in result.output


def test_split_balances_bytes_and_drops_empty_buckets():
    buckets = volumes._split([(100, "big"), (60, "a"), (50, "b"), (10, "c")], 2)
    assert sorted(map(sorted, buckets)) == [["a", "b"], ["big", "c"]]
    assert volumes._split([(1, "only")], 4) == [["only"]]


def _fake_popen(monkeypatch, code=0):
    """Records each parallel rsync's argv and its --files-from list."""
    started = []

    class Popen:
        def __init__(self, cmd):
            listfile = next(a for a in cmd if a.startswith("--files-from="))
            started.append((cmd, Path(listfile.split("=", 1)[1]).read_text().splitlines()))

        def wait(self):
            return code

    monkeypatch.setattr(volumes.subprocess, "Popen", Popen)
    return started


def test_put_directory_splits_files_across_parallel_rsyncs(monkeypatch, tmp_path):
    _, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"})
    started = _fake_popen(monkeypatch)
    tree = tmp_path / "ckpt"
    (tree / "empty").mkdir(parents=True)
    for i in range(6):
        (tree / f"shard-{i}").write_bytes(b"x" * (i + 1))
    result = _run("put", "data", str(tree), "runs/")
    assert result.exit_code == 0, result.output
    # One rsync creates the directories first, so the parallel ones never
    # race to mkdir the same path; then the files go over _STREAMS rsyncs.
    # The tree pass is the transfer's own source and destination filtered to
    # directories, never a --files-from list with "." (openrsync recurses
    # into "." and would copy every file single-stream).
    (dirs_pass,) = commands
    assert dirs_pass[-4:] == ["--include=*/", "--exclude=*", str(tree), "host:/volume/runs/"]
    assert not any(a.startswith("--files-from") for a in dirs_pass)
    assert len(started) == volumes._STREAMS
    for cmd, _ in started:
        # "ckpt" (no trailing slash) copies the directory itself: the lists
        # are relative to its parent and every entry starts with "ckpt/".
        assert cmd[-2:] == [f"{tmp_path}/", "host:/volume/runs/"]
        assert cmd[:4] == ["/bin/rsync", "-a", "-v", "--partial-dir=.rsync-partial"]
    listed = [p for _, paths in started for p in paths]
    assert sorted(listed) == [f"ckpt/shard-{i}" for i in range(6)]


def test_get_directory_lists_remote_files_and_splits_them(monkeypatch, tmp_path):
    listing = "\n".join(
        [
            "0|directory|/volume/runs/a/sub",
            "5|regular file|/volume/runs/a/one",
            "7|regular file|/volume/runs/a/sub/two",
            "3|symbolic link|/volume/runs/a/link",
        ]
    )
    _, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"}, find_stdout=listing)
    started = _fake_popen(monkeypatch)
    result = _run("get", "data", "runs/a/", "out")
    assert result.exit_code == 0, result.output
    assert commands[0][-1] == "find /volume/runs/a/ -mindepth 1 -exec stat -c '%s|%F|%n' {} +"
    assert len(started) == 3
    assert {tuple(cmd[-2:]) for cmd, _ in started} == {("host:/volume/runs/a/", "out")}
    listed = sorted(p for _, paths in started for p in paths)
    assert listed == ["link", "one", "sub/two"]
    assert commands[1][-4:] == ["--include=*/", "--exclude=*", "host:/volume/runs/a/", "out"]


def test_parallel_failure_reports_the_exit_code(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, {"ssh", "rsync"})
    _fake_popen(monkeypatch, code=23)
    tree = tmp_path / "t"
    tree.mkdir()
    (tree / "a").write_text("a")
    (tree / "b").write_text("b")
    result = _run("put", "data", f"{tree}/", "/")
    assert result.exit_code == 23
    assert "Transfer failed" in result.output
