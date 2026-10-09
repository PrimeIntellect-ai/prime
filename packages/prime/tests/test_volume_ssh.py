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
from prime_cli.core import NotFoundError
from prime_cli.main import app
from prime_cli.volume_gateway import GatewayError, relay
from typer.testing import CliRunner


def _no_route(*a, **kw):
    """An older backend: no POST …/transfer route, so get/put use a session."""
    raise NotFoundError("HTTP 404: Not Found")


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


def test_list_says_what_a_volume_status_means(monkeypatch):
    from prime_cli.api.training import Volume

    vols = [
        Volume(name="a", status="PENDING", clusterId="", pvcName=""),
        Volume(name="b", status="RUNNING", clusterId="", pvcName=""),
        Volume(name="c", status="TERMINATING", clusterId="", pvcName=""),
    ]
    monkeypatch.setattr(
        volumes, "_client", lambda: (SimpleNamespace(list_volumes=lambda **kw: vols), None)
    )
    env = {"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}

    table = CliRunner().invoke(app, ["volumes", "list"], env=env)
    assert table.exit_code == 0, table.output
    assert "Cluster" not in table.output, "volumes aren't tied to a cluster"
    for label in ("PENDING", "CREATED", "DELETING"):
        assert label in table.output
    assert "RUNNING" not in table.output

    # JSON keeps the API's values for scripts.
    as_json = CliRunner().invoke(app, ["volumes", "list", "-o", "json"], env=env)
    assert [v["status"] for v in json.loads(as_json.output)] == [
        "PENDING",
        "RUNNING",
        "TERMINATING",
    ]


def test_expand_raises_the_cap(monkeypatch):
    from prime_cli.api.training import Volume

    calls = []

    def expand_volume(name, size, team_id=None):
        calls.append((name, size))
        return Volume(name=name, size=size, status="RUNNING", clusterId="", pvcName="")

    monkeypatch.setattr(
        volumes, "_client", lambda: (SimpleNamespace(expand_volume=expand_volume), None)
    )
    env = {"PRIME_DISABLE_VERSION_CHECK": "1"}
    result = CliRunner().invoke(app, ["volumes", "expand", "ckpts", "--size", "10Ti"], env=env)
    assert result.exit_code == 0, result.output
    assert calls == [("ckpts", "10Ti")] and "10Ti" in result.output
    gone = CliRunner().invoke(app, ["volumes", "resize", "ckpts", "--size", "10Ti"], env=env)
    assert gone.exit_code != 0, "resize is gone in favor of expand"


def test_create_passes_the_cluster_through(monkeypatch):
    from prime_cli.api.training import Volume

    calls = []

    def create_volume(name, size, team_id=None, cluster=None, warm=True):
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
    assert "Volume ckpts (5Ti) is PENDING" in result.output
    assert "--cluster is deprecated" in result.output
    as_json = CliRunner().invoke(app, ["volumes", "create", "ckpts", "-o", "json"], env=env)
    assert json.loads(as_json.output)["cluster"] == "gpu-east"
    # the deprecation warning goes to stderr so --output json stays parseable
    warned = CliRunner().invoke(
        app, ["volumes", "create", "ckpts", "--cluster", "gpu-east", "-o", "json"], env=env
    )
    assert json.loads(warned.stdout)["cluster"] == "gpu-east"
    assert "--cluster is deprecated" not in warned.stdout
    assert calls[:2] == [("ckpts", "5Ti", "t1", "gpu-east"), ("ckpts", "5Ti", "t1", None)]


def test_client_sends_cluster_only_when_set():
    posted = []
    body = {"name": "v", "status": "PENDING", "clusterId": "c1", "pvcName": "vol-v"}
    api = SimpleNamespace(post=lambda path, json=None: posted.append(json) or body)
    client = HostedTrainingClient(api)
    client.create_volume("v", "1Ti", cluster="gpu-east")
    client.create_volume("v", "1Ti")
    assert posted[0]["cluster"] == "gpu-east"
    assert "cluster" not in posted[1]


def _local_sources(monkeypatch, tmp_path):
    """Run in tmp_path, with the local sources the session-route tests put
    (a put checks its source exists before asking for a route)."""
    monkeypatch.chdir(tmp_path)
    for name in ("f", "f.txt", "-f.txt", "checkpoint:final"):
        (tmp_path / name).write_text("x")
    (tmp_path / "dir").mkdir(exist_ok=True)


def _setup(monkeypatch, tmp_path, which, run_code=0, stuck=False, find_stdout=""):
    _local_sources(monkeypatch, tmp_path)
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
        route_volume_transfer=_no_route,
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
        lambda: (
            SimpleNamespace(
                create_volume_session=lambda *a, **kw: reused, route_volume_transfer=_no_route
            ),
            "t1",
        ),
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
    assert _run("get", "data", "x", f"{tmp_path}/out:1").exit_code == 0
    transfers = [c for c in commands if c[0] != "ssh"]  # scp's symlink check
    assert transfers[0][-2] == "./checkpoint:final"
    assert transfers[1][-1] == "./out:1"
    assert transfers[2][-1] == f"{tmp_path}/out:1"


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
    "bad", ["..", "../x", "/..", "a/../../b", "my file", "a/*.pt", "x;rm", "$HOME", "it's"]
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
    # posixpath semantics: "." and empty segments collapse, ".." climbs.
    assert volumes._remote_path(".") == "/volume/"
    assert volumes._remote_path("./") == "/volume/"
    assert volumes._remote_path("./runs/x") == "/volume/runs/x"
    assert volumes._remote_path("a/../runs/x") == "/volume/runs/x"
    assert volumes._remote_path("a//b/") == "/volume/a/b/"
    assert volumes._remote_path("//a/./b") == "/volume/a/b"
    assert volumes._remote_path("a/b/.") == "/volume/a/b/"
    assert volumes._remote_path("a/b/c/..") == "/volume/a/b/"


def test_failed_transfer_exit_code(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, {"ssh", "rsync"}, run_code=23)
    result = _run("get", "data", "x")
    assert result.exit_code == 23
    assert "tailnet" in result.output


def test_wait_failure_stops_session(monkeypatch, tmp_path):
    _, stopped, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"}, stuck=True)
    monkeypatch.setattr(volumes.time, "sleep", lambda s: None)
    monkeypatch.setattr(volumes.time, "monotonic", iter([0, 4000]).__next__)
    result = _run("get", "data", "x")
    assert result.exit_code == 1
    assert stopped == [("data", "s1")] and not commands


SHA = "ab" * 32
GATEWAY = SimpleNamespace(host="gw.example.com", port=443, cert_sha256=SHA.upper())


def _proxy_argv(config_text):
    line = next(ln for ln in config_text.splitlines() if ln.strip().startswith("ProxyCommand "))
    return shlex.split(line.split("ProxyCommand ", 1)[1].replace("%%", "%"))


def _gateway_ssh(monkeypatch, tmp_path, gateway=GATEWAY):
    _local_sources(monkeypatch, tmp_path)
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
    client = SimpleNamespace(
        create_volume_session=lambda *a, **kw: session, route_volume_transfer=_no_route
    )
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
    result = _run("put", "data", str(tree), "ckpts/")
    assert result.exit_code == 0, result.output
    # One rsync creates the directories first, so the parallel ones never
    # race to mkdir the same path; then the files go over _STREAMS rsyncs.
    # The tree pass is the transfer's own source and destination filtered to
    # directories, never a --files-from list with "." (openrsync recurses
    # into "." and would copy every file single-stream).
    (dirs_pass,) = commands
    assert dirs_pass[-4:] == ["--include=*/", "--exclude=*", str(tree), "host:/volume/ckpts/"]
    assert not any(a.startswith("--files-from") for a in dirs_pass)
    assert len(started) == volumes._STREAMS
    for cmd, _ in started:
        # "ckpt" (no trailing slash) copies the directory itself: the lists
        # are relative to its parent and every entry starts with "ckpt/".
        assert cmd[-2:] == [f"{tmp_path}/", "host:/volume/ckpts/"]
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


# --- warm create, phase-driven wait, routed get/put (ENG-6585) -------------


def test_client_sends_warm_and_routes_transfers():
    posted = []
    volume = {"name": "v", "status": "PENDING", "clusterId": "", "pvcName": ""}
    route = {
        "via": "r2",
        "endpoint": "https://acct.r2.cloudflarestorage.com",
        "bucket": "b",
        "prefix": "vol1/",
        "accessKeyId": "AK",
        "secretAccessKey": "SK",
        "sessionToken": "ST",
        "expiresAt": "2026-10-09T12:00:00Z",
    }

    def post(path, json=None):
        posted.append((path, json))
        return route if path.endswith("/transfer") else volume

    client = HostedTrainingClient(SimpleNamespace(post=post))
    client.create_volume("v", "1Ti")
    client.create_volume("v", "1Ti", warm=False)
    got = client.route_volume_transfer("v", "put", "data/x/", team_id="t1")
    client.route_volume_transfer("v", "put", "", ["a", "b"])
    assert posted[0][1]["warm"] is True and posted[1][1]["warm"] is False
    assert posted[2] == (
        "/training/volumes/v/transfer",
        {"mode": "put", "path": "data/x/", "teamId": "t1"},
    )
    assert posted[3][1] == {"mode": "put", "path": "", "entries": ["a", "b"]}
    assert (got.via, got.prefix, got.access_key_id) == ("r2", "vol1/", "AK")
    assert got.session_token == "ST"


@pytest.mark.parametrize("flags,warm", [([], True), (["--no-warm"], False)])
def test_create_warm_by_default(monkeypatch, flags, warm):
    from prime_cli.api.training import Volume

    sent = []

    def create_volume(name, size, team_id=None, cluster=None, warm=True):
        sent.append(warm)
        return Volume(name=name, size=size, status="PENDING", clusterId="", pvcName="")

    monkeypatch.setattr(
        volumes, "_client", lambda: (SimpleNamespace(create_volume=create_volume), None)
    )
    result = _run("create", "ckpts", *flags)
    assert result.exit_code == 0, result.output
    assert sent == [warm]
    assert ("Starting a session in the background" in result.output) is warm
    as_json = _run("create", "ckpts", "-o", "json", *flags)
    assert json.loads(as_json.output)["name"] == "ckpts"


def test_ssh_shows_the_phase_and_how_the_session_ends(monkeypatch, tmp_path):
    def poll(**over):
        fields = dict(
            id="s1",
            status="DEPLOYING",
            read_only=False,
            ssh_connection=None,
            host_public_key=None,
            error_message=None,
            phase="creating",
            progress=None,
        )
        return SimpleNamespace(**{**fields, **over})

    polls = [
        poll(phase="staging", progress="1.2 GiB / 4 GiB, 30%"),
        poll(status="RUNNING", phase="ready", ssh_connection="prime@h.corp.ts.net"),
    ]
    _poll_client(monkeypatch, tmp_path, polls, [])
    first = poll()
    monkeypatch.setattr(
        volumes,
        "_client",
        lambda: (
            SimpleNamespace(
                create_volume_session=lambda *a, **kw: first,
                get_volume_session=lambda *a, **kw: polls.pop(0),
            ),
            None,
        ),
    )
    result = CliRunner().invoke(
        app, ["volumes", "ssh", "data"], env={"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}
    )
    assert result.exit_code == 0, result.output
    assert "Copying data from bucket... 1.2 GiB / 4 GiB, 30%" in result.output
    assert "Changes sync to the volume every minute." in result.output
    assert "stops after 30 minutes idle" in result.output
    assert "prime volumes stop data s1" in result.output


def test_ssh_wait_ends_on_a_failed_phase(monkeypatch, tmp_path):
    session = SimpleNamespace(
        id="s1",
        status="DEPLOYING",
        read_only=True,
        ssh_connection=None,
        host_public_key=None,
        error_message="staging failed",
        phase="failed",
    )
    stopped = []
    _poll_client(monkeypatch, tmp_path, [], stopped)
    monkeypatch.setattr(
        volumes,
        "_client",
        lambda: (
            SimpleNamespace(
                create_volume_session=lambda *a, **kw: session,
                stop_volume_session=lambda name, sid, **kw: stopped.append(sid),
            ),
            None,
        ),
    )
    result = CliRunner().invoke(
        app, ["volumes", "ssh", "data"], env={"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}
    )
    assert result.exit_code == 1
    assert "Session is FAILED: staging failed" in result.output
    assert stopped == ["s1"]


def test_route_via_session_reuses_it(monkeypatch, tmp_path, _session_dir):
    from prime_cli.api.training import VolumeSession, VolumeTransferRoute

    created, stopped, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"})
    live = VolumeSession.model_validate(
        {
            "id": "s7",
            "volumeName": "data",
            "status": "RUNNING",
            "readOnly": False,
            "sshConnection": "u@host.tailnet.ts.net",
        }
    )
    client, _ = volumes._client()
    client.route_volume_transfer = lambda *a, **kw: VolumeTransferRoute(via="session", session=live)
    result = _run("put", "data", "f.txt", "dir/")
    assert result.exit_code == 0, result.output
    assert created == [] and stopped == []
    assert "Reusing session s7 (read-write)" in result.output
    assert commands[0][-2:] == ["f.txt", "host:/volume/dir/"]


class FakeS3:
    """The few boto3 S3 client calls the direct path makes, over a dict."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.metadata = {}
        self.entries = []  # `entries` of each route call

    def upload_file(self, path, bucket, key, ExtraArgs=None, Config=None, Callback=None):
        assert bucket == "b" and Config.max_concurrency == volumes._R2_PART_WORKERS
        self.objects[key] = Path(path).read_bytes()
        self.metadata[key] = ExtraArgs["Metadata"]
        Callback(len(self.objects[key]))

    def download_file(self, bucket, key, path, Config=None, Callback=None):
        Path(path).write_bytes(self.objects[key])
        Callback(len(self.objects[key]))

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))[:MaxKeys]
        return {"Contents": [{"Key": k, "Size": len(self.objects[k])} for k in keys]}

    def get_paginator(self, name):
        return SimpleNamespace(paginate=lambda **kw: [self.list_objects_v2(**kw)])

    def head_object(self, Bucket, Key):
        from botocore.exceptions import ClientError

        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ContentLength": len(self.objects[Key])}


R2_ROUTE = dict(
    via="r2",
    endpoint="https://acct.r2.cloudflarestorage.com",
    bucket="b",
    prefix="vol1/",
    accessKeyId="AK",
    secretAccessKey="SK",
    sessionToken="ST",
    expiresAt="2099-01-01T00:00:00Z",
)


def _direct(monkeypatch, s3):
    """A backend that routes to R2; returns the route calls made."""
    from prime_cli.api.training import VolumeTransferRoute

    routes = []

    def route(name, mode, path, entries=None, team_id=None):
        routes.append((mode, path))
        s3.entries.append(entries)
        return VolumeTransferRoute.model_validate(R2_ROUTE)

    client = SimpleNamespace(
        route_volume_transfer=route,
        create_volume_session=lambda *a, **kw: pytest.fail("no session on the direct path"),
    )
    monkeypatch.setattr(volumes, "_client", lambda: (client, "t1"))
    monkeypatch.setattr(volumes, "_r2_client", lambda refresh, r: s3)
    return routes


def test_r2_client_is_scoped_to_the_route():
    from prime_cli.api.training import VolumeTransferRoute

    route = VolumeTransferRoute.model_validate(R2_ROUTE)
    s3 = volumes._r2_client(lambda: route, route)
    assert s3.meta.endpoint_url == R2_ROUTE["endpoint"]
    assert s3.meta.region_name == "auto"
    assert s3.meta.config.signature_version == "s3v4"
    creds = s3._request_signer._credentials.get_frozen_credentials()
    assert (creds.access_key, creds.secret_key, creds.token) == ("AK", "SK", "ST")


@pytest.mark.parametrize(
    "local,remote,existing,keys",
    [
        ("f.txt", "/", {}, ["f.txt"]),
        ("f.txt", "dir/", {}, ["dir/f.txt"]),
        ("f.txt", "dir", {}, ["dir"]),  # no such directory: rsync names the file "dir"
        ("f.txt", "dir", {"vol1/dir/x": b""}, ["dir/f.txt", "dir/x"]),
        ("tree", "/", {}, ["tree/a", "tree/sub/b"]),
        ("tree", "x", {}, ["x/tree/a", "x/tree/sub/b"]),
        ("tree/", "x/", {}, ["x/a", "x/sub/b"]),
        ("tree/", "/", {}, ["a", "sub/b"]),
        ("tree/.", "x/", {}, ["x/a", "x/sub/b"]),  # rsync: "dir/." is "dir/"
        ("tree/./", "x", {}, ["x/a", "x/sub/b"]),
    ],
)
def test_direct_put_mirrors_rsync_layout(monkeypatch, tmp_path, local, remote, existing, keys):
    monkeypatch.chdir(tmp_path)
    Path("f.txt").write_text("f")
    (tmp_path / "tree" / "sub").mkdir(parents=True)
    Path("tree/a").write_text("a")
    Path("tree/sub/b").write_text("bb")
    Path("tree/link").symlink_to(tmp_path / "f.txt")
    s3 = FakeS3(existing)
    routes = _direct(monkeypatch, s3)
    result = _run("put", "data", local, remote)
    assert result.exit_code == 0, result.output
    assert routes == [("put", volumes._remote_path(remote).removeprefix("/volume/"))]
    assert sorted(k.removeprefix("vol1/") for k in s3.objects) == keys
    if local.startswith("tree"):
        assert "Skipping 1 symbolic links" in result.output


@pytest.mark.parametrize(
    "remote,local,files",
    [
        ("f.txt", "out", ["out"]),  # no such local dir: the file is named "out"
        ("f.txt", "out/", ["out/f.txt"]),
        ("d", "out", ["out/d/a", "out/d/sub/b"]),
        ("d/", "out", ["out/a", "out/sub/b"]),
        ("d/.", "out", ["out/a", "out/sub/b"]),
        ("/", "out", ["out/d/a", "out/d/sub/b", "out/f.txt", "out/runs/r1/m"]),
        ("runs/r1", ".", ["r1/m"]),
    ],
)
def test_direct_get_mirrors_rsync_layout(monkeypatch, tmp_path, remote, local, files):
    monkeypatch.chdir(tmp_path)
    s3 = FakeS3(
        {
            "vol1/f.txt": b"f",
            "vol1/d/a": b"a",
            "vol1/d/sub/b": b"bb",
            "vol1/runs/r1/m": b"m",
            "vol1/runs/.sessions/s1/synced-at": b"t",
            "vol1/d/../escape": b"x",
        }
    )
    routes = _direct(monkeypatch, s3)
    result = _run("get", "data", remote, local)
    assert result.exit_code == 0, result.output
    assert routes == [("get", volumes._remote_path(remote).removeprefix("/volume/"))]
    got = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert got == files
    assert not (tmp_path / "escape").exists()


def test_direct_get_missing_path_fails(monkeypatch, tmp_path):
    _direct(monkeypatch, FakeS3({"vol1/a": b"a"}))
    result = _run("get", "data", "nope", str(tmp_path))
    assert result.exit_code == 1
    assert "No such file or directory on the volume: /nope" in result.output


@pytest.mark.parametrize(
    "local,remote",
    [
        ("f.txt", "runs/"),
        ("f.txt", "/runs/x"),
        ("f.txt", "./runs/x"),
        ("f.txt", "a/../runs/x"),
        ("runs", "/"),
        ("runs", "."),
        ("top/", "/"),
        ("top/.", "/"),
    ],
)
def test_put_never_writes_under_runs(monkeypatch, tmp_path, local, remote):
    monkeypatch.chdir(tmp_path)
    Path("f.txt").write_text("f")
    Path("runs").write_text("r")
    (tmp_path / "top" / "runs").mkdir(parents=True)
    routes = _direct(monkeypatch, FakeS3())
    result = _run("put", "data", local, remote)
    assert result.exit_code == 2
    assert "runs/ holds run outputs" in result.output
    assert routes == []


def test_get_of_runs_with_a_live_session_goes_direct(monkeypatch, tmp_path):
    """The backend answers r2 for a get under runs/ even with a live session
    (sessions never stage runs/), so the route must be told the path."""
    from prime_cli.api.training import VolumeTransferRoute

    monkeypatch.chdir(tmp_path)
    s3 = FakeS3({"vol1/runs/r1/m": b"m"})
    asked = []

    def route(name, mode, path, entries=None, team_id=None):
        asked.append((mode, path))
        live = {
            "via": "session",
            "session": {"id": "s1", "volumeName": "data", "status": "RUNNING", "readOnly": False},
        }
        return VolumeTransferRoute.model_validate(R2_ROUTE if path.startswith("runs/") else live)

    client = SimpleNamespace(
        route_volume_transfer=route,
        create_volume_session=lambda *a, **kw: pytest.fail("no session for runs/"),
    )
    monkeypatch.setattr(volumes, "_client", lambda: (client, "t1"))
    refreshes = []
    monkeypatch.setattr(volumes, "_r2_client", lambda refresh, r: refreshes.append(refresh) or s3)
    result = _run("get", "data", "/runs/r1/", "out")
    assert result.exit_code == 0, result.output
    assert asked == [("get", "runs/r1/")]
    assert (tmp_path / "out" / "m").read_bytes() == b"m"
    # Credential refreshes ask the same question.
    refreshes[0]()
    assert asked[-1] == ("get", "runs/r1/")


@pytest.mark.parametrize("remote", ["./runs/x", "a/../runs/x"])
def test_put_under_runs_refused_on_the_session_route_too(monkeypatch, tmp_path, remote):
    created, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"})
    result = _run("put", "data", "f.txt", remote)
    assert result.exit_code == 2
    assert "runs/ holds run outputs" in result.output
    assert not created and not commands


@pytest.mark.parametrize("args", [("get", "data", "."), ("get", "data", "./")])
def test_direct_get_dot_is_the_root(monkeypatch, tmp_path, args):
    monkeypatch.chdir(tmp_path)
    routes = _direct(monkeypatch, FakeS3({"vol1/f.txt": b"f", "vol1/d/a": b"a"}))
    result = _run(*args, "out")
    assert result.exit_code == 0, result.output
    assert routes == [("get", "")]
    assert (tmp_path / "out" / "f.txt").read_bytes() == b"f"
    assert (tmp_path / "out" / "d" / "a").read_bytes() == b"a"


@pytest.mark.parametrize("remote,key", [(".", "f.txt"), ("./d/", "d/f.txt"), ("d/./", "d/f.txt")])
def test_direct_put_normalizes_dot_segments(monkeypatch, tmp_path, remote, key):
    monkeypatch.chdir(tmp_path)
    Path("f.txt").write_text("f")
    s3 = FakeS3()
    routes = _direct(monkeypatch, s3)
    result = _run("put", "data", "f.txt", remote)
    assert result.exit_code == 0, result.output
    assert routes == [("put", key.removesuffix("f.txt"))]
    assert list(s3.objects) == [f"vol1/{key}"]


@pytest.mark.parametrize("kind", ["file", "dir"])
@pytest.mark.parametrize("route", ["r2", "session"])
def test_put_refuses_a_symlink_source(monkeypatch, tmp_path, kind, route):
    """A top-level link would be followed by the direct upload; refuse it
    before any route call, on both routes."""
    if route == "r2":
        routes = _direct(monkeypatch, FakeS3())
        created = commands = []
    else:
        created, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"})
        routes = []
    target = tmp_path / "target"
    if kind == "dir":
        target.mkdir()
        (target / "a").write_text("a")
    else:
        target.write_text("a")
    link = tmp_path / "link"
    link.symlink_to(target)
    result = _run("put", "data", str(link), "/")
    assert result.exit_code == 1
    assert f"{link} is a symlink; pass the path it points to" in result.output.replace("\n", "")
    assert routes == [] and not created and not commands


@pytest.mark.parametrize(
    "detail",
    [
        "volume 'data' has an active read-write session (s9, alice); "
        "use --read-only, or end that session first",
        "an upload to volume 'data' is in progress until 12:34:56Z; retry after it finishes",
        "your read-write session s9 uses a previous SSH key; end it "
        "(prime volumes stop data s9) first",
    ],
)
@pytest.mark.parametrize("args", [["put", "data", "f.txt", "/"], ["get", "data", "f.txt", "."]])
def test_transfer_conflict_is_reported_not_routed_through_a_session(
    monkeypatch, tmp_path, detail, args
):
    """409 (single writer): the route's detail, exit 1, and no fallback to
    a session (only a 404 falls back)."""
    from prime_cli.core import APIError

    monkeypatch.chdir(tmp_path)
    Path("f.txt").write_text("f")

    def route(*a, **kw):
        raise APIError(f"HTTP 409: {detail}")

    client = SimpleNamespace(
        route_volume_transfer=route,
        create_volume_session=lambda *a, **kw: pytest.fail("no session on a 409"),
    )
    monkeypatch.setattr(volumes, "_client", lambda: (client, "t1"))
    monkeypatch.setattr(
        volumes.subprocess, "run", lambda *a, **kw: pytest.fail("no transfer on a 409")
    )
    result = _run(*args)
    assert result.exit_code == 1
    out = " ".join(result.output.split())
    assert f"Error: {detail}" in out and "HTTP 409" not in out and "Tip:" not in out


@pytest.mark.parametrize("args", [["ssh", "data"], ["put", "data", "f.txt", "/"]])
def test_session_create_conflict_is_reported_once(monkeypatch, tmp_path, args):
    """A 409 creating a read-write session: the detail without the HTTP
    prefix, exit 1, one create call (no retry); only ssh gets the hint."""
    from prime_cli.core import APIError

    monkeypatch.chdir(tmp_path)
    Path("f.txt").write_text("f")
    key = tmp_path / "key"
    key.write_text("test")
    monkeypatch.setattr(volumes.Config, "ssh_key_path", property(lambda self: str(key)))
    monkeypatch.setattr(volumes.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    detail = (
        "volume 'data' has an active read-write session (s9, alice); "
        "use --read-only, or end that session first"
    )
    creates = []

    def create(*a, **kw):
        creates.append(kw)
        raise APIError(f"HTTP 409: {detail}")

    client = SimpleNamespace(route_volume_transfer=_no_route, create_volume_session=create)
    monkeypatch.setattr(volumes, "_client", lambda: (client, None))
    monkeypatch.setattr(volumes.subprocess, "run", lambda *a, **kw: pytest.fail("no ssh on a 409"))
    result = _run(*args)
    assert result.exit_code == 1
    assert len(creates) == 1 and creates[0]["read_only"] is False
    out = " ".join(result.output.split())
    assert f"Error: {detail}" in out and "HTTP 409" not in out
    tip = "Tip: prime volumes ssh data --read-only opens a read-only session"
    assert (tip in out) == (args[0] == "ssh")


class _FailingS3(FakeS3):
    def upload_file(self, *a, **kw):
        from boto3.exceptions import S3UploadFailedError

        raise S3UploadFailedError("Failed to upload f.txt to b/vol1/f.txt: AccessDenied")

    def download_file(self, *a, **kw):
        from boto3.exceptions import S3UploadFailedError

        raise S3UploadFailedError("Failed to download vol1/f.txt: AccessDenied")


@pytest.mark.parametrize("args", [["put", "data", "f.txt", "/"], ["get", "data", "f.txt", "out"]])
def test_direct_transfer_boto3_error_is_reported(monkeypatch, tmp_path, args):
    """boto3 wraps a ClientError from upload_file/download_file in
    S3UploadFailedError: reported as "Transfer failed", exit 1, no traceback."""
    monkeypatch.chdir(tmp_path)
    Path("f.txt").write_text("f")
    _direct(monkeypatch, _FailingS3({"vol1/f.txt": b"f"}))
    result = _run(*args)
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)  # not a traceback
    out = " ".join(result.output.split())
    assert "Transfer failed: Failed to" in out and "AccessDenied" in out


def test_r2_credentials_refresh_near_expiry_only(monkeypatch):
    """Put credentials live 15 minutes: fresh ones are used as-is (botocore's
    default 15-minute window would refresh on every request), and ones near
    expiry are refreshed through a new route call."""
    from datetime import datetime, timedelta, timezone

    from prime_cli.api.training import VolumeTransferRoute

    def route_expiring_in(minutes, key):
        at = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        return VolumeTransferRoute.model_validate(
            {**R2_ROUTE, "accessKeyId": key, "expiresAt": at.isoformat()}
        )

    calls = []

    def refresh():
        calls.append(1)
        return route_expiring_in(15, "AK2")

    s3 = volumes._r2_client(refresh, route_expiring_in(15, "AK1"))
    creds = s3._request_signer._credentials
    for _ in range(3):
        assert creds.get_frozen_credentials().access_key == "AK1"
    assert calls == []

    s3 = volumes._r2_client(refresh, route_expiring_in(1, "AK1"))
    creds = s3._request_signer._credentials
    assert creds.get_frozen_credentials().access_key == "AK2"
    assert creds.get_frozen_credentials().access_key == "AK2"
    assert calls == [1]


def test_refused_credential_refresh_fails_the_transfer_cleanly(monkeypatch, tmp_path):
    """A 409 on a mid-transfer refresh is reported as a failed transfer."""
    from datetime import datetime, timedelta, timezone

    from prime_cli.api.training import VolumeTransferRoute
    from prime_cli.core import APIError

    at = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
    route = VolumeTransferRoute.model_validate({**R2_ROUTE, "expiresAt": at})

    def refresh():
        raise APIError("HTTP 409: volume 'data' has a live read-write SSH session (s9, alice)")

    creds = volumes._r2_client(refresh, route)._request_signer._credentials
    with pytest.raises(RuntimeError, match="^volume 'data' has a live read-write"):
        creds.get_frozen_credentials()


def test_dot_source_copies_contents_on_the_session_route(monkeypatch, tmp_path):
    created, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"})
    (tmp_path / "dir" / "runs").mkdir()
    result = _run("put", "data", "dir/.", "/")
    assert result.exit_code == 2  # dir/. puts dir's runs/ at the root
    assert "runs/ holds run outputs" in result.output
    assert not created and not commands
    assert _run("get", "data", "d/.", "out").exit_code == 0
    assert commands[-1][-2:] == ["host:/volume/d/", "out"]


@pytest.mark.parametrize("route", ["r2", "session"])
@pytest.mark.parametrize("problem", ["missing", "unreadable", "unwritable"])
def test_bad_local_path_fails_before_the_route_call(monkeypatch, tmp_path, route, problem):
    """A put route reserves the volume's upload window, so a local typo
    must fail before it is asked for."""
    if route == "r2":
        routes = _direct(monkeypatch, FakeS3({"vol1/f.txt": b"f"}))
        created = commands = []
    else:
        created, _, commands = _setup(monkeypatch, tmp_path, {"ssh", "rsync"})
        routes = []
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "f").write_text("f")
    if problem == "missing":
        args, error = ["put", "data", str(tmp_path / "nope"), "/"], "No such file or directory"
    elif problem == "unreadable":
        args, error = ["put", "data", str(locked / "f"), "/"], "Permission denied"
        (locked / "f").chmod(0)
    else:
        args, error = ["get", "data", "f.txt", str(locked / "sub" / "out")], "Cannot write to"
        locked.chmod(0o500)
    try:
        result = _run(*args)
    finally:
        (locked / "f").chmod(0o600) if problem == "unreadable" else locked.chmod(0o700)
    assert result.exit_code == 1
    assert error in result.output
    assert routes == [] and not created and not commands


@pytest.mark.parametrize(
    "existing,local,remote,clash",
    [
        ({"vol1/a": b"f"}, "f.txt", "a/x", "/a is a file"),  # put f a; put g a/x
        ({"vol1/a": b"f"}, "f.txt", "a/b/", "/a is a file"),
        ({"vol1/x/tree": b"f"}, "tree", "x/", "/x/tree is a file"),
        ({"vol1/x/tree/sub": b"f"}, "tree", "x/", "/x/tree/sub is a file"),
        ({"vol1/x/a/old": b"o"}, "tree/", "x", "/x/a is a directory"),
        ({"vol1/d/f.txt/old": b"o"}, "f.txt", "d/", "/d/f.txt is a directory"),
    ],
)
def test_direct_put_refuses_a_file_directory_clash(
    monkeypatch, tmp_path, existing, local, remote, clash
):
    monkeypatch.chdir(tmp_path)
    Path("f.txt").write_text("f")
    (tmp_path / "tree" / "sub").mkdir(parents=True)
    Path("tree/a").write_text("a")
    Path("tree/sub/b").write_text("b")
    s3 = FakeS3(existing)
    _direct(monkeypatch, s3)
    result = _run("put", "data", local, remote)
    assert result.exit_code == 1, result.output
    assert clash in result.output.replace("\n", "")
    assert s3.objects == existing  # nothing uploaded


def test_direct_put_into_a_directory_and_marker_objects_are_fine(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("f.txt").write_text("f")
    (tmp_path / "tree" / "sub").mkdir(parents=True)
    Path("tree/sub/b").write_text("b")
    s3 = FakeS3({"vol1/a/old": b"o", "vol1/x/tree/sub/": b"", "vol1/x/tree/sub/b": b"o"})
    _direct(monkeypatch, s3)
    assert _run("put", "data", "f.txt", "a").exit_code == 0  # into a/, like rsync
    result = _run("put", "data", "tree", "x/")  # overwrites x/tree/sub/b
    assert result.exit_code == 0, result.output
    assert s3.objects["vol1/a/f.txt"] == b"f" and s3.objects["vol1/x/tree/sub/b"] == b"b"


def test_direct_get_refuses_a_symlinked_parent_in_the_destination(monkeypatch, tmp_path):
    """out/sub -> elsewhere: the get would write outside out/. Refused before
    anything is written; out/ itself being a link is the user's choice."""
    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    real = tmp_path / "real"
    (real / "d").mkdir(parents=True)
    Path("out").symlink_to(real)  # the destination root may be a link
    s3 = FakeS3({"vol1/d/a": b"a", "vol1/d/sub/b": b"b"})
    _direct(monkeypatch, s3)
    (real / "d" / "sub").symlink_to(elsewhere)
    result = _run("get", "data", "d", "out")
    assert result.exit_code == 1, result.output
    assert "out/d/sub is a symlink inside the destination" in result.output.replace("\n", "")
    assert list(elsewhere.iterdir()) == [] and not (real / "d" / "a").exists()
    # A link at the file path itself is refused too.
    (real / "d" / "sub").unlink()
    (real / "d" / "a").symlink_to(elsewhere / "a")
    result = _run("get", "data", "d", "out")
    assert result.exit_code == 1, result.output
    assert "out/d/a is a symlink" in result.output.replace("\n", "")
    assert list(elsewhere.iterdir()) == []
    (real / "d" / "a").unlink()
    assert _run("get", "data", "d", "out").exit_code == 0
    assert (real / "d" / "sub" / "b").read_bytes() == b"b"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads unreadable directories")
def test_direct_put_fails_on_an_unreadable_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tree" / "locked").mkdir(parents=True)
    Path("tree/a").write_text("a")
    Path("tree/locked/b").write_text("b")
    s3 = FakeS3()
    _direct(monkeypatch, s3)
    (tmp_path / "tree" / "locked").chmod(0)
    try:
        result = _run("put", "data", "tree", "x/")
    finally:
        (tmp_path / "tree" / "locked").chmod(0o700)
    assert result.exit_code == 1, result.output
    out = result.output.replace("\n", "")
    assert "Transfer failed" in out and "tree/locked" in out
    assert s3.objects == {}


def test_direct_put_sets_rclone_md5_and_mtime_metadata(monkeypatch, tmp_path):
    """Multipart objects have no MD5 ETag; rclone reads these instead."""
    import base64

    monkeypatch.chdir(tmp_path)
    Path("f.txt").write_bytes(b"hello")
    os.utime("f.txt", ns=(0, 1_700_000_000_120_000_000))
    Path("g.txt").write_bytes(b"")
    os.utime("g.txt", ns=(0, 1_700_000_000_000_000_000))
    s3 = FakeS3()
    _direct(monkeypatch, s3)
    assert _run("put", "data", "f.txt", "d/").exit_code == 0
    assert _run("put", "data", "g.txt", "d/").exit_code == 0
    md5 = base64.b64encode(hashlib.md5(b"hello").digest()).decode()
    assert s3.metadata["vol1/d/f.txt"] == {"md5chksum": md5, "mtime": "1700000000.12"}
    empty = base64.b64encode(hashlib.md5(b"").digest()).decode()
    assert s3.metadata["vol1/d/g.txt"] == {"md5chksum": empty, "mtime": "1700000000"}


@pytest.mark.parametrize(
    "local,remote,entries",
    [
        ("f.txt", "/", ["f.txt"]),
        ("tree", "/", ["tree"]),
        ("tree/", "/", ["a", "link", "sub"]),
        ("tree/.", ".", ["a", "link", "sub"]),
        ("tree", "x/", None),  # not a root put
    ],
)
def test_root_put_sends_its_top_level_entries(monkeypatch, tmp_path, local, remote, entries):
    monkeypatch.chdir(tmp_path)
    Path("f.txt").write_text("f")
    (tmp_path / "tree" / "sub").mkdir(parents=True)
    Path("tree/a").write_text("a")
    Path("tree/sub/b").write_text("b")
    Path("tree/link").symlink_to(tmp_path / "f.txt")
    s3 = FakeS3()
    _direct(monkeypatch, s3)
    result = _run("put", "data", local, remote)
    assert result.exit_code == 0, result.output
    assert s3.entries == [entries]
    tops = {k.removeprefix("vol1/").split("/", 1)[0] for k in s3.objects}
    assert entries is None or tops <= set(entries)


def test_root_put_of_too_many_names_fails_before_the_route(monkeypatch, tmp_path):
    many = tmp_path / "many"
    many.mkdir()
    for i in range(volumes._ROOT_PUT_MAX_ENTRIES + 1):
        (many / f"f{i}").write_text("x")
    s3 = FakeS3()
    routes = _direct(monkeypatch, s3)
    result = _run("put", "data", f"{many}/.", "/")
    assert result.exit_code == 2
    out = " ".join(result.output.split())
    assert "at most 256 top-level names; this one writes 257" in out
    assert "subdirectory" in out and routes == []
    assert _run("put", "data", str(many), "/").exit_code == 0  # one name: many


def test_refused_root_put_is_reported(monkeypatch, tmp_path):
    from prime_cli.core import APIError

    monkeypatch.chdir(tmp_path)
    Path("f.txt").write_text("f")

    def route(*a, **kw):
        raise APIError("HTTP 400: a root put must list its entries")

    client = SimpleNamespace(route_volume_transfer=route)
    monkeypatch.setattr(volumes, "_client", lambda: (client, "t1"))
    result = _run("put", "data", "f.txt", "/")
    assert result.exit_code == 1
    out = " ".join(result.output.split())
    assert "Error: a root put must list its entries" in out and "HTTP 400" not in out
