"""Live reads (ENG-7113): while a run or read-write session holds a volume,
ls/du/cat/get go through a read-only SSH view of it instead of storage.

The remote commands run for real in the session image's base (Alpine, so
BusyBox find/stat/cat) over a directory mounted at /volume, in place of
ssh; those tests need Docker and skip without it. The parsing and routing
tests are hermetic.
"""

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from prime_cli.api.training import HostedTrainingClient, VolumeSession, VolumeTransferRoute
from prime_cli.commands import volumes
from prime_cli.main import app
from test_volume_objects import storage  # noqa: F401 (fixture)
from typer.testing import CliRunner

ENV = {"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}
# The volume-ssh image's base (model-factory-iac volume-ssh/Dockerfile).
IMAGE = "alpine:3.22"
REAL_RUN, REAL_POPEN = subprocess.run, subprocess.Popen


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    return REAL_RUN(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0 or (
        REAL_RUN(["docker", "pull", "-q", IMAGE], capture_output=True).returncode == 0
    )


docker = pytest.mark.skipif(not _docker_ok(), reason="needs Docker")


def _run(*args):
    return CliRunner().invoke(app, ["volumes", *args], env=ENV)


def _live_session(**over):
    return VolumeSession.model_validate(
        {
            "id": "s1",
            "volumeName": "data",
            "status": "RUNNING",
            "readOnly": True,
            "sshConnection": "prime@vol-ssh-x.tailnet.ts.net",
            "liveViewOf": "run1",
            **over,
        }
    )


@pytest.fixture
def live(monkeypatch, tmp_path):
    """A backend that routes reads to a live view, and `ssh` replaced by the
    same command in a container mounting `tree` at /volume read-only."""
    tree = tmp_path / "volume"
    tree.mkdir()
    key = tmp_path / "key"
    key.write_text("k")
    folder = tmp_path / "volume-ssh"
    folder.mkdir()
    monkeypatch.setattr(volumes, "_session_dir", lambda: folder)
    monkeypatch.setattr(volumes.Config, "ssh_key_path", property(lambda self: str(key)))
    routes = []

    def route(name, mode, path, entries=None, team_id=None, lease_id=None, live=False):
        routes.append((mode, path, live))
        return VolumeTransferRoute(via="session", session=_live_session())

    client = SimpleNamespace(
        route_volume_transfer=route,
        list_volumes=lambda **kw: [SimpleNamespace(name="data", size="1Ti", status="RUNNING")],
        create_volume_session=lambda *a, **kw: pytest.fail("the route picks the session"),
    )
    monkeypatch.setattr(volumes, "_client", lambda: (client, "t1"))
    remote = []

    def in_container(argv):
        if argv[0] != "ssh":
            return argv  # subprocess.run's own Popen, or docker itself
        assert argv[1] == "-F" and argv[3] == "vol-ssh-x"
        remote.append(argv[4])
        mount = ["-v", f"{tree}:/volume:ro"]
        return ["docker", "run", "--rm", *mount, IMAGE, "sh", "-c", argv[4]]

    monkeypatch.setattr(
        volumes.subprocess, "run", lambda argv, **kw: REAL_RUN(in_container(argv), **kw)
    )
    monkeypatch.setattr(
        volumes.subprocess, "Popen", lambda argv, **kw: REAL_POPEN(in_container(argv), **kw)
    )
    return SimpleNamespace(tree=tree, routes=routes, remote=remote)


def _files(tree, files: dict):
    for rel, data in files.items():
        path = tree / rel
        if rel.endswith("/"):
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


CHECKPOINT = {
    "runs/run1/checkpoints/step_10/model.safetensors": b"w" * 1000,
    "runs/run1/checkpoints/step_10/STABLE": b"",
    "runs/run1/checkpoints/step_20/": b"",
    "runs/.sessions/s1/state": b"internal",
    "datasets/a.jsonl": b'{"x": 1}\n',
}


@docker
def test_ls_lists_the_live_volume_in_the_storage_shape(live):
    _files(live.tree, CHECKPOINT)
    result = _run("ls", "data", "runs/run1/checkpoints/", "-o", "json")
    assert result.exit_code == 0, result.output
    entries = {e["path"]: e for e in json.loads(result.stdout)}
    assert set(entries) == {
        "runs/run1/checkpoints/step_10/",
        "runs/run1/checkpoints/step_20/",
    }
    assert entries["runs/run1/checkpoints/step_10/"]["type"] == "directory"
    assert entries["runs/run1/checkpoints/step_10/"]["size"] is None
    # Notes go to stderr: stdout is pure JSON.
    assert "Live read-only view of run1" in result.stderr
    assert live.routes == [("get", "", True)]

    recursive = _run("ls", "data", "runs/", "-R", "-o", "json")
    paths = {e["path"]: e for e in json.loads(recursive.stdout)}
    assert paths["runs/run1/checkpoints/step_10/model.safetensors"]["size"] == 1000
    assert paths["runs/run1/checkpoints/step_10/model.safetensors"]["modified"]
    assert not [p for p in paths if p.startswith("runs/.sessions")]

    one = _run("ls", "data", "datasets/a.jsonl", "-l")
    assert one.exit_code == 0, one.output
    assert "datasets/a.jsonl" in one.stdout and "9" in one.stdout

    plain = _run("ls", "data")
    assert plain.stdout.split() == ["datasets/", "runs/"]


@docker
def test_live_reads_report_a_missing_path(live):
    _files(live.tree, CHECKPOINT)
    for args in (["ls", "data", "nope"], ["du", "data", "nope"], ["cat", "data", "nope"]):
        result = _run(*args)
        assert result.exit_code == 1, (args, result.output)
        assert "No such file" in result.stderr and result.stdout == ""
    # A file named as a directory isn't one.
    assert _run("ls", "data", "datasets/a.jsonl/").exit_code == 1


@docker
def test_du_counts_live_files(live):
    _files(live.tree, CHECKPOINT)
    result = _run("du", "data", "runs/run1", "-o", "json")
    assert result.exit_code == 0, result.output
    usage = json.loads(result.stdout)
    assert (usage["bytes"], usage["objects"]) == (1000, 2)
    # The whole volume, without the sessions' internal state.
    assert (usage["volumeBytes"], usage["volumeObjects"]) == (1009, 3)
    assert usage["capacity"] == "1Ti"


@docker
def test_cat_streams_the_live_file(live):
    payload = bytes(range(256)) * 600
    _files(live.tree, {"runs/run1/log.bin": payload})
    result = _run("cat", "data", "runs/run1/log.bin")
    assert result.exit_code == 0, result.output
    assert result.stdout_bytes == payload
    assert _run("cat", "data", "runs/run1/").exit_code == 2  # a directory operand


def test_live_listing_parses_nul_separated_records(monkeypatch):
    """Hermetic: the remote command's shape and what its records become —
    including a name with a newline and one with a |."""
    seen = []

    def run(command, **kw):
        seen.append(command)
        out = (
            "4096|directory|1700000000\n/volume/runs/run1/ckpt\0"
            "12|regular file|1700000001\n/volume/runs/run1/a|b.txt\0"
            "7|regular file|1700000003\n/volume/runs/run1/first\nsecond.txt\0"
            "5|regular empty file|1700000002\n/volume/runs/.sessions/x\0"
        )
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    live = SimpleNamespace(run=run)
    entries = volumes._live_entries(live, "runs/run1", recursive=False)
    assert [e.to_dict() for e in entries] == [
        {
            "path": "runs/run1/a|b.txt",
            "size": 12,
            "modified": "2023-11-14T22:13:21+00:00",
            "type": "file",
        },
        {"path": "runs/run1/ckpt/", "size": None, "modified": "2023-11-14T22:13:20+00:00",
         "type": "directory"},
        {
            "path": "runs/run1/first\nsecond.txt",
            "size": 7,
            "modified": "2023-11-14T22:13:23+00:00",
            "type": "file",
        },
    ]  # fmt: skip
    assert "-maxdepth 1" in seen[0] and "/volume/runs/run1" in seen[0]


@docker
def test_ls_and_du_report_a_name_with_a_newline_whole(live):
    _files(live.tree, {"datasets/first\nsecond.txt": b"abc"})
    listing = _run("ls", "data", "datasets", "-R", "-o", "json")
    assert listing.exit_code == 0, listing.output
    assert [e["path"] for e in json.loads(listing.stdout)] == ["datasets/first\nsecond.txt"]
    usage = _run("du", "data", "datasets", "-o", "json")
    assert json.loads(usage.stdout)["bytes"] == 3


@docker
def test_du_honors_a_trailing_slash_on_a_file(live):
    _files(live.tree, {"datasets/a.jsonl": b"12345"})
    plain = _run("du", "data", "datasets/a.jsonl", "-o", "json")
    assert json.loads(plain.stdout)["bytes"] == 5
    slashed = _run("du", "data", "datasets/a.jsonl/")
    assert slashed.exit_code == 1, slashed.output
    assert "No such file" in slashed.stderr and slashed.stdout == ""


def test_get_goes_through_the_live_view(monkeypatch, tmp_path):
    folder = tmp_path / "volume-ssh"
    folder.mkdir()
    key = tmp_path / "key"
    key.write_text("k")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(volumes, "_session_dir", lambda: folder)
    monkeypatch.setattr(volumes.Config, "ssh_key_path", property(lambda self: str(key)))
    routes, commands = [], []

    def route(name, mode, path, entries=None, team_id=None, lease_id=None, live=False):
        routes.append((mode, path, live))
        return VolumeTransferRoute(via="session", session=_live_session())

    client = SimpleNamespace(
        route_volume_transfer=route,
        create_volume_session=lambda *a, **kw: pytest.fail("the route picks the session"),
    )
    monkeypatch.setattr(volumes, "_client", lambda: (client, "t1"))
    monkeypatch.setattr(volumes.shutil, "which", lambda n: f"/bin/{n}")
    monkeypatch.setattr(
        volumes.subprocess,
        "run",
        lambda cmd, **kw: commands.append(cmd) or SimpleNamespace(returncode=0, stdout=""),
    )
    result = _run("get", "data", "runs/run1/checkpoints/step_10", "out")
    assert result.exit_code == 0, result.output
    assert routes == [("get", "runs/run1/checkpoints/step_10", True)]
    assert "Live read-only view of run1" in result.output
    assert commands[-1][-2:] == ["vol-ssh-x:/volume/runs/run1/checkpoints/step_10", "out"]


def test_put_never_asks_for_a_live_route(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("x")
    asked = []

    def route(name, mode, path, entries=None, team_id=None, lease_id=None, live=False):
        asked.append(live)
        raise volumes.APIError("stop here")

    monkeypatch.setattr(
        volumes, "_client", lambda: (SimpleNamespace(route_volume_transfer=route), "t1")
    )
    _run("put", "data", "f.txt", "/")
    assert asked == [False]


def test_storage_reads_print_the_routes_note(storage):  # noqa: F811
    storage.route.note = "the run run1 is syncing its changes back; they appear once that finishes"
    result = _run("ls", "data", "-o", "json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)
    assert "Note: the run run1 is syncing" in result.stderr


def test_client_sends_live_only_when_asked():
    posted = []
    api = SimpleNamespace(post=lambda path, json: posted.append(json) or {"via": "r2"})
    client = HostedTrainingClient(api)
    client.route_volume_transfer("v", "get", "", live=True)
    client.route_volume_transfer("v", "get", "")
    assert posted == [{"mode": "get", "path": "", "live": True}, {"mode": "get", "path": ""}]
    route = VolumeTransferRoute.model_validate({"via": "r2", "note": "syncing"})
    assert route.note == "syncing"
    assert _live_session().live_view_of == "run1"
