"""`prime volumes sessions` — listing a volume's SSH sessions.

Hermetic: a fake HostedTrainingClient; no network.
"""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from prime_cli.api.training import HostedTrainingClient, VolumeSession
from prime_cli.commands import volumes
from prime_cli.core import APIError
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

ENV = {"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Pin the module console to a wide, deterministic size.

    The console is built once at import, and rich ignores a width set without
    a height (it falls back to the terminal / dumb-terminal 80x25), so a
    width-only setting would make these table assertions depend on the
    terminal the tests happen to run under.
    """
    monkeypatch.setattr(volumes.console, "_width", 200)
    monkeypatch.setattr(volumes.console, "_height", 50)


def _run(*args):
    return CliRunner().invoke(app, ["volumes", *args], env=ENV)


def _text(result) -> str:
    """Result output with ANSI stripped, so assertions hold on a tty too."""
    return strip_ansi(result.output)


def _session(**over) -> VolumeSession:
    fields = {
        "id": "s1",
        "volumeName": "data",
        "status": "RUNNING",
        "readOnly": True,
        "sshConnection": "prime@vol-ssh-abc.tailde92d5.ts.net",
        "createdAt": "2026-10-01T07:15:03Z",
    }
    fields.update(over)
    return VolumeSession.model_validate(fields)


def _client(monkeypatch, **methods):
    """Install a fake client; `methods` are the client calls to expose."""
    monkeypatch.setattr(volumes, "_client", lambda: (SimpleNamespace(**methods), "team1"))


def test_client_sessions_wire_contract():
    wire = {
        "id": "s1",
        "volumeName": "data",
        "status": "RUNNING",
        "readOnly": False,
        "sshConnection": "prime@vol-ssh-abc.tailde92d5.ts.net",
        "createdAt": "2026-10-01T07:15:03Z",
    }
    requests = []

    class FakeAPI:
        def get(self, path, params):
            requests.append(("GET", path, params))
            return {"sessions": [wire, {**wire, "id": "s2"}]}

    client = HostedTrainingClient(FakeAPI())
    listed = client.list_volume_sessions("data", team_id="team1")
    assert [s.id for s in listed] == ["s1", "s2"]
    assert listed[0].created_at == "2026-10-01T07:15:03Z"
    assert listed[0].ssh_connection == "prime@vol-ssh-abc.tailde92d5.ts.net"
    assert requests == [("GET", "/training/volumes/data/sessions", {"teamId": "team1"})]


def test_client_sessions_without_team_param():
    requests = []

    class FakeAPI:
        def get(self, path, params):
            requests.append(params)
            return {"sessions": []}

    assert HostedTrainingClient(FakeAPI()).list_volume_sessions("data") == []
    assert requests == [None]


def test_sessions_table(monkeypatch):
    _client(
        monkeypatch,
        list_volume_sessions=lambda name, **kw: [
            _session(),
            _session(id="s2", readOnly=False, status="TOMBSTONED"),
        ],
    )
    result = _run("sessions", "data")
    assert result.exit_code == 0, _text(result)
    for header in ("VOLUME", "SESSION ID", "STATUS", "MODE", "CREATED"):
        assert header in _text(result)
    for cell in ("data", "s1", "s2", "RUNNING", "TOMBSTONED", " ro ", " rw "):
        assert cell in _text(result)


def test_sessions_output_json_is_full_dicts(monkeypatch):
    _client(
        monkeypatch,
        list_volume_sessions=lambda name, **kw: [_session(), _session(id="s2", readOnly=False)],
    )
    result = _run("sessions", "data", "--output", "json")
    assert result.exit_code == 0, _text(result)
    data = json.loads(_text(result))
    assert [s["id"] for s in data] == ["s1", "s2"]
    assert data[0]["volumeName"] == "data"
    assert data[0]["sshConnection"] == "prime@vol-ssh-abc.tailde92d5.ts.net"
    assert data[0]["createdAt"] == "2026-10-01T07:15:03Z"
    assert data[0]["readOnly"] is True
    assert data[1]["readOnly"] is False


def test_sessions_without_name_spans_volumes(monkeypatch):
    calls = []
    sessions = {
        "data": [_session()],
        "scratch": [_session(id="s2", readOnly=False, volumeName="scratch")],
    }

    def fake_list(name, **kw):
        calls.append((name, kw))
        return sessions.get(name, [])

    _client(
        monkeypatch,
        list_volumes=lambda **kw: [
            SimpleNamespace(name="data"),
            SimpleNamespace(name="datasets-empty"),
            SimpleNamespace(name="scratch"),
        ],
        list_volume_sessions=fake_list,
    )
    result = _run("sessions")
    assert result.exit_code == 0, _text(result)
    assert calls == [
        ("data", {"team_id": "team1"}),
        ("datasets-empty", {"team_id": "team1"}),
        ("scratch", {"team_id": "team1"}),
    ]
    for volume in ("data", "scratch"):
        assert volume in _text(result)
    assert "s1" in _text(result) and "s2" in _text(result)
    # A volume with no sessions never appears in the table.
    assert "datasets-empty" not in _text(result)


def test_sessions_without_name_json(monkeypatch):
    sessions = {
        "data": [_session()],
        "scratch": [_session(id="s2", volumeName="scratch")],
    }
    _client(
        monkeypatch,
        list_volumes=lambda **kw: [
            SimpleNamespace(name="data"),
            SimpleNamespace(name="scratch"),
        ],
        list_volume_sessions=lambda name, **kw: sessions.get(name, []),
    )
    result = _run("sessions", "--output", "json")
    assert result.exit_code == 0, _text(result)
    data = json.loads(_text(result))
    assert [s["id"] for s in data] == ["s1", "s2"]
    assert {s["volumeName"] for s in data} == {"data", "scratch"}


def test_sessions_empty_list_is_an_empty_table(monkeypatch):
    calls = []
    _client(monkeypatch, list_volume_sessions=lambda name, **kw: calls.append(name) or [])
    result = _run("sessions", "data")
    assert result.exit_code == 0, _text(result)
    assert calls == ["data"]
    assert "SESSION ID" in _text(result)
    assert "RUNNING" not in _text(result)


def test_sessions_without_name_and_without_volumes_lists_nothing(monkeypatch):
    calls = []
    _client(
        monkeypatch,
        list_volumes=lambda **kw: [],
        list_volume_sessions=lambda name, **kw: calls.append(name) or [],
    )
    result = _run("sessions")
    assert result.exit_code == 0, _text(result)
    assert calls == []


@pytest.mark.parametrize("args", [["sessions", "data"], ["sessions"]])
def test_sessions_api_error_exits(monkeypatch, args):
    def boom(*a, **kw):
        raise APIError("volume 'data' not found")

    _client(
        monkeypatch,
        list_volumes=lambda **kw: [SimpleNamespace(name="data")],
        list_volume_sessions=boom,
    )
    result = _run(*args)
    assert result.exit_code == 1
    assert "volume 'data' not found" in _text(result)


def test_sessions_without_name_aborts_on_a_volume_error(monkeypatch):
    def fake_list(name, **kw):
        if name == "scratch":
            raise APIError("503 over quota")
        return [_session(volumeName=name)] if name == "data" else []

    _client(
        monkeypatch,
        list_volumes=lambda **kw: [
            SimpleNamespace(name="data"),
            SimpleNamespace(name="scratch"),
        ],
        list_volume_sessions=fake_list,
    )
    result = _run("sessions")
    assert result.exit_code == 1
    assert "503 over quota" in _text(result)


def test_session_age_helper():
    now = datetime.now(timezone.utc)
    assert volumes._session_age(None) == "-"
    assert volumes._session_age("not-a-timestamp") == "-"
    recent = (now - timedelta(seconds=30)).isoformat()
    assert volumes._session_age(recent) == "30s"
    older = (now - timedelta(days=2, hours=3)).isoformat()
    assert volumes._session_age(older) == "2d"
