import json
from types import SimpleNamespace

import pytest
from prime_cli.api.training import VolumeSession
from prime_cli.commands import volumes
from prime_cli.core import APIError, NotFoundError
from prime_cli.main import app
from typer.testing import CliRunner

SESSION = VolumeSession(
    id="s1",
    volumeName="data",
    status="RUNNING",
    readOnly=False,
    createdAt="2026-09-30T00:00:00",
)


def _run(monkeypatch, list_volume_sessions, *args):
    client = SimpleNamespace(list_volume_sessions=list_volume_sessions)
    monkeypatch.setattr(volumes, "_client", lambda: (client, "t1"))
    return CliRunner().invoke(
        app, ["volumes", "sessions", "data", *args], env={"PRIME_DISABLE_VERSION_CHECK": "1"}
    )


def test_sessions_table(monkeypatch):
    calls = []
    result = _run(monkeypatch, lambda name, team_id: calls.append((name, team_id)) or [SESSION])
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    for cell in ("Session", "Status", "Read-only", "Created", "s1", "RUNNING", "no"):
        assert cell in out
    assert "2026-09-30T00:00:00" in out
    assert calls == [("data", "t1")]


def test_sessions_json(monkeypatch):
    result = _run(monkeypatch, lambda name, team_id: [SESSION], "--output", "json")
    assert result.exit_code == 0, result.output
    (row,) = json.loads(result.output)
    assert (row["id"], row["status"], row["readOnly"], row["createdAt"]) == (
        "s1",
        "RUNNING",
        False,
        "2026-09-30T00:00:00",
    )


@pytest.mark.parametrize(
    "error_cls,status,detail",
    [(NotFoundError, 404, "Not Found"), (APIError, 405, "Method Not Allowed")],
)
def test_sessions_on_a_backend_without_the_route(monkeypatch, error_cls, status, detail):
    def unsupported(name, team_id):
        error = error_cls(f"HTTP {status}: {detail}")
        error.body = {"detail": detail}
        raise error

    result = _run(monkeypatch, unsupported)
    assert result.exit_code == 1
    assert "not supported by this backend" in " ".join(result.output.split())


def test_sessions_on_a_missing_volume_shows_the_error(monkeypatch):
    def missing(name, team_id):
        error = NotFoundError("HTTP 404: volume 'data' not found")
        error.body = {"detail": "volume 'data' not found"}
        raise error

    result = _run(monkeypatch, missing)
    assert result.exit_code == 1
    assert "volume 'data' not found" in result.output
