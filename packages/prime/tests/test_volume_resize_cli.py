from types import SimpleNamespace

from prime_cli.api.training import Volume
from prime_cli.commands import volumes
from prime_cli.core import APIError
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()
TEST_ENV = {"PRIME_DISABLE_VERSION_CHECK": "1"}


def _volume(size="1Ti", pending=None, error=None):
    return Volume.model_validate({
        "name": "data", "size": size, "resizePending": pending, "resizeError": error,
        "status": "RUNNING", "clusterId": "cluster", "namespace": "ns", "pvcName": "pvc",
    })


def test_resize_waits_until_committed(monkeypatch):
    observed = []
    responses = [_volume(pending="2Ti"), _volume(size="2Ti")]
    client = SimpleNamespace(
        resize_volume=lambda name, size, team_id: _volume(pending="2Ti"),
        list_volumes=lambda team_id, timeout: (
            observed.append((team_id, timeout)) or [responses.pop(0)]
        ),
    )
    monkeypatch.setattr(volumes, "_client", lambda: (client, "team"))
    monkeypatch.setattr(volumes.time, "sleep", lambda seconds: None)

    result = runner.invoke(app, ["volumes", "resize", "data", "--size", "2Ti"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Volume data is now 2Ti" in result.output
    assert observed == [("team", 10), ("team", 10)]


def test_resize_timeout_reports_pending_not_failure(monkeypatch):
    client = SimpleNamespace(
        resize_volume=lambda name, size, team_id: _volume(pending="2Ti"),
        list_volumes=lambda team_id, timeout: [_volume(pending="2Ti")],
    )
    monkeypatch.setattr(volumes, "_client", lambda: (client, None))
    monkeypatch.setattr(volumes.time, "monotonic", iter([0, 180]).__next__)

    result = runner.invoke(app, ["volumes", "resize", "data", "--size", "2Ti"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "still resizing" in result.output
    assert "prime volumes list" in result.output


def test_resize_native_success_without_pending(monkeypatch):
    client = SimpleNamespace(resize_volume=lambda name, size, team_id: _volume(size="2Ti"))
    monkeypatch.setattr(volumes, "_client", lambda: (client, None))

    result = runner.invoke(app, ["volumes", "resize", "data", "--size", "2Ti"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Volume data is now 2Ti" in result.output


def test_resize_poll_error_does_not_imply_resize_failed(monkeypatch):
    def failed_list(team_id, timeout):
        raise APIError("Service unavailable")

    client = SimpleNamespace(
        resize_volume=lambda name, size, team_id: _volume(pending="2Ti"),
        list_volumes=failed_list,
    )
    monkeypatch.setattr(volumes, "_client", lambda: (client, None))
    monkeypatch.setattr(volumes.time, "sleep", lambda seconds: None)

    result = runner.invoke(app, ["volumes", "resize", "data", "--size", "2Ti"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "Service unavailable" in result.output
    assert "Resize may still be running" in result.output
    assert "prime volumes list" in result.output


def test_resize_exposes_persistent_backend_error(monkeypatch):
    calls = []
    client = SimpleNamespace(
        resize_volume=lambda name, size, team_id: _volume(pending="2Ti"),
        list_volumes=lambda team_id, timeout: calls.append(1) or [
            _volume(pending="2Ti", error="ControllerResizeError: bad <quota>")
        ],
    )
    monkeypatch.setattr(volumes, "_client", lambda: (client, None))
    monkeypatch.setattr(volumes.time, "sleep", lambda seconds: None)

    result = runner.invoke(app, ["volumes", "resize", "data", "--size", "2Ti"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "ControllerResizeError: bad <quota>" in result.output
    assert "Resize remains pending" in result.output
    assert "Contact support" in result.output
    assert calls == [1]


def test_resize_replay_exposes_existing_error_without_poll(monkeypatch):
    client = SimpleNamespace(
        resize_volume=lambda name, size, team_id: _volume(
            pending="2Ti", error="PVC missing"
        ),
    )
    monkeypatch.setattr(volumes, "_client", lambda: (client, None))

    result = runner.invoke(app, ["volumes", "resize", "data", "--size", "2Ti"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "PVC missing" in result.output
    assert "Resize remains pending" in result.output
