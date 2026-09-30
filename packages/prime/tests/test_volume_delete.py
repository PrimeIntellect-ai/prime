from types import SimpleNamespace

import httpx
import pytest
from prime_cli.api.training import HostedTrainingClient
from prime_cli.commands import volumes
from prime_cli.core import APIClient, APIError, NotFoundError
from prime_cli.main import app
from typer.testing import CliRunner


def _in_use(kind: str | None, count: int = 2) -> APIError:
    message = (
        f"HTTP 409: volume 'data' is mounted by {count} live run(s); "
        "stop them before deleting the volume"
    )
    error = APIError(message)
    if kind:
        error.body = {
            "detail": message,
            "errorCode": "volume_in_use",
            "kind": kind,
            "count": count,
        }
    return error


def _out(result) -> str:
    """Output with Rich's line wrapping undone."""
    return " ".join(result.output.split())


class FakeClient:
    def __init__(self, delete_errors, sessions=(), polls=None, list_error=None):
        self.delete_errors = list(delete_errors)
        self.sessions = [SimpleNamespace(id=s) for s in sessions]
        # Per session id, the statuses (or errors) successive polls return.
        self.polls = {k: list(v) for k, v in (polls or {}).items()}
        self.list_error = list_error
        self.calls = []

    def delete_volume(self, name, team_id=None):
        self.calls.append(("delete", name))
        if self.delete_errors:
            error = self.delete_errors.pop(0)
            if error:
                raise error

    def list_volume_sessions(self, name, team_id=None):
        self.calls.append(("list", name))
        if self.list_error:
            raise self.list_error
        return self.sessions

    def stop_volume_session(self, name, session_id, team_id=None):
        self.calls.append(("stop", session_id))

    def get_volume_session(self, name, session_id, team_id=None):
        self.calls.append(("get", session_id))
        result = self.polls[session_id].pop(0)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(id=session_id, status=result)


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr(volumes.time, "sleep", lambda s: None)

    def _run(client, *args, input=None):
        monkeypatch.setattr(volumes, "_client", lambda: (client, "t1"))
        return CliRunner().invoke(
            app,
            ["volumes", "delete", "data", *args],
            input=input,
            env={"PRIME_DISABLE_VERSION_CHECK": "1"},
        )

    return _run


def test_runs_refusal_keeps_the_message_and_hints_train_stop(run):
    client = FakeClient([_in_use("runs")])
    result = run(client, "--yes")
    assert result.exit_code == 1
    assert "mounted by 2 live run(s)" in _out(result)
    assert "prime train stop" in _out(result)
    assert client.calls == [("delete", "data")]


def test_old_backend_refusal_hints_both_stop_commands(run):
    """Older backends send only `detail`, counting runs and sessions together."""
    client = FakeClient([_in_use(None)])
    result = run(client, "--yes")
    assert result.exit_code == 1
    assert "mounted by 2 live run(s)" in _out(result)
    assert "prime train stop" in _out(result) and "prime volumes stop data" in _out(result)
    assert client.calls == [("delete", "data")]


def test_sessions_refusal_ends_sessions_waits_then_retries_delete(run):
    client = FakeClient(
        [_in_use("sessions"), None],
        sessions=["s1", "s2"],
        polls={
            "s1": ["TERMINATING", NotFoundError("gone")],
            "s2": [APIError("502"), "STOPPED"],
        },
    )
    result = run(client, input="y\ny\n")
    assert result.exit_code == 0, result.output
    assert "has 2 active SSH session(s). End them and delete the volume?" in _out(result)
    assert client.calls == [
        ("delete", "data"),
        ("list", "data"),
        ("stop", "s1"),
        ("stop", "s2"),
        ("get", "s1"),
        ("get", "s2"),
        ("get", "s1"),
        ("get", "s2"),
        ("delete", "data"),
    ]
    assert "Session s1 ended. Session s2 ended." in _out(result)
    assert "Deleting volume data." in _out(result)


def test_declining_the_prompt_stops_nothing(run):
    client = FakeClient([_in_use("sessions", 1)], sessions=["s1"])
    result = run(client, input="y\nn\n")
    assert result.exit_code == 0
    assert client.calls == [("delete", "data"), ("list", "data")]


def test_other_members_sessions_are_not_touched(run):
    client = FakeClient([_in_use("sessions", 3)], sessions=["s1"])
    result = run(client, "--yes")
    assert result.exit_code == 1
    assert "1 of them are yours" in _out(result)
    assert client.calls == [("delete", "data"), ("list", "data")]


def test_backend_without_list_route_asks_to_stop_manually(run):
    client = FakeClient([_in_use("sessions")], list_error=NotFoundError("HTTP 404"))
    result = run(client, "--yes")
    assert result.exit_code == 1
    assert "cannot list them" in _out(result)
    assert "prime volumes stop data <session-id>" in _out(result)
    assert ("stop", "s1") not in client.calls


def test_sessions_that_never_end_time_out_without_retrying(run, monkeypatch):
    monkeypatch.setattr(volumes, "_SESSION_END_TIMEOUT", 0)
    client = FakeClient([_in_use("sessions", 1)], sessions=["s1"], polls={"s1": ["TERMINATING"]})
    result = run(client, "--yes")
    assert result.exit_code == 1
    assert "still ending" in _out(result)
    assert "prime volumes delete data" in _out(result)
    assert client.calls.count(("delete", "data")) == 1


def test_client_lists_sessions():
    requests = []

    class FakeAPI:
        def get(self, path, params):
            requests.append((path, params))
            return {
                "sessions": [
                    {
                        "id": "s1",
                        "volumeName": "data",
                        "status": "RUNNING",
                        "readOnly": True,
                        "createdAt": "2026-09-30T00:00:00",
                    }
                ]
            }

    (session,) = HostedTrainingClient(FakeAPI()).list_volume_sessions("data", team_id="t1")
    assert (session.id, session.created_at) == ("s1", "2026-09-30T00:00:00")
    assert requests == [("/training/volumes/data/sessions", {"teamId": "t1"})]


def test_api_error_keeps_the_structured_error_body(monkeypatch):
    body = {"detail": "busy", "errorCode": "volume_in_use", "kind": "sessions", "count": 1}
    client = APIClient(api_key="k")
    client.client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(409, json=body))
    )
    with pytest.raises(APIError) as err:
        client.delete("/training/volumes/data")
    assert str(err.value) == "HTTP 409: busy"
    assert err.value.body == body
