from types import SimpleNamespace

import httpx
import pytest
from prime_cli.api.training import HostedTrainingClient
from prime_cli.commands import volumes
from prime_cli.core import APIClient, APIError, NotFoundError
from prime_cli.main import app
from typer.testing import CliRunner


def _in_use(kind: str | None, count: int = 2) -> APIError:
    if kind == "sessions":
        detail = (
            f"volume 'data' has {count} active volume SSH session(s); "
            "end them before deleting the volume"
        )
    else:
        detail = (
            f"volume 'data' is mounted by {count} live run(s); stop them before deleting the volume"
        )
    error = APIError(f"HTTP 409: {detail}")
    if kind:
        error.body = {
            "detail": detail,
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
        self.sessions = [
            SimpleNamespace(
                id=s, status="RUNNING", read_only=True, created_at="2026-09-30T00:00:00"
            )
            for s in sessions
        ]
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


def _unsupported(status: int) -> APIError:
    """What a backend without GET /volumes/{name}/sessions answers."""
    detail = {404: "Not Found", 405: "Method Not Allowed"}[status]
    error = (NotFoundError if status == 404 else APIError)(f"HTTP {status}: {detail}")
    error.body = {"detail": detail}
    return error


FULL_FLOW_CALLS = [
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


def _two_sessions(delete_errors):
    return FakeClient(
        delete_errors,
        sessions=["s1", "s2"],
        polls={
            "s1": ["TERMINATING", NotFoundError("gone")],
            "s2": [APIError("502"), "STOPPED"],
        },
    )


def test_choice_1_shows_sessions_ends_them_waits_then_retries_delete(run):
    client = _two_sessions([_in_use("sessions"), None])
    result = run(client, input="y\n1\n")
    assert result.exit_code == 0, result.output
    out = _out(result)
    # The standard error line comes first, then the table, then the prompt.
    assert out.startswith(
        "Delete volume data and all run data on it? [y/N]: y "
        "Error: HTTP 409: volume 'data' has 2 active volume SSH session(s); "
        "end them before deleting the volume ┏"
    )
    assert "active SSH session(s). ┏" not in out
    assert "┃ Session ┃ Status ┃ Read-only ┃ Created ┃" in out
    assert "│ s1 │ RUNNING │ yes │ 2026-09-30T00:00:00 │" in out
    assert "1) end session(s) and delete the volume" in out
    assert "3) cancel Select [3]:" in out
    # The table comes before the prompt.
    assert out.index("│ s2 │") < out.index("Select [3]")
    assert client.calls == FULL_FLOW_CALLS
    assert "Session s1 ended. Session s2 ended." in out
    assert "Deleting volume data." in out


def test_yes_picks_choice_1_without_prompting(run):
    client = _two_sessions([_in_use("sessions"), None])
    result = run(client, "--yes")
    assert result.exit_code == 0, result.output
    assert "Select" not in result.output
    assert "│ s1 │ RUNNING │" in _out(result)
    assert client.calls == FULL_FLOW_CALLS


def test_choice_2_ends_sessions_and_keeps_the_volume(run):
    client = _two_sessions([_in_use("sessions")])
    result = run(client, input="y\n2\n")
    assert result.exit_code == 0, result.output
    assert client.calls == FULL_FLOW_CALLS[:-1]
    assert "Session(s) ended; volume data kept." in _out(result)
    assert "Deleting volume" not in result.output


@pytest.mark.parametrize("answer", ["3\n", "\n"], ids=["choice-3", "enter-defaults-to-3"])
def test_choice_3_cancels_and_stops_nothing(run, answer):
    client = FakeClient([_in_use("sessions", 1)], sessions=["s1"])
    result = run(client, input="y\n" + answer)
    assert result.exit_code == 0
    assert "prime volumes stop data <session-id>" in _out(result)
    assert client.calls == [("delete", "data"), ("list", "data")]


def test_other_members_sessions_suppress_the_prompt(run):
    client = FakeClient([_in_use("sessions", 3)], sessions=["s1"])
    result = run(client, input="y\n")
    assert result.exit_code == 1
    assert "Only 1 of the 3 session(s) are yours" in _out(result)
    assert "Select" not in result.output
    assert client.calls == [("delete", "data"), ("list", "data")]


@pytest.mark.parametrize("status", [404, 405])
def test_backend_without_list_route_asks_to_stop_manually(run, status):
    client = FakeClient([_in_use("sessions")], list_error=_unsupported(status))
    result = run(client, "--yes")
    assert result.exit_code == 1
    assert "cannot list the sessions" in _out(result)
    assert "prime volumes stop data <session-id>" in _out(result)
    assert "Select" not in result.output
    assert client.calls == [("delete", "data"), ("list", "data")]


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
