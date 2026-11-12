"""Tests for `prime factory slurm` (the deployment roster surface)."""

import json
from typing import Any, Dict, List, Optional

import pytest
from prime_cli.client import APIError
from prime_cli.main import app
from prime_cli.utils.formatters import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "PRIME_API_KEY": "dummy",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "COLUMNS": "220",
    "TERM": "xterm",
}


def _cluster(cluster_id: str = "job-123", **overrides: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": cluster_id,
        "primeClusterId": "pc-1",
        "displayName": "research-b300-slurm",
        "status": "RUNNING",
        "gpuType": "B300",
        "gpuCount": 48,
        "createdAt": "2026-10-08T10:00:00Z",
        "startedAt": "2026-10-08T10:05:00Z",
    }
    row.update(overrides)
    return row


def _member(username: str = "carol", **overrides: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "username": username,
        "uid": 1001,
        "sshAuthorizedKeys": ["ssh-ed25519 AAAA key1 comment"],
        "sudo": False,
        "status": "ACTIVE",
        "linkedUserId": None,
        "linkedUserName": None,
        "linkedUserEmail": None,
    }
    row.update(overrides)
    return row


def _clusters_payload(rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {"data": rows if rows is not None else [_cluster()]}


def _members_payload(rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {"data": rows if rows is not None else [_member()]}


def _connection_payload(**overrides: Any) -> Dict[str, Any]:
    # snake_case on purpose: that is how the API spells this payload.
    row: Dict[str, Any] = {
        "connectable": True,
        "ssh_host": "203.0.113.10",
        "ssh_port": 22033,
        "ssh_command": "ssh -p 22033 username@203.0.113.10",
    }
    row.update(overrides)
    return row


class _StubConfig:
    def __init__(self, team_id: Optional[str]) -> None:
        self._team_id = team_id

    @property
    def team_id(self) -> Optional[str]:
        return self._team_id


class _DummyAPIClient:
    def __init__(
        self,
        get_payload: Optional[Dict[str, Any]] = None,
        connection_payload: Optional[Dict[str, Any]] = None,
        connection_error: Optional[Exception] = None,
    ) -> None:
        self._get_payload = get_payload
        self._connection_payload = connection_payload
        self._connection_error = connection_error
        self.calls: List[Dict[str, Any]] = []
        self.post_calls: List[Dict[str, Any]] = []

    def get(self, endpoint, params=None, timeout=None):
        self.calls.append({"endpoint": endpoint, "params": params})
        # `/connection` is a separate route with its own payload shape.
        if endpoint.endswith("/connection"):
            if self._connection_error is not None:
                raise self._connection_error
            return self._connection_payload
        return self._get_payload

    def post(self, endpoint, json=None):
        self.post_calls.append({"endpoint": endpoint, "json": json})
        body = dict(json or {})
        body.setdefault("uid", 1001)
        body.setdefault("sudo", False)
        body.setdefault("status", "ACTIVE")
        return body

    def delete(self, endpoint, params=None):
        self.calls.append({"endpoint": endpoint, "params": params})


def _install(
    monkeypatch: pytest.MonkeyPatch,
    get_payload: Optional[Dict[str, Any]] = None,
    team_id: Optional[str] = "team-123",
    connection_payload: Optional[Dict[str, Any]] = None,
    connection_error: Optional[Exception] = None,
) -> _DummyAPIClient:
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    dummy = _DummyAPIClient(
        get_payload=get_payload,
        connection_payload=connection_payload if connection_payload else _connection_payload(),
        connection_error=connection_error,
    )
    monkeypatch.setattr("prime_cli.commands.factory_slurm.APIClient", lambda: dummy)
    monkeypatch.setattr("prime_cli.commands.factory_slurm.Config", lambda: _StubConfig(team_id))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return dummy


def test_slurm_list_table_and_json_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _clusters_payload()
    dummy = _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "slurm", "list"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "research-b300-slurm" in output
    assert "RUNNING" in output and "B300" in output and "48" in output
    assert dummy.calls[0]["endpoint"] == "/slurm-clusters/team-123"

    json_result = runner.invoke(app, ["factory", "slurm", "list", "--json"], env=TEST_ENV)
    assert json.loads(json_result.stdout) == payload


def test_slurm_list_empty_and_no_team(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"data": []})
    result = runner.invoke(app, ["factory", "slurm", "list"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "No Slurm deployments found." in strip_ansi(result.output)

    _install(monkeypatch, {"data": []}, team_id=None)
    hint = runner.invoke(app, ["factory", "slurm", "list"], env=TEST_ENV)
    assert hint.exit_code == 0, hint.output
    assert "prime switch" in strip_ansi(hint.output)


def test_slurm_get_detail_and_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _clusters_payload())

    result = runner.invoke(app, ["factory", "slurm", "get", "job-123"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "research-b300-slurm" in output
    assert "48" in output

    miss = runner.invoke(app, ["factory", "slurm", "get", "nope"], env=TEST_ENV)
    miss_output = strip_ansi(miss.output)
    assert miss.exit_code == 1, miss.output
    assert "No Slurm deployment matched 'nope'" in miss_output


def test_slurm_get_renders_ssh_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch, _clusters_payload())

    result = runner.invoke(app, ["factory", "slurm", "get", "job-123"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "SSH host: 203.0.113.10" in output
    assert "SSH port: 22033" in output
    # Rendered from the server's own ssh_command, never rebuilt here.
    assert "SSH: ssh -p 22033 username@203.0.113.10" in output
    # The team is a query parameter on this route, not a path segment.
    connection_call = dummy.calls[-1]
    assert connection_call["endpoint"] == "/slurm-clusters/job-123/connection"
    assert connection_call["params"] == {"team_id": "team-123"}


def test_slurm_get_json_merges_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    # The cluster row is echoed verbatim (including fields this CLI
    # version does not model) with the connection payload attached.
    row = _cluster(volumeBindings=[{"volumeId": "vol-1"}])
    _install(monkeypatch, _clusters_payload([row]))

    result = runner.invoke(app, ["factory", "slurm", "get", "job-123", "--json"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {**row, "connection": _connection_payload()}


def test_slurm_get_omits_port_when_the_api_omits_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # A null ssh_port is the default port 22, not a missing value: the
    # CLI must render the server's command as-is rather than inventing
    # a -p flag the dashboard doesn't show.
    _install(
        monkeypatch,
        _clusters_payload(),
        connection_payload=_connection_payload(
            ssh_port=None, ssh_command="ssh username@203.0.113.10"
        ),
    )
    result = runner.invoke(app, ["factory", "slurm", "get", "job-123"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "SSH host: 203.0.113.10" in output
    assert "SSH port:" not in output
    assert "SSH: ssh username@203.0.113.10" in output
    assert "-p" not in output


def test_slurm_get_renders_command_override_without_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An admin-set full-command override is opaque (jump hosts, custom
    # identity files) and arrives with no host or port to render.
    _install(
        monkeypatch,
        _clusters_payload(),
        connection_payload=_connection_payload(
            ssh_host=None,
            ssh_port=None,
            ssh_command="ssh -J bastion.example.com username@login.internal",
        ),
    )
    result = runner.invoke(app, ["factory", "slurm", "get", "job-123"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "SSH: ssh -J bastion.example.com username@login.internal" in output
    assert "SSH host:" not in output and "SSH port:" not in output


def test_slurm_get_never_invents_connect_info(monkeypatch: pytest.MonkeyPatch) -> None:
    # A deployment the API reports as not connectable has no usable
    # login target: the detail view renders what the API provides and
    # never fabricates a host or port.
    _install(
        monkeypatch,
        _clusters_payload([_cluster(status="PENDING")]),
        connection_payload=_connection_payload(
            connectable=False, ssh_host=None, ssh_port=None, ssh_command=None
        ),
    )
    result = runner.invoke(app, ["factory", "slurm", "get", "job-123"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "SSH: unavailable" in output
    assert "SSH host:" not in output and "SSH port:" not in output


def test_slurm_get_survives_connection_lookup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # The roster details are still worth printing when only the connect
    # lookup fails (older backend, missing scope).
    _install(
        monkeypatch,
        _clusters_payload(),
        connection_error=APIError("HTTP 404: Not Found"),
    )

    result = runner.invoke(app, ["factory", "slurm", "get", "job-123"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "research-b300-slurm" in output
    assert "Could not read SSH coordinates" in output
    # Not "unavailable": a failed lookup knows nothing either way, and
    # claiming the deployment isn't RUNNING would contradict the
    # RUNNING status printed two lines above.
    assert "SSH: unknown" in output
    assert "not RUNNING" not in output

    json_result = runner.invoke(app, ["factory", "slurm", "get", "job-123", "--json"], env=TEST_ENV)
    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.stdout)["connection"] is None


def test_slurm_members_table_and_json(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _members_payload(
        [_member("carol"), _member("bob", uid=1002, sudo=True, sshAuthorizedKeys=[])]
    )
    dummy = _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "slurm", "members", "job-123"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "carol" in output and "bob" in output
    assert "ssh-ed25519 AAAA" in output  # truncated key line still visible
    assert dummy.calls[0]["endpoint"] == "/slurm-clusters/team-123/job-123/members"

    json_result = runner.invoke(
        app, ["factory", "slurm", "members", "job-123", "--json"], env=TEST_ENV
    )
    assert json.loads(json_result.stdout) == payload


def test_slurm_add_member_body_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch)

    result = runner.invoke(
        app,
        [
            "factory",
            "slurm",
            "add-member",
            "job-123",
            "dave",
            "--ssh-key",
            "ssh-ed25519 AAAA dave-key",
            "--link-user",
            "user-9",
        ],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "Successfully added dave" in output
    assert dummy.post_calls[0]["endpoint"] == "/slurm-clusters/team-123/job-123/members"
    body = dummy.post_calls[0]["json"]
    assert body["username"] == "dave"
    assert body["sshAuthorizedKeys"] == ["ssh-ed25519 AAAA dave-key"]
    assert body["linkedUserId"] == "user-9"


def test_slurm_remove_member_confirms_and_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch)

    skipped = runner.invoke(
        app,
        ["factory", "slurm", "remove-member", "job-123", "carol"],
        env=TEST_ENV,
        input="n\n",
    )
    assert skipped.exit_code == 0, skipped.output
    assert "Cancelled" in strip_ansi(skipped.output)
    assert not dummy.calls

    result = runner.invoke(
        app,
        ["factory", "slurm", "remove-member", "job-123", "carol", "--yes"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "Successfully removed carol" in output
    assert dummy.calls[0]["endpoint"] == "/slurm-clusters/team-123/job-123/members/carol"


def test_slurm_escaping_regressions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rich markup in cluster names, statuses, usernames, and key lines must
    # never crash rendering or be interpreted as markup.
    payload = _clusters_payload([_cluster(displayName="slurm-[bold]team", status="RUN[red]NING")])
    _install(monkeypatch, payload)
    result = runner.invoke(app, ["factory", "slurm", "list"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "slurm-[bold]team" in output
    assert "RUN[red]NING" in output

    _install(
        monkeypatch,
        _members_payload([_member("user-[bold]name", sshAuthorizedKeys=["key [red] line"])]),
    )
    members = runner.invoke(app, ["factory", "slurm", "members", "job-123"], env=TEST_ENV)
    members_output = strip_ansi(members.output)
    assert members.exit_code == 0, members.output
    assert "user-[bold]name" in members_output


def test_slurm_list_malformed_success_body_is_clean_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenClient:
        def get(self, endpoint, params=None, timeout=None):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr("prime_cli.commands.factory_slurm.APIClient", lambda: _BrokenClient())
    monkeypatch.setattr("prime_cli.commands.factory_slurm.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(app, ["factory", "slurm", "list"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 1, result.output
    assert "malformed response body" in output
    assert "ValueError" not in result.output


def test_slurm_list_schema_drift_is_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A row missing gpuCount: pydantic drift surfaces as the coarse
    # clean error in BOTH table and json modes — never a traceback.
    bad_payload = {
        "data": [
            {
                "id": "job-123",
                "primeClusterId": "pc-1",
                "displayName": "broken-row",
                "status": "RUNNING",
                "gpuType": "B300",
                # gpuCount missing
                "createdAt": "2026-10-08T10:00:00Z",
            }
        ]
    }
    _install(monkeypatch, bad_payload)

    result = runner.invoke(app, ["factory", "slurm", "list"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 1, result.output
    assert "malformed response body" in output
    assert "ValidationError" not in result.output

    json_result = runner.invoke(app, ["factory", "slurm", "list", "--json"], env=TEST_ENV)
    json_output = strip_ansi(json_result.output)
    assert json_result.exit_code == 1, json_result.output
    assert "malformed response body" in json_output
    # stdout carries no partial JSON


def test_slurm_json_mode_diagnostics_stay_off_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Missing team in --json mode: the hint goes to stderr, stdout stays
    # parseable (empty) for JSON consumers.
    _install(monkeypatch, {"data": []}, team_id=None)

    result = runner.invoke(app, ["factory", "slurm", "list", "--json"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "prime switch" not in result.stdout
    assert "No team" not in result.stdout


def test_slurm_add_member_malformed_response_is_clean_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BadPostClient:
        def post(self, endpoint, json=None):
            return {"unexpected": "shape"}

    monkeypatch.setattr("prime_cli.commands.factory_slurm.APIClient", lambda: _BadPostClient())
    monkeypatch.setattr("prime_cli.commands.factory_slurm.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(
        app,
        ["factory", "slurm", "add-member", "job-123", "dave", "--ssh-key", "key"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)
    assert result.exit_code == 1, result.output
    assert "Error" in output
    assert "ValidationError" not in result.output


def test_slurm_missing_data_field_is_malformed_not_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The documented envelope requires data: an omitted or non-list field is
    # schema drift, never an authoritative empty roster.
    _install(monkeypatch, {})
    result = runner.invoke(app, ["factory", "slurm", "list"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 1, result.output
    assert "malformed response body" in output
    assert "No Slurm deployments found." not in output

    _install(monkeypatch, {"data": "not-a-list"})
    members = runner.invoke(app, ["factory", "slurm", "members", "job-123"], env=TEST_ENV)
    members_output = strip_ansi(members.output)
    assert members.exit_code == 1, members.output
    assert "malformed response body" in members_output
    assert "No members found." not in members_output


def test_slurm_members_id_only_link_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    # A member with linkedUserId but no name/email is still linked
    # (--link-user operates on IDs) — render the ID, never a dash.
    payload = _members_payload(
        [_member("carol", linkedUserId="user-9", linkedUserName=None, linkedUserEmail=None)]
    )
    _install(monkeypatch, payload)

    result = runner.invoke(app, ["factory", "slurm", "members", "job-123"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "user-9" in output


def test_slurm_add_member_non_json_success_body_is_clean_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # HTTP 2xx with a non-JSON body on the POST path: the coarse
    # malformed-response error, never a decoder traceback.
    class _BadPostClient:
        def post(self, endpoint, json=None):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr("prime_cli.commands.factory_slurm.APIClient", lambda: _BadPostClient())
    monkeypatch.setattr("prime_cli.commands.factory_slurm.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(
        app,
        ["factory", "slurm", "add-member", "job-123", "dave", "--ssh-key", "key"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)
    assert result.exit_code == 1, result.output
    assert "malformed response body" in output
    assert "ValueError" not in result.output
    assert "Traceback" not in result.output
