"""Tests for `prime factory slurm` (the deployment roster surface)."""

import json
from typing import Any, Dict, List, Optional

import pytest
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


class _StubConfig:
    def __init__(self, team_id: Optional[str]) -> None:
        self._team_id = team_id

    @property
    def team_id(self) -> Optional[str]:
        return self._team_id


class _DummyAPIClient:
    def __init__(self, get_payload: Optional[Dict[str, Any]] = None) -> None:
        self._get_payload = get_payload
        self.calls: List[Dict[str, Any]] = []
        self.post_calls: List[Dict[str, Any]] = []

    def get(self, endpoint, params=None, timeout=None):
        self.calls.append({"endpoint": endpoint, "params": params})
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
) -> _DummyAPIClient:
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    dummy = _DummyAPIClient(get_payload=get_payload)
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


def test_slurm_get_never_invents_connect_info(monkeypatch: pytest.MonkeyPatch) -> None:
    # The roster API carries no connect fields: the detail view renders
    # only what the API provides — never ssh_host, ssh_port, or
    # connectable claims.
    _install(monkeypatch, _clusters_payload())
    result = runner.invoke(app, ["factory", "slurm", "get", "job-123"], env=TEST_ENV)
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "ssh" not in output.lower()
    assert "connectable" not in output.lower()


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
