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


def test_slurm_create_validates_nodes_required(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)

    # Missing --nodes
    result = runner.invoke(
        app,
        ["factory", "slurm", "create", "pc-1", "--name", "test-cluster"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)
    assert result.exit_code != 0, result.output
    assert "--nodes" in output.lower() or "missing" in output.lower()


def test_slurm_create_validates_positive_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)

    result = runner.invoke(
        app,
        ["factory", "slurm", "create", "pc-1", "--name", "test", "--nodes", "0"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)
    assert result.exit_code == 1, result.output
    assert "positive" in output

    result = runner.invoke(
        app,
        ["factory", "slurm", "create", "pc-1", "--name", "test", "--nodes", "-1"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)
    assert result.exit_code == 1, result.output
    assert "positive" in output


def test_slurm_create_with_default_platform_user(monkeypatch: pytest.MonkeyPatch) -> None:
    created_response = {
        "id": "job-456",
        "primeClusterId": "pc-1",
        "status": "PENDING",
        "createdAt": "2026-10-29T18:00:00Z",
    }

    class _CreateClient:
        def __init__(self):
            self.post_calls = []

        def post(self, endpoint, json=None, timeout=None):
            self.post_calls.append({"endpoint": endpoint, "json": json})
            if "/deploy" in endpoint:
                return created_response
            return {}

        def get(self, endpoint, params=None, timeout=None):
            # Mock whoami
            if "/user/whoami" in endpoint:
                return {"data": {"id": "user-123", "slug": "alice", "email": "alice@example.com"}}
            return _clusters_payload([])

    client = _CreateClient()
    monkeypatch.setattr("prime_cli.commands.factory_slurm.APIClient", lambda: client)
    monkeypatch.setattr("prime_cli.commands.factory_slurm.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    # Mock SSH key reading
    def mock_read_ssh_keys(key_file=None):
        return ["ssh-ed25519 AAAA alice-key"]

    monkeypatch.setattr("prime_cli.commands.factory_slurm._read_ssh_keys", mock_read_ssh_keys)

    result = runner.invoke(
        app,
        ["factory", "slurm", "create", "pc-1", "--name", "new-cluster", "--nodes", "2", "--yes"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "new-cluster" in output and "created" in output
    assert "job-456" in output
    assert len(client.post_calls) == 1
    assert "/slurm-clusters/team-123/deploy" in client.post_calls[0]["endpoint"]
    body = client.post_calls[0]["json"]
    assert body["primeClusterId"] == "pc-1"
    assert body["slurmClusterName"] == "new-cluster"
    assert body["numNodes"] == 2
    assert "members" in body
    assert len(body["members"]) == 1
    assert body["members"][0]["username"] == "alice"
    assert body["members"][0]["sshAuthorizedKeys"] == ["ssh-ed25519 AAAA alice-key"]
    assert body["members"][0]["sudo"] is False
    assert body["members"][0]["linkedUserId"] == "user-123"
    assert body["containersEnabled"] is False


def test_slurm_create_with_multiple_members_separate_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that each member gets their own SSH keys, not shared keys."""
    created_response = {
        "id": "job-789",
        "primeClusterId": "pc-2",
        "status": "PENDING",
        "createdAt": "2026-10-29T18:00:00Z",
    }

    class _CreateClient:
        def __init__(self):
            self.post_calls = []

        def post(self, endpoint, json=None, timeout=None):
            self.post_calls.append({"endpoint": endpoint, "json": json})
            if "/deploy" in endpoint:
                return created_response
            return {}

        def get(self, endpoint, params=None, timeout=None):
            if "/user/whoami" in endpoint:
                return {"data": {"id": "user-123", "slug": "alice", "email": "alice@example.com"}}
            return _clusters_payload([])

    client = _CreateClient()
    monkeypatch.setattr("prime_cli.commands.factory_slurm.APIClient", lambda: client)
    monkeypatch.setattr("prime_cli.commands.factory_slurm.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    # Mock SSH key reading - return different keys per file
    def mock_read_ssh_keys(key_file=None):
        if key_file == "~/.ssh/alice.pub":
            return ["ssh-ed25519 AAAA alice-key"]
        elif key_file == "~/.ssh/bob.pub":
            return ["ssh-ed25519 BBBB bob-key"]
        else:
            return ["ssh-ed25519 AAAA default-key"]

    monkeypatch.setattr("prime_cli.commands.factory_slurm._read_ssh_keys", mock_read_ssh_keys)

    result = runner.invoke(
        app,
        [
            "factory",
            "slurm",
            "create",
            "pc-2",
            "--name",
            "multi-cluster",
            "--nodes",
            "4",
            "--member",
            "alice:~/.ssh/alice.pub",
            "--member",
            "bob:~/.ssh/bob.pub",
            "--sudo",
            "--yes",
        ],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "multi-cluster" in output
    assert len(client.post_calls) == 1
    body = client.post_calls[0]["json"]
    assert body["numNodes"] == 4
    assert len(body["members"]) == 2

    # Verify alice (platform user) has her own key and sudo
    alice_member = next(m for m in body["members"] if m["username"] == "alice")
    assert alice_member["sshAuthorizedKeys"] == ["ssh-ed25519 AAAA alice-key"]
    assert alice_member["sudo"] is True  # Platform user gets sudo
    assert alice_member["linkedUserId"] == "user-123"

    # Verify bob has his own separate key and no sudo
    bob_member = next(m for m in body["members"] if m["username"] == "bob")
    assert bob_member["sshAuthorizedKeys"] == ["ssh-ed25519 BBBB bob-key"]
    assert bob_member["sudo"] is False
    assert "linkedUserId" not in bob_member


def test_slurm_create_bare_member_requires_platform_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that bare --member (without :KEYFILE) only works for platform user."""
    created_response = {
        "id": "job-456",
        "primeClusterId": "pc-1",
        "status": "PENDING",
        "createdAt": "2026-10-29T18:00:00Z",
    }

    class _CreateClient:
        def __init__(self):
            self.post_calls = []

        def post(self, endpoint, json=None, timeout=None):
            self.post_calls.append({"endpoint": endpoint, "json": json})
            if "/deploy" in endpoint:
                return created_response
            return {}

        def get(self, endpoint, params=None, timeout=None):
            if "/user/whoami" in endpoint:
                return {"data": {"id": "user-123", "slug": "alice", "email": "alice@example.com"}}
            return _clusters_payload([])

    client = _CreateClient()
    monkeypatch.setattr("prime_cli.commands.factory_slurm.APIClient", lambda: client)
    monkeypatch.setattr("prime_cli.commands.factory_slurm.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    # Mock SSH key reading
    def mock_read_ssh_keys(key_file=None):
        return ["ssh-ed25519 AAAA test-key"]

    monkeypatch.setattr("prime_cli.commands.factory_slurm._read_ssh_keys", mock_read_ssh_keys)

    # Should fail for non-platform user
    result = runner.invoke(
        app,
        ["factory", "slurm", "create", "pc-1", "--name", "test", "--nodes", "1", "--member", "bob"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)
    assert result.exit_code == 1, result.output
    assert "bob" in output and "requires a key file" in output.lower()

    # Should work for platform user
    result = runner.invoke(
        app,
        [
            "factory",
            "slurm",
            "create",
            "pc-1",
            "--name",
            "test",
            "--nodes",
            "1",
            "--member",
            "alice",
            "--yes",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output


def test_slurm_create_confirms_and_cancels(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _install(monkeypatch)

    def mock_get_platform_user():
        return {"id": "user-123", "username": "testuser"}

    monkeypatch.setattr(
        "prime_cli.commands.factory_slurm._get_platform_user", mock_get_platform_user
    )

    # Mock SSH key reading
    def mock_read_ssh_keys(key_file=None):
        return ["ssh-ed25519 AAAA test-key"]

    monkeypatch.setattr("prime_cli.commands.factory_slurm._read_ssh_keys", mock_read_ssh_keys)

    result = runner.invoke(
        app,
        ["factory", "slurm", "create", "pc-1", "--name", "test", "--nodes", "2"],
        env=TEST_ENV,
        input="n\n",
    )
    output = strip_ansi(result.output)
    assert result.exit_code == 0, result.output
    assert "Deployment cancelled" in output
    assert len(dummy.post_calls) == 0


def test_slurm_create_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    created_response = {
        "id": "job-999",
        "primeClusterId": "pc-1",
        "status": "PENDING",
        "createdAt": "2026-10-29T18:00:00Z",
    }

    class _CreateClient:
        def post(self, endpoint, json=None, timeout=None):
            if "/deploy" in endpoint:
                return created_response
            return {}

        def get(self, endpoint, params=None, timeout=None):
            if "/user/whoami" in endpoint:
                return {"data": {"id": "user-123", "slug": "testuser", "email": "test@example.com"}}
            return _clusters_payload([])

    monkeypatch.setattr("prime_cli.commands.factory_slurm.APIClient", lambda: _CreateClient())
    monkeypatch.setattr("prime_cli.commands.factory_slurm.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    # Mock SSH key reading
    def mock_read_ssh_keys(key_file=None):
        return ["ssh-ed25519 AAAA test-key"]

    monkeypatch.setattr("prime_cli.commands.factory_slurm._read_ssh_keys", mock_read_ssh_keys)

    result = runner.invoke(
        app,
        [
            "factory",
            "slurm",
            "create",
            "pc-1",
            "--name",
            "json-cluster",
            "--nodes",
            "1",
            "--yes",
            "--json",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert parsed["id"] == "job-999"
    assert parsed["primeClusterId"] == "pc-1"
    assert parsed["status"] == "PENDING"


def test_slurm_create_malformed_response_is_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadPostClient:
        def post(self, endpoint, json=None, timeout=None):
            if "/deploy" in endpoint:
                return {"unexpected": "shape"}  # Missing 'id'
            return {}

        def get(self, endpoint, params=None, timeout=None):
            if "/user/whoami" in endpoint:
                return {"data": {"id": "user-123", "slug": "testuser", "email": "test@example.com"}}
            return {"data": []}

    monkeypatch.setattr("prime_cli.commands.factory_slurm.APIClient", lambda: _BadPostClient())
    monkeypatch.setattr("prime_cli.commands.factory_slurm.Config", lambda: _StubConfig("team-123"))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    # Mock SSH key reading
    def mock_read_ssh_keys(key_file=None):
        return ["ssh-ed25519 AAAA test-key"]

    monkeypatch.setattr("prime_cli.commands.factory_slurm._read_ssh_keys", mock_read_ssh_keys)

    result = runner.invoke(
        app,
        ["factory", "slurm", "create", "pc-1", "--name", "test", "--nodes", "1", "--yes"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)
    assert result.exit_code == 1, result.output
    assert "malformed response body" in output
    assert "ValidationError" not in result.output


def test_slurm_create_validates_username(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)

    def mock_get_platform_user():
        return {"id": "user-123", "username": "alice"}

    monkeypatch.setattr(
        "prime_cli.commands.factory_slurm._get_platform_user", mock_get_platform_user
    )
    monkeypatch.setattr(
        "prime_cli.commands.factory_slurm._read_ssh_keys", lambda key_file=None: ["key"]
    )

    # Invalid: starts with number (provide keyfile to get past member parsing)
    result = runner.invoke(
        app,
        [
            "factory",
            "slurm",
            "create",
            "pc-1",
            "--name",
            "test",
            "--nodes",
            "1",
            "--member",
            "1user:~/.ssh/key.pub",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "Invalid username" in strip_ansi(result.output)

    # Invalid: uppercase (provide keyfile to get past member parsing)
    result = runner.invoke(
        app,
        [
            "factory",
            "slurm",
            "create",
            "pc-1",
            "--name",
            "test",
            "--nodes",
            "1",
            "--member",
            "Alice:~/.ssh/key.pub",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "Invalid username" in strip_ansi(result.output)

    # Invalid: too short (provide keyfile to get past member parsing)
    result = runner.invoke(
        app,
        [
            "factory",
            "slurm",
            "create",
            "pc-1",
            "--name",
            "test",
            "--nodes",
            "1",
            "--member",
            "a:~/.ssh/key.pub",
        ],
        env=TEST_ENV,
    )
    assert result.exit_code == 1, result.output
    assert "Invalid username" in strip_ansi(result.output)


def test_slurm_create_no_ssh_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)

    def mock_get_platform_user():
        return {"id": "user-123", "username": "testuser"}

    monkeypatch.setattr(
        "prime_cli.commands.factory_slurm._get_platform_user", mock_get_platform_user
    )
    monkeypatch.setattr("prime_cli.commands.factory_slurm._read_ssh_keys", lambda key_file=None: [])

    result = runner.invoke(
        app,
        ["factory", "slurm", "create", "pc-1", "--name", "test", "--nodes", "1"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)
    assert result.exit_code == 1, result.output
    assert "No SSH keys found" in output
