"""Tests for the sandbox LAN network SDK surface."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prime_sandboxes.models import LanRule
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxClient

NETWORK = {
    "id": "net1",
    "name": "lab",
    "team_id": None,
    "rules": [{"from": "tag:web", "to": "tag:db", "ports": [5432]}],
    "version": 2,
    "synced": True,
    "created_at": "2026-10-13T00:00:00Z",
    "members": [{"sandbox_id": "sbx1", "tags": ["web"], "synced": True}],
}


def _sync_client(response, team_id=None):
    client = object.__new__(SandboxClient)
    client.client = MagicMock()
    client.client.config.team_id = team_id
    client.client.request.return_value = response
    return client


def test_set_lan_rules_sends_wire_names():
    client = _sync_client(NETWORK)
    network = client.set_lan_rules(
        "net1", [LanRule(from_="tag:web", to="tag:db", ports=[5432]), LanRule(from_="*", to="*")]
    )
    client.client.request.assert_called_once_with(
        "PUT",
        "/sandbox/networks/net1/rules",
        json={
            "rules": [
                {"from": "tag:web", "to": "tag:db", "ports": [5432]},
                {"from": "*", "to": "*"},
            ]
        },
    )
    assert network.rules[0].from_ == "tag:web"
    assert network.members[0].sandbox_id == "sbx1"


def test_list_lan_networks_unwraps():
    client = _sync_client({"networks": [NETWORK]})
    assert [n.id for n in client.list_lan_networks()] == ["net1"]
    client.client.request.assert_called_once_with("GET", "/sandbox/networks", params=None)


def test_lan_networks_default_to_config_team():
    client = _sync_client(NETWORK, team_id="team1")
    client.create_lan_network("lab")
    client.client.request.assert_called_once_with(
        "POST", "/sandbox/networks", json={"name": "lab", "team_id": "team1"}
    )
    client.client.request.return_value = {"networks": []}
    client.list_lan_networks()
    client.client.request.assert_called_with(
        "GET", "/sandbox/networks", params={"team_id": "team1"}
    )


@pytest.mark.asyncio
async def test_async_add_lan_member():
    client = object.__new__(AsyncSandboxClient)
    client.client = SimpleNamespace(
        request=AsyncMock(return_value={"sandbox_id": "sbx1", "tags": ["db"], "synced": False})
    )
    member = await client.add_lan_member("net1", "sbx1", ["db"])
    client.client.request.assert_awaited_once_with(
        "PUT", "/sandbox/networks/net1/members/sbx1", json={"tags": ["db"]}
    )
    assert member.tags == ["db"] and not member.synced


def _create_request(**kw):
    from prime_sandboxes.models import CreateSandboxRequest

    return CreateSandboxRequest(name="a", docker_image="python:3.11", **kw)


def test_create_joins_lan_after_running():
    client = _sync_client({"id": "sbx1"})
    calls = []
    client.wait_for_creation = lambda sid: calls.append(("wait", sid))
    client.add_lan_member = lambda *a: calls.append(("join", *a))
    client.get = lambda sid: calls.append(("get", sid)) or "running-sandbox"
    client.delete = MagicMock()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("prime_sandboxes.sandbox.Sandbox.model_validate", lambda r: SimpleNamespace(**r))
        result = client.create(_create_request(lan_network_id="net1", lan_tags=["web"]))
    payload = client.client.request.call_args.kwargs["json"]
    assert "lan_network_id" not in payload and "lan_tags" not in payload
    assert calls == [("wait", "sbx1"), ("join", "net1", "sbx1", ["web"]), ("get", "sbx1")]
    assert result == "running-sandbox"
    client.delete.assert_not_called()


def test_create_deletes_sandbox_when_join_fails():
    client = _sync_client({"id": "sbx1"})
    client.wait_for_creation = lambda sid: None
    client.add_lan_member = MagicMock(side_effect=RuntimeError("409"))
    client.delete = MagicMock(side_effect=RuntimeError("delete failed too"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("prime_sandboxes.sandbox.Sandbox.model_validate", lambda r: SimpleNamespace(**r))
        with pytest.raises(RuntimeError, match="409"):
            client.create(_create_request(lan_network_id="net1"))
    client.delete.assert_called_once_with("sbx1")


def test_lan_tags_require_network():
    with pytest.raises(ValueError, match="lan_tags requires lan_network_id"):
        _create_request(lan_tags=["web"])
