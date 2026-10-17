"""Tests for the sandbox LAN network SDK surface."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prime_sandboxes.core import APIError
from prime_sandboxes.models import LanNetwork, LanRule
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


RULES = [LanRule(from_="tag:web", to="tag:db", ports=[5432])]


def test_create_lan_network_applies_its_rules_at_creation():
    client = _sync_client(NETWORK)
    network = client.create_lan_network("lab", rules=RULES)
    client.client.request.assert_called_once_with(
        "POST",
        "/sandbox/networks",
        json={"name": "lab", "rules": [{"from": "tag:web", "to": "tag:db", "ports": [5432]}]},
    )
    assert network.rules[0].to == "tag:db"


def test_create_lan_network_deletes_a_network_whose_rules_were_ignored():
    # A platform from before rules on create stores its default rules instead.
    client = _sync_client({**NETWORK, "rules": [{"from": "*", "to": "*"}]})
    with pytest.raises(APIError, match="ignored LAN network net1's rules.*was deleted"):
        client.create_lan_network("lab", rules=RULES)
    client.client.request.assert_called_with("DELETE", "/sandbox/networks/net1")


def test_create_lan_network_ignores_rule_order_and_port_order():
    rules = [
        LanRule(from_="tag:web", to="tag:db", ports=[6379, 5432]),
        LanRule(from_="*", to="tag:log"),
    ]
    stored = [
        {"from": "*", "to": "tag:log"},
        {"from": "tag:web", "to": "tag:db", "ports": [5432, 6379]},
    ]
    client = _sync_client({**NETWORK, "rules": stored})
    assert client.create_lan_network("lab", rules=rules).id == "net1"
    assert client.client.request.call_count == 1  # no delete


def test_a_failed_cleanup_names_the_network_left_behind():
    client = _sync_client({**NETWORK, "rules": [{"from": "*", "to": "*"}]})
    client.client.request.side_effect = [
        {**NETWORK, "rules": [{"from": "*", "to": "*"}]},
        APIError("boom"),
    ]
    with pytest.raises(APIError, match="net1.*deleting it failed"):
        client.create_lan_network("lab", rules=RULES)


@pytest.mark.asyncio
async def test_async_create_lan_network_deletes_a_network_whose_rules_were_ignored():
    client = object.__new__(AsyncSandboxClient)
    client.client = MagicMock()
    client.client.config.team_id = None
    client.client.request = AsyncMock(return_value={**NETWORK, "rules": []})
    with pytest.raises(APIError, match="ignored LAN network net1's rules.*was deleted"):
        await client.create_lan_network("lab", rules=RULES)
    client.client.request.assert_awaited_with("DELETE", "/sandbox/networks/net1")
    # Matching rules come back as created.
    client.client.request = AsyncMock(return_value=NETWORK)
    network = await client.create_lan_network("lab", rules=RULES)
    assert network.id == "net1"


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


def test_create_records_lan_and_returns_immediately():
    client = _sync_client({"id": "sbx1"})
    client._pending_lan_joins = {}
    client.wait_for_creation = MagicMock()
    client._join_lan = MagicMock()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("prime_sandboxes.sandbox.Sandbox.model_validate", lambda r: SimpleNamespace(**r))
        client.create(_create_request(lan_network_id="net1", lan_tags=["web"]))
    payload = client.client.request.call_args.kwargs["json"]
    assert "lan_network_id" not in payload and "lan_tags" not in payload
    assert client._pending_lan_joins == {"sbx1": ("net1", ["web"])}
    client.wait_for_creation.assert_not_called()
    client._join_lan.assert_not_called()


def test_wait_for_creation_joins_pending_lan():
    client = _sync_client(None)
    client._pending_lan_joins = {"sbx1": ("net1", ["web"])}
    client._sandbox_status_batcher = MagicMock()
    client._sandbox_status_batcher.get.return_value = SimpleNamespace(status="RUNNING")
    client._poll_leases = MagicMock()
    client._is_sandbox_reachable = lambda sid: True
    client._join_lan = MagicMock()
    client.wait_for_creation("sbx1")
    client._join_lan.assert_called_once_with("net1", "sbx1", ["web"])
    assert client._pending_lan_joins == {}
    client.wait_for_creation("sbx1")  # already joined: no second join
    client._join_lan.assert_called_once()


def test_failed_join_stays_pending_for_retry():
    client = _sync_client(None)
    client._pending_lan_joins = {"sbx1": ("net1", None)}
    client._join_lan = MagicMock(side_effect=TimeoutError("not synced"))
    with pytest.raises(TimeoutError):
        client._join_pending_lan(["sbx1", "sbx2"])
    assert client._pending_lan_joins == {"sbx1": ("net1", None)}


@pytest.mark.asyncio
async def test_async_join_pending_lan():
    client = object.__new__(AsyncSandboxClient)
    client._pending_lan_joins = {"sbx1": ("net1", None), "sbx2": ("net1", ["db"])}
    client._join_lan = AsyncMock()
    await client._join_pending_lan(["sbx1", "sbx2", "sbx3"])
    assert client._join_lan.await_count == 2
    assert client._pending_lan_joins == {}


def test_lan_tags_require_network():
    with pytest.raises(ValueError, match="lan_tags requires lan_network_id"):
        _create_request(lan_tags=["web"])


def test_join_lan_polls_until_synced(monkeypatch):
    monkeypatch.setattr("prime_sandboxes.sandbox.time.sleep", lambda s: None)
    client = _sync_client({"sandbox_id": "sbx1", "tags": [], "synced": False})
    unsynced = {**NETWORK, "members": [{"sandbox_id": "sbx1", "tags": [], "synced": False}]}
    client.get_lan_network = MagicMock(
        side_effect=[LanNetwork.model_validate(n) for n in (unsynced, NETWORK)]
    )
    client._join_lan("net1", "sbx1", None)
    assert client.get_lan_network.call_count == 2


def test_join_lan_times_out(monkeypatch):
    monkeypatch.setattr("prime_sandboxes.sandbox.time.sleep", lambda s: None)
    monkeypatch.setattr("prime_sandboxes.sandbox._LAN_SYNC_TIMEOUT_SECONDS", -1)
    client = _sync_client({"sandbox_id": "sbx1", "tags": [], "synced": False})
    with pytest.raises(TimeoutError, match="not synced"):
        client._join_lan("net1", "sbx1", None)
