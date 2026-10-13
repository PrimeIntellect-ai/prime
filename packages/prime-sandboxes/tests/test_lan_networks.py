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


def _sync_client(response):
    client = object.__new__(SandboxClient)
    client.client = MagicMock()
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
