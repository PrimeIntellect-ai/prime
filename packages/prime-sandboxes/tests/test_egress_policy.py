"""Tests for the egress policy SDK surface."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from prime_sandboxes.models import (
    CreateSandboxRequest,
    EgressPolicyStatus,
    Sandbox,
    UpdateSandboxRequest,
    validate_egress_lists,
)
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxClient


def _policy_response():
    return {
        "policy": {"allowlist": None, "denylist": []},
        "generation": 2,
        "applied_generation": 2,
        "applied": True,
    }


def _sync_client():
    client = object.__new__(SandboxClient)
    client.client = MagicMock()
    client.client.request.return_value = _policy_response()
    return client


def _async_client():
    client = object.__new__(AsyncSandboxClient)
    client.client = SimpleNamespace(request=AsyncMock(return_value=_policy_response()))
    return client


class TestCreateSandboxRequestNetworkLists:
    def test_removed_network_access_is_gone(self):
        request = CreateSandboxRequest(name="t", docker_image="img")
        assert "network_access" not in type(request).model_fields
        assert "network_access" not in request.model_dump()

    def test_lists_require_vm(self):
        with pytest.raises(ValidationError, match="only supported for"):
            CreateSandboxRequest(
                name="t", docker_image="img", network_allowlist=["api.example.com"]
            )

    def test_vm_allowlist_accepted(self):
        request = CreateSandboxRequest(
            name="t",
            docker_image="img",
            vm=True,
            network_allowlist=["api.example.com", "1.2.3.4", "10.0.0.0/8"],
        )
        assert request.network_allowlist == ["api.example.com", "1.2.3.4", "10.0.0.0/8"]

    def test_mutual_exclusion(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            CreateSandboxRequest(
                name="t",
                docker_image="img",
                vm=True,
                network_allowlist=["a.com"],
                network_denylist=["b.com"],
            )

    def test_serializes_snake_case_and_keeps_empty_list(self):
        request = CreateSandboxRequest(name="t", docker_image="img", vm=True, network_allowlist=[])
        payload = request.model_dump(by_alias=False, exclude_none=True)
        assert payload["network_allowlist"] == []
        assert "network_denylist" not in payload

    @pytest.mark.parametrize(
        "entry",
        [
            "*",
            "https://example.com",
            "example.com:443",
            "example.com?q=1",
            "user@example.com",
            "api.*.example.com",
            "::1",
            "2001:db8::/32",
            "",
        ],
    )
    def test_invalid_entries_rejected(self, entry):
        with pytest.raises(ValidationError):
            CreateSandboxRequest(name="t", docker_image="img", vm=True, network_denylist=[entry])

    def test_rule_count_cap(self):
        entries = [f"h{i}.example.com" for i in range(257)]
        with pytest.raises(ValidationError, match="at most"):
            CreateSandboxRequest(name="t", docker_image="img", vm=True, network_allowlist=entries)


class TestUpdateSandboxRequest:
    def test_network_access_removed(self):
        assert "network_access" not in UpdateSandboxRequest.model_fields


class TestSandboxModel:
    def test_network_access_removed_and_lists_parse(self):
        assert "network_access" not in Sandbox.model_fields
        sandbox = Sandbox.model_validate(
            {
                "id": "sbx-1",
                "name": "t",
                "dockerImage": "img",
                "cpuCores": 1.0,
                "memoryGB": 2.0,
                "diskSizeGB": 10.0,
                "diskMountPath": "/w",
                "gpuCount": 0,
                "vm": True,
                "network_allowlist": ["api.example.com"],
                "status": "RUNNING",
                "timeoutMinutes": 60,
                "createdAt": "2026-07-17T00:00:00Z",
                "updatedAt": "2026-07-17T00:00:00Z",
                # Old servers may still emit the removed boolean; it must be
                # ignored rather than rejected.
                "networkAccess": True,
            }
        )
        assert sandbox.network_allowlist == ["api.example.com"]
        assert sandbox.network_denylist is None


class TestEgressPolicyStatus:
    def test_parses_api_response(self):
        status = EgressPolicyStatus.model_validate(
            {
                "policy": {"allowlist": ["api.example.com"], "denylist": None},
                "generation": 3,
                "applied_generation": 3,
                "applied": True,
            }
        )
        assert status.policy.allowlist == ["api.example.com"]
        assert status.generation == 3
        assert status.applied

    def test_parses_unapplied_generation(self):
        status = EgressPolicyStatus.model_validate(
            {
                "policy": {"allowlist": None, "denylist": []},
                "generation": 0,
                "applied_generation": -1,
                "applied": False,
            }
        )

        assert status.generation == 0
        assert status.applied_generation == -1
        assert not status.applied


class TestSandboxNetworkMethods:
    def test_get_network(self):
        client = _sync_client()

        status = client.get_network("sbx-1")

        client.client.request.assert_called_once_with("GET", "/sandbox/sbx-1/egress-policy")
        assert status.applied

    @pytest.mark.parametrize(
        ("kwargs", "payload"),
        [
            (
                {"allow": ["api.example.com"]},
                {"allowlist": ["api.example.com"], "denylist": None},
            ),
            (
                {"deny": ["ads.example.com"]},
                {"allowlist": None, "denylist": ["ads.example.com"]},
            ),
            ({"allow": ["*"]}, {"allowlist": None, "denylist": []}),
            ({"deny": ["*"]}, {"allowlist": [], "denylist": None}),
        ],
    )
    def test_set_network_maps_explicit_modes(self, kwargs, payload):
        client = _sync_client()

        status = client.set_network("sbx-1", **kwargs)

        client.client.request.assert_called_once_with(
            "PUT", "/sandbox/sbx-1/egress-policy", json=payload
        )
        assert status.applied

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({}, "exactly one"),
            (
                {"allow": ["api.example.com"], "deny": ["ads.example.com"]},
                "exactly one",
            ),
            ({"allow": []}, "at least one destination"),
            ({"deny": []}, "at least one destination"),
            ({"allow": ["*", "api.example.com"]}, "only destination"),
            ({"deny": ["ads.example.com", "*"]}, "only destination"),
        ],
    )
    def test_set_network_rejects_invalid_replacements(self, kwargs, message):
        client = _sync_client()

        with pytest.raises(ValueError, match=message):
            client.set_network("sbx-1", **kwargs)

        client.client.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_methods_use_same_interface(self):
        client = _async_client()

        read_status = await client.get_network("sbx-1")
        write_status = await client.set_network("sbx-1", allow=["api.example.com"])

        assert read_status.applied
        assert write_status.applied
        assert client.client.request.await_args_list[0].args == (
            "GET",
            "/sandbox/sbx-1/egress-policy",
        )
        assert client.client.request.await_args_list[1].args == (
            "PUT",
            "/sandbox/sbx-1/egress-policy",
        )
        assert client.client.request.await_args_list[1].kwargs == {
            "json": {"allowlist": ["api.example.com"], "denylist": None}
        }


class TestValidateEgressLists:
    def test_empty_lists_are_valid(self):
        validate_egress_lists([], None)
        validate_egress_lists(None, [])

    def test_both_none_is_valid(self):
        validate_egress_lists(None, None)
