"""Tests for the dev-sandboxes pydantic models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from prime_dev_sandboxes.models import (
    ACTIVE_STATES,
    CAPPED_STATES,
    TERMINAL_OPERATION_STATES,
    CreateEnvironmentRequest,
    DevEnvironment,
    EnvironmentListResponse,
    EnvironmentState,
    Operation,
    OperationKind,
    OperationState,
    Template,
)


def test_create_request_defaults():
    request = CreateEnvironmentRequest(template="platform-dev", ref="origin/develop")

    assert request.ttl_minutes == 120
    assert request.team_id is None


def test_environment_state_enum_matches_contract():
    assert [s.value for s in EnvironmentState] == [
        "ALLOCATING",
        "PREPARING",
        "STARTING",
        "VERIFYING",
        "READY",
        "FAILED",
        "DELETING",
        "DELETED",
    ]


def test_operation_states_and_kinds():
    assert [s.value for s in OperationState] == ["running", "succeeded", "failed", "interrupted"]
    assert [k.value for k in OperationKind] == ["create", "exec", "delete"]
    assert TERMINAL_OPERATION_STATES == frozenset(
        {OperationState.SUCCEEDED, OperationState.FAILED, OperationState.INTERRUPTED}
    )


def test_active_and_capped_states():
    # Capped states drive the per-user/per-team allocation limits.
    assert EnvironmentState.DELETING in ACTIVE_STATES
    assert EnvironmentState.DELETING not in CAPPED_STATES
    assert EnvironmentState.FAILED not in CAPPED_STATES
    assert EnvironmentState.READY in CAPPED_STATES


def test_environment_parsing():
    env = DevEnvironment.model_validate(
        {
            "id": "dev-env-abc123",
            "state": "READY",
            "template": "platform-dev",
            "ref": "https://github.com/PrimeIntellect-ai/prime/pull/5800",
            "resolved_sha": "24e7e3eb",
            "ttl_minutes": 120,
            "urls": {"frontend": "https://frontend.example.com"},
            "expiry": "2026-10-14T12:00:00+00:00",
            "created_at": "2026-10-14T10:00:00+00:00",
            "some_future_field": 1,
        }
    )

    assert env.is_ready and env.is_active
    assert env.resolved_sha == "24e7e3eb"
    assert env.urls["frontend"] == "https://frontend.example.com"
    assert env.expiry == datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
    # Additive growth must not break an older SDK.
    assert env.model_dump()["some_future_field"] == 1


def test_environment_state_validation_rejects_unknown_state():
    with pytest.raises(ValidationError):
        DevEnvironment.model_validate(
            {"id": "dev-env-1", "state": "NOT_A_STATE", "template": "platform-dev", "ref": "main"}
        )


def test_operation_parsing():
    op = Operation.model_validate(
        {
            "operation_id": "op-1",
            "kind": "create",
            "state": "running",
            "stage": "VERIFYING",
            "progress": {"phase": 3, "total": 5},
            "diagnostics": ["dockerd up", "compose up"],
            "result": {},
        }
    )

    assert not op.is_terminal
    assert op.kind is OperationKind.CREATE
    assert op.state is OperationState.RUNNING

    done = op.model_validate(
        {
            "operation_id": "op-1",
            "kind": "create",
            "state": "succeeded",
            "error": None,
            "result": {"environment_id": "dev-env-1"},
        }
    )
    assert done.is_terminal
    assert done.error is None


def test_operation_error_detail():
    op = Operation.model_validate(
        {
            "operation_id": "op-2",
            "kind": "exec",
            "state": "failed",
            "error": {
                "code": "HEALTH_FAILED",
                "stage": "readiness",
                "message": "migrations did not complete",
                "retryable": False,
                "owned_resource_ids": ["dev-env-1"],
            },
        }
    )

    assert op.is_terminal
    assert op.error is not None
    assert op.error.group == "readiness"
    assert op.error.owned_resource_ids == ["dev-env-1"]


def test_list_response_defaults():
    page = EnvironmentListResponse.model_validate({})
    assert page.environments == []
    assert page.total == 0
    assert page.has_next is False


def test_template_parsing():
    template = Template.model_validate(
        {
            "id": "platform-dev",
            "image": "prime/prime-engineering/platform-dev:v3",
            "cpu": 8,
            "memory_gb": 32,
            "disk_gb": 60,
            "boot": "baked_image",
            "version": "3",
            "maintainer": "prime-engineering",
        }
    )

    assert template.id == "platform-dev"
    assert template.cpu == 8
