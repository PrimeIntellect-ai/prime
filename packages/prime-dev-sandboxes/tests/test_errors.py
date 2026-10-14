"""Tests for the structured error taxonomy (5 groups / 19 codes)."""

import pytest

from prime_dev_sandboxes.core.client import APIStatusError
from prime_dev_sandboxes.errors import (
    ALL_CODES,
    GROUPS,
    DevSandboxError,
    group_of,
)


def test_taxonomy_shape():
    # Ported verbatim from the skill's errors.py: five groups, twenty codes.
    # (The contract prose says "19"; the skill source of truth has 20:
    # admission 5, template 2, infrastructure 5, readiness 5, lifecycle 3.)
    assert len(GROUPS) == 5
    assert len(ALL_CODES) == 20
    assert set(GROUPS) == {"admission", "template", "infrastructure", "readiness", "lifecycle"}
    assert group_of("TASK_CONFLICT") == "admission"
    assert group_of("TEMPLATE_UNAVAILABLE") == "template"
    assert group_of("CAPACITY_UNAVAILABLE") == "infrastructure"
    assert group_of("READY_TIMEOUT") == "readiness"
    assert group_of("EXEC_TIMEOUT") == "lifecycle"


def test_unknown_code_rejected():
    with pytest.raises(ValueError):
        DevSandboxError(code="NOT_A_CODE", stage="api", message="nope")


def test_retryable_defaults_apply_without_provider_evidence():
    error = DevSandboxError(code="CAPACITY_UNAVAILABLE", stage="allocate", message="no capacity")
    assert error.retryable is True

    fatal = DevSandboxError(code="INVALID_INPUT", stage="admission", message="bad ref")
    assert fatal.retryable is False


def test_provider_evidence_wins():
    error = DevSandboxError(
        code="CAPACITY_UNAVAILABLE",
        stage="allocate",
        message="provider says no",
        retryable=False,
        provider_evidence=True,
    )
    assert error.retryable is False


def test_from_dict_roundtrip():
    error = DevSandboxError.from_dict(
        {
            "code": "TASK_CONFLICT",
            "stage": "admission",
            "message": "active env for same (team, user, ref) with different sha",
            "operation_id": "op-9",
            "owned_resource_ids": ["dev-env-1"],
        }
    )
    assert error.code == "TASK_CONFLICT"
    assert error.group == "admission"
    assert error.operation_id == "op-9"
    assert error.to_dict()["group"] == "admission"


def test_from_api_error_with_structured_body():
    api_error = APIStatusError(
        "HTTP 409: conflict",
        409,
        {
            "error": {
                "code": "TASK_CONFLICT",
                "stage": "admission",
                "message": "conflicting active environment",
                "operation_id": "op-1",
            }
        },
    )

    error = DevSandboxError.from_api_error(api_error)
    assert error.code == "TASK_CONFLICT"
    assert error.group == "admission"
    assert error.operation_id == "op-1"
    assert error.retryable is False


def test_from_api_error_with_fastapi_detail_falls_back():
    api_error = APIStatusError("HTTP 404: not found", 404, {"detail": "No such environment"})

    error = DevSandboxError.from_api_error(api_error)
    assert error.code == "INVALID_INPUT"
    assert "No such environment" in error.message


def test_from_api_error_maps_unauthorized():
    api_error = APIStatusError("HTTP 401: unauthorized", 401, {})
    error = DevSandboxError.from_api_error(api_error)
    assert error.code == "AUTH_FAILED"
