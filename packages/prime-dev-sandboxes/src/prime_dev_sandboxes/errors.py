"""Structured error taxonomy for dev sandboxes.

Ported from the dev-sandbox skill (five groups, nineteen codes) so client,
CLI, and backend share one vocabulary. Every failure surfaced by the API is
a structured error carrying: code, stage, retryable, message, provider_code,
operation_id, owned_resource_ids, and a diagnostic_path. Retryability is
provider evidence, not just group membership: the defaults below are only a
fallback for callers that lack provider evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .core.client import APIStatusError

GROUPS: dict[str, tuple[str, ...]] = {
    "admission": (
        "INVALID_INPUT",
        "TASK_CONFLICT",
        "LIMIT_REACHED",
        "AUTH_FAILED",
        "REF_NOT_FOUND",
    ),
    "template": (
        "TEMPLATE_INCOMPATIBLE",
        "TEMPLATE_UNAVAILABLE",
    ),
    "infrastructure": (
        "CAPACITY_UNAVAILABLE",
        "PROVISION_TIMEOUT",
        "PROVISION_FAILED",
        "DISK_LOW",
        "DEPENDENCY_UNAVAILABLE",
    ),
    "readiness": (
        "SECRET_STAGE_FAILED",
        "MIGRATION_FAILED",
        "HEALTH_FAILED",
        "TUNNEL_FAILED",
        "READY_TIMEOUT",
    ),
    "lifecycle": (
        "LEASE_EXPIRED",
        "EXEC_TIMEOUT",
        "CLEANUP_INCOMPLETE",
    ),
}

ALL_CODES: frozenset[str] = frozenset(c for codes in GROUPS.values() for c in codes)

# Group-level retryability fallbacks. Provider evidence always wins.
_RETRYABLE_DEFAULT: frozenset[str] = frozenset(
    {"CAPACITY_UNAVAILABLE", "PROVISION_TIMEOUT", "DEPENDENCY_UNAVAILABLE", "TUNNEL_FAILED"}
)


def group_of(code: str) -> str:
    for group, codes in GROUPS.items():
        if code in codes:
            return group
    raise ValueError(f"unknown error code: {code}")


@dataclass
class DevSandboxError(Exception):
    """Structured failure returned by the dev-sandboxes API."""

    code: str
    stage: str
    message: str
    retryable: bool = False
    provider_code: Optional[str] = None
    operation_id: Optional[str] = None
    owned_resource_ids: Sequence[str] = field(default_factory=list)
    diagnostic_path: Optional[str] = None
    provider_evidence: bool = False  # True when retryable came from provider signals

    def __post_init__(self) -> None:
        if self.code not in ALL_CODES:
            raise ValueError(f"code {self.code!r} not in taxonomy")
        if not self.provider_evidence:
            self.retryable = self.retryable or self.code in _RETRYABLE_DEFAULT

    @property
    def group(self) -> str:
        return group_of(self.code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "group": self.group,
            "stage": self.stage,
            "retryable": self.retryable,
            "message": self.message,
            "provider_code": self.provider_code,
            "operation_id": self.operation_id,
            "owned_resource_ids": list(self.owned_resource_ids),
            "diagnostic_path": self.diagnostic_path,
            "provider_evidence": self.provider_evidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DevSandboxError":
        return cls(
            code=d["code"],
            stage=d.get("stage", ""),
            message=d.get("message", ""),
            retryable=bool(d.get("retryable", False)),
            provider_code=d.get("provider_code"),
            operation_id=d.get("operation_id"),
            owned_resource_ids=list(d.get("owned_resource_ids", [])),
            diagnostic_path=d.get("diagnostic_path"),
            provider_evidence=bool(d.get("provider_evidence", True)),
        )

    @classmethod
    def from_api_error(cls, error: APIStatusError) -> "DevSandboxError":
        """Build from an HTTP error response carrying a structured error body.

        Accepts both the structured ``{"error": {...}}`` shape and the flat
        ``{"code": ..., "stage": ...}`` shape. Responses without a taxonomy
        code fall back to the transport error's status and text.
        """
        payload = error.payload or {}
        body = payload.get("error") if isinstance(payload.get("error"), dict) else payload
        if isinstance(body, dict) and body.get("code") in ALL_CODES:
            return cls.from_dict(body)
        fallback_code = "AUTH_FAILED" if error.status_code == 401 else "INVALID_INPUT"
        message = str(error)
        if isinstance(body, dict) and "detail" in body:
            message = str(body["detail"])
        return cls(
            code=fallback_code,
            stage="api",
            message=message,
        )

    def __str__(self) -> str:  # pragma: no cover - display only
        bits = [f"[{self.code}/{self.group}] stage={self.stage} retryable={self.retryable}"]
        if self.provider_code:
            bits.append(f"provider_code={self.provider_code}")
        if self.operation_id:
            bits.append(f"operation={self.operation_id}")
        if self.owned_resource_ids:
            bits.append(f"owned={list(self.owned_resource_ids)}")
        bits.append(self.message)
        if self.diagnostic_path:
            bits.append(f"diagnostics: {self.diagnostic_path}")
        return " ".join(bits)
