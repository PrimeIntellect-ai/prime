"""Inference trace-capture API client.

This talks to the platform API (``APIClient``), not the Prime Traces service:
enablement is team policy the platform owns, while the captured traces
themselves are read through ``prime traces``. The platform takes the team as a
``teamId`` param; the traces service takes it as an ``X-Prime-Team-ID`` header.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError

ENDPOINT = "/inference/trace-capture"

# The policy key is (teamId, model, loraId); a base route has no adapter.
BASE_LORA_ID = "__base__"

# Effective state per target, as the platform computes it. The three blocked and
# ineligible states all mean "not capturing" but need different fixes, so none of
# them may collapse into `off`.
STATE_ON = "on"
STATE_OFF = "off"
STATE_BLOCKED_PRIVACY_OPT_OUT = "blocked_privacy_opt_out"
STATE_BLOCKED_STORAGE_FULL = "blocked_storage_full"
STATE_INELIGIBLE = "ineligible"

STATES = (
    STATE_ON,
    STATE_OFF,
    STATE_BLOCKED_PRIVACY_OPT_OUT,
    STATE_BLOCKED_STORAGE_FULL,
    STATE_INELIGIBLE,
)


class TraceCaptureForbiddenError(APIError):
    """The platform refused the call: reads need team membership, writes need ADMIN.

    Its own type because the fix is different from every other failure, and
    ``APIClient`` collapses 403 into a bare ``APIError``.
    """


class TraceCaptureTarget(BaseModel):
    """One ``(model, loraId)`` capture target with the state the platform computed.

    ``enabled`` is the team's stored opt-in alone; ``state`` is that opt-in after
    the operator eligibility, privacy and storage gates, so the two disagree
    whenever capture is on but vetoed.
    """

    model: str = Field(..., description="Inference model ID")
    lora_id: str = Field(BASE_LORA_ID, alias="loraId")
    display_name: Optional[str] = Field(None, alias="displayName")
    state: str = Field(..., description=f"One of: {', '.join(STATES)}")
    enabled: bool = Field(False, description="The team's stored opt-in, before the other gates")
    eligible: bool = Field(True, description="Whether the resolved route is approved for capture")
    sample_rate: Optional[float] = Field(None, alias="sampleRate")
    updated_at: Optional[datetime] = Field(None, alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True)

    @property
    def run_id(self) -> str:
        """The synthetic run captured traces carry: ``inference/<model>[:<lora_id>]``."""
        return synthetic_run_id(self.model, self.lora_id)


def synthetic_run_id(model: str, lora_id: str = BASE_LORA_ID) -> str:
    """The run ID the gateway stamps on traces captured for this target."""
    if lora_id and lora_id != BASE_LORA_ID:
        return f"inference/{model}:{lora_id}"
    return f"inference/{model}"


def _failed(action: str, exc: APIError) -> APIError:
    """One error for the command layer, with a permission refusal kept apart."""
    message = str(exc)
    if message.startswith("HTTP 403"):
        return TraceCaptureForbiddenError(message.removeprefix("HTTP 403: "))
    return APIError(f"Failed to {action}: {message}")


def _target_list(response: Dict[str, Any]) -> List[Any]:
    """The targets out of a listing response, under either accepted key."""
    for key in ("targets", "items"):
        value = response.get(key)
        if isinstance(value, list):
            return value
    return []


class TraceCaptureClient:
    """Client for the platform's inference trace-capture API."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list_targets(self, params: Optional[Dict[str, Any]] = None) -> List[TraceCaptureTarget]:
        """List the team's capture targets, each with its effective state."""
        try:
            response = self.client.get(ENDPOINT, params=params)
        except APIError as e:
            raise _failed("list trace-capture targets", e) from e
        return [TraceCaptureTarget.model_validate(item) for item in _target_list(response)]

    def set_target(
        self,
        *,
        team_id: str,
        model: str,
        enabled: bool,
        lora_id: str = BASE_LORA_ID,
        sample_rate: Optional[float] = None,
    ) -> Optional[TraceCaptureTarget]:
        """Turn capture on or off for one target. Requires team ADMIN.

        Returns the target with its recomputed effective state, or ``None`` when
        the route acknowledges without echoing one back, so an enable that lands
        blocked is never reported as plain success.
        """
        body: Dict[str, Any] = {
            "teamId": team_id,
            "model": model,
            "loraId": lora_id,
            "enabled": enabled,
        }
        if sample_rate is not None:
            body["sampleRate"] = sample_rate
        try:
            # APIClient has no put() helper; request() takes any method.
            response = self.client.request("PUT", ENDPOINT, json=body)
        except APIError as e:
            action = "enable trace capture" if enabled else "disable trace capture"
            raise _failed(action, e) from e
        target = response.get("target") if isinstance(response.get("target"), dict) else response
        if not isinstance(target, dict) or "state" not in target:
            return None
        return TraceCaptureTarget.model_validate(target)
