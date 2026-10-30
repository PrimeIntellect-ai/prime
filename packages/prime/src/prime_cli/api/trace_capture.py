"""Inference trace-capture API client.

Enablement is team policy on the platform API; the captured traces are read
through ``prime traces`` under each route's ``run_id``.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from prime_cli.core import APIClient, APIError, NotFoundError, ServerError

ENDPOINT = "/inference/trace-capture"
BASE_LORA_ID = "__base__"

InactiveReason = Literal["privacy_opt_out", "storage_limit", "route_unavailable"]


class TraceCaptureForbiddenError(APIError):
    """403: writing capture policy needs team ADMIN."""


class TraceCaptureUnavailableError(APIError):
    """404: the feature is off, the caller is not a member, or the route is ineligible."""


class TraceCaptureRoute(BaseModel):
    """``enabled`` is the team's choice; ``active`` means calls are captured now."""

    model: str
    lora_id: str = BASE_LORA_ID
    name: Optional[str] = None
    run_id: str
    enabled: bool
    active: bool
    inactive_reason: Optional[InactiveReason] = None


def _failed(action: str, exc: APIError) -> APIError:
    if isinstance(exc, NotFoundError):
        return TraceCaptureUnavailableError(str(exc))
    if str(exc).startswith("HTTP 403"):
        return TraceCaptureForbiddenError(str(exc))
    if isinstance(exc, ServerError) and str(exc).startswith("HTTP 503"):
        return APIError(f"Failed to {action}: inference is unreachable, try again later")
    return APIError(f"Failed to {action}: {exc}")


class TraceCaptureClient:
    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list_routes(self, team_id: str) -> List[TraceCaptureRoute]:
        try:
            response = self.client.get(ENDPOINT, params={"teamId": team_id})
        except APIError as e:
            raise _failed("list trace-capture routes", e) from e
        return [TraceCaptureRoute.model_validate(r) for r in response.get("routes", [])]

    def set_route(
        self, team_id: str, model: str, enabled: bool, lora_id: str = BASE_LORA_ID
    ) -> TraceCaptureRoute:
        body: Dict[str, Any] = {"model": model, "lora_id": lora_id, "enabled": enabled}
        try:
            response = self.client.request("PUT", ENDPOINT, params={"teamId": team_id}, json=body)
        except APIError as e:
            raise _failed("enable trace capture" if enabled else "disable trace capture", e) from e
        return TraceCaptureRoute.model_validate(response)
