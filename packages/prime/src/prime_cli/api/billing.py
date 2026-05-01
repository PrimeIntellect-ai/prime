"""Billing API client — token usage and cost for an RFT run."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from prime_cli.core import APIClient, APIError


class RunUsageBreakdown(BaseModel):
    tokens: int = 0
    input_tokens: int = Field(0, alias="input_tokens")
    output_tokens: int = Field(0, alias="output_tokens")
    cost_usd: float = Field(0.0, alias="cost_usd")

    model_config = ConfigDict(populate_by_name=True)


class RunPricing(BaseModel):
    training_per_mtok: Optional[float] = Field(None, alias="training_per_mtok")
    inference_input_per_mtok: Optional[float] = Field(None, alias="inference_input_per_mtok")
    inference_output_per_mtok: Optional[float] = Field(None, alias="inference_output_per_mtok")

    model_config = ConfigDict(populate_by_name=True)


class RunUsage(BaseModel):
    run_id: str
    run_name: Optional[str] = None
    base_model: Optional[str] = None
    status: Optional[str] = None
    training: RunUsageBreakdown
    inference: RunUsageBreakdown
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    pricing: RunPricing
    record_count: int = 0


class BillingClient:
    """Thin client for `/api/v1/billing/*` endpoints."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def get_run_usage(self, run_id: str) -> RunUsage:
        """Fetch token + cost totals for a single RFT run."""
        try:
            response = self.client.get(f"/billing/runs/{run_id}/usage")
        except APIError:
            # Let typed APIError subclasses (UnauthorizedError, etc.) propagate
            # — wrapping them strips the type and the caller's ability to
            # branch on auth/payment failures.
            raise
        except Exception as exc:  # noqa: BLE001
            raise APIError(_format_error("Failed to get run usage", exc)) from exc

        try:
            return RunUsage.model_validate(response)
        except ValidationError as exc:
            # Wrap shape drift as APIError so the caller's existing
            # except-APIError branch surfaces it as a clean CLI error
            # instead of an unhandled traceback.
            raise APIError(f"Unexpected /billing/runs/{run_id}/usage response: {exc}") from exc


def _format_error(prefix: str, exc: Exception) -> str:
    response = getattr(exc, "response", None)
    text = getattr(response, "text", None) if response is not None else None
    return f"{prefix}: {text}" if text else f"{prefix}: {exc}"
