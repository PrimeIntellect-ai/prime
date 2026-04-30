"""Wallet API client — current balance + recent billing rows."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError


class BillingEntry(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    last_billed_at: Optional[datetime] = None
    amount_usd: float
    currency: str
    resource_type: str
    resource_id: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class Wallet(BaseModel):
    wallet_id: str
    team_id: Optional[str] = None
    balance_usd: float = 0.0
    currency: str
    total_billings: int = 0
    recent_billings: List[BillingEntry] = Field(default_factory=list)


class WalletClient:
    """Thin client for `/api/v1/billing/wallet`."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def get(
        self,
        limit: int = 20,
        offset: int = 0,
        team_id: Optional[str] = None,
    ) -> Wallet:
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if team_id:
            params["teamId"] = team_id
        try:
            response = self.client.get("/billing/wallet", params=params)
        except APIError:
            # Let typed APIError subclasses (UnauthorizedError, etc.) propagate
            # — wrapping them strips the type and the caller's ability to
            # branch on auth/payment failures.
            raise
        except Exception as exc:  # noqa: BLE001
            raise APIError(_format_error("Failed to get wallet", exc)) from exc
        return Wallet.model_validate(response)


def _format_error(prefix: str, exc: Exception) -> str:
    response = getattr(exc, "response", None)
    text = getattr(response, "text", None) if response is not None else None
    return f"{prefix}: {text}" if text else f"{prefix}: {exc}"
