"""Optional platform preview contracts for Lab config launches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from prime_cli.api.rl import RLClient
from prime_cli.client import APIClient

PlatformPreviewStatus = Literal["ok", "warning", "unavailable", "error"]


@dataclass(frozen=True)
class PlatformPreviewResult:
    status: PlatformPreviewStatus
    message: str
    warnings: tuple[str, ...] = ()


def preview_lab_config(
    config_kind: str,
    config: dict[str, Any],
    *,
    api_client_factory: Callable[..., Any] = APIClient,
    rl_client_factory: Callable[[Any], Any] = RLClient,
) -> PlatformPreviewResult:
    """Call optional backend preview contracts without requiring them for launch."""

    try:
        if config_kind == "rl":
            response = rl_client_factory(api_client_factory()).preview_run(config)
        else:
            response = api_client_factory().post("/hosted-evaluations/preview", json=config)
    except Exception as exc:
        return PlatformPreviewResult(
            status="unavailable",
            message=f"Platform preview unavailable: {exc}",
        )

    warnings = _preview_warnings(response)
    message = _preview_message(response)
    if warnings:
        return PlatformPreviewResult(status="warning", message=message, warnings=warnings)
    return PlatformPreviewResult(status="ok", message=message, warnings=warnings)


def _preview_message(response: Any) -> str:
    if isinstance(response, dict):
        for key in ("message", "summary", "status"):
            value = response.get(key)
            if value:
                return str(value)
    return "Platform preview passed."


def _preview_warnings(response: Any) -> tuple[str, ...]:
    if not isinstance(response, dict):
        return ()
    raw_warnings = response.get("warnings") or response.get("issues") or ()
    if not isinstance(raw_warnings, list):
        return ()
    return tuple(str(warning) for warning in raw_warnings if warning)
