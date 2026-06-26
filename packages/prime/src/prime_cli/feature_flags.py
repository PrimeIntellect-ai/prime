from __future__ import annotations

from typing import TypeAlias

from prime_cli.core import APIClient, APIError, Config

JsonValue: TypeAlias = bool | int | float | str | None | list["JsonValue"] | dict[str, "JsonValue"]
FeatureFlagDefaults: TypeAlias = dict[str, JsonValue]


class FeatureFlagsClient:
    """Authenticated client for Prime feature flag evaluation."""

    def __init__(self, client: APIClient | None = None, config: Config | None = None) -> None:
        self.client = client or APIClient()
        self.config = config or self.client.config

    def evaluate(self, defaults: FeatureFlagDefaults) -> FeatureFlagDefaults:
        """Evaluate feature flags and fall back per key when the API omits a value."""
        if not defaults:
            return {}

        from prime_cli import __version__

        payload: dict[str, JsonValue] = {
            "flags": defaults,
            "cli_version": __version__,
        }
        if self.config.team_id:
            payload["team_id"] = self.config.team_id

        response = self.client.post("/feature-flags/evaluate", json=payload)
        data = response.get("data")
        if not isinstance(data, dict):
            raise APIError("Feature flag response missing data")

        flags = data.get("flags")
        if not isinstance(flags, dict):
            raise APIError("Feature flag response missing flags")

        return {key: flags.get(key, default) for key, default in defaults.items()}


def evaluate_feature_flags(
    defaults: FeatureFlagDefaults,
    client: APIClient | None = None,
    config: Config | None = None,
) -> FeatureFlagDefaults:
    """Evaluate Prime feature flags, returning defaults if evaluation is unavailable."""
    try:
        return FeatureFlagsClient(client=client, config=config).evaluate(defaults)
    except APIError:
        return defaults.copy()


def is_feature_enabled(
    flag_key: str,
    default: bool = False,
    client: APIClient | None = None,
    config: Config | None = None,
) -> bool:
    """Evaluate a boolean Prime feature flag."""
    value = evaluate_feature_flags({flag_key: default}, client=client, config=config)[flag_key]
    return value is True
