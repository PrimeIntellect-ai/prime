from __future__ import annotations

from typing import Any

from prime_cli import __version__
from prime_cli.core import APIClient, APIError
from prime_cli.feature_flags import (
    FeatureFlagsClient,
    evaluate_feature_flags,
    is_feature_enabled,
)


class DummyConfig:
    def __init__(self, team_id: str | None = None) -> None:
        self.team_id = team_id


class DummyFeatureFlagAPIClient(APIClient):
    def __init__(
        self,
        response: dict[str, Any] | None = None,
        team_id: str | None = None,
        error: APIError | None = None,
    ) -> None:
        self.config = DummyConfig(team_id)
        self.response = response or {"data": {"flags": {}}}
        self.error = error
        self.posts: list[tuple[str, dict[str, Any] | None]] = []

    def post(self, endpoint: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        self.posts.append((endpoint, json))
        if self.error:
            raise self.error
        return self.response


def test_feature_flags_client_evaluates_flags_with_cli_context() -> None:
    client = DummyFeatureFlagAPIClient(
        response={"data": {"flags": {"cli-new-flow": True}}},
        team_id="team-1",
    )

    result = FeatureFlagsClient(client=client).evaluate(
        {"cli-new-flow": False, "copy.variant": "control"}
    )

    assert result == {"cli-new-flow": True, "copy.variant": "control"}
    assert client.posts == [
        (
            "/feature-flags/evaluate",
            {
                "flags": {"cli-new-flow": False, "copy.variant": "control"},
                "cli_version": __version__,
                "team_id": "team-1",
            },
        )
    ]


def test_evaluate_feature_flags_returns_defaults_when_api_unavailable() -> None:
    client = DummyFeatureFlagAPIClient(error=APIError("feature flag service unavailable"))
    defaults = {"cli-new-flow": False}

    result = evaluate_feature_flags(defaults, client=client)

    assert result == defaults
    assert result is not defaults


def test_feature_flags_client_skips_empty_request() -> None:
    client = DummyFeatureFlagAPIClient()

    assert FeatureFlagsClient(client=client).evaluate({}) == {}
    assert client.posts == []


def test_is_feature_enabled_only_accepts_boolean_true() -> None:
    enabled_client = DummyFeatureFlagAPIClient(response={"data": {"flags": {"enabled": True}}})
    string_client = DummyFeatureFlagAPIClient(response={"data": {"flags": {"enabled": "true"}}})

    assert is_feature_enabled("enabled", client=enabled_client) is True
    assert is_feature_enabled("enabled", default=True, client=string_client) is False
