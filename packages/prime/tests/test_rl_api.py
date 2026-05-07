from __future__ import annotations

from typing import Any

from prime_cli.api.rl import RLClient


class FakeAPIClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any] | None]] = []

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.requests.append((endpoint, params))
        return {
            "chartData": {
                "histogramData": [
                    {
                        "binStart": 0.0,
                        "binEnd": 0.2,
                        "count": 1,
                        "range": "0.000-0.200",
                    }
                ]
            },
            "step": 160,
            "unexpected": "raw response detail",
        }


def test_get_distributions_preserves_chart_histogram_data() -> None:
    api_client = FakeAPIClient()
    client = RLClient(api_client)  # type: ignore[arg-type]

    result = client.get_distributions("run-1", distribution_type="rewards", step=160)

    assert api_client.requests == [
        ("/rft/runs/run-1/distributions", {"type": "rewards", "step": 160})
    ]
    assert result == {
        "bins": [
            {
                "binStart": 0.0,
                "binEnd": 0.2,
                "count": 1,
                "range": "0.000-0.200",
            }
        ],
        "step": 160,
    }
