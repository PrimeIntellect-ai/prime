from __future__ import annotations

from typing import Any

from prime_cli.api.rl import RLClient, RLRun


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


def test_run_model_allows_dedicated_full_ft_without_lora_fields() -> None:
    run = RLRun.model_validate(
        {
            "id": "full-ft-run",
            "name": "dedicated",
            "userId": "user-1",
            "status": "RUNNING",
            "kind": "DEDICATED_FULL_FT",
            "createdAt": "2026-05-17T00:00:00Z",
            "updatedAt": "2026-05-17T00:00:00Z",
        }
    )

    assert run.kind == "DEDICATED_FULL_FT"
    assert run.rollouts_per_example is None
    assert run.seq_len is None
    assert run.max_steps is None
    assert run.batch_size is None
    assert run.base_model is None


def test_run_model_maps_the_dedicated_runs_prime_cluster() -> None:
    """Dedicated runs answer with primeClusterId (their rftClusterId is
    NULL); the model must carry the field or `prime train get` shows no
    cluster at all."""
    run = RLRun.model_validate(
        {
            "id": "kuvz6ncuc5qzezo3uk3afrur",
            "userId": "user-1",
            "status": "RUNNING",
            "kind": "DEDICATED_FULL_FT",
            "primeClusterId": "ccl9wyl7n0002f2dn0g0x3p0q",
            "rftClusterId": "",
            "createdAt": "2026-05-17T00:00:00Z",
            "updatedAt": "2026-05-17T00:00:00Z",
        }
    )

    assert run.prime_cluster_id == "ccl9wyl7n0002f2dn0g0x3p0q"
    assert run.cluster_id == ""
    assert "prime_cluster_id" in run.model_dump()


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
