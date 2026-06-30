from __future__ import annotations

from typing import Any

from prime_cli.api.rl import RLClient, RLRun


class FakeAPIClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any] | None]] = []
        self.posts: list[tuple[str, dict[str, Any] | None]] = []

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

    def post(self, endpoint: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        self.posts.append((endpoint, json))
        return {
            "run": {
                "id": "run-1",
                "userId": "user-1",
                "status": "QUEUED",
                "rolloutsPerExample": json["rollouts_per_example"] if json else 8,
                "seqLen": 2048,
                "maxSteps": json["max_steps"] if json else 100,
                "batchSize": json["batch_size"] if json else 128,
                "baseModel": json["model"]["name"] if json else "model",
                "maxInflightRollouts": json.get("max_inflight_rollouts") if json else None,
                "createdAt": "2026-05-17T00:00:00Z",
                "updatedAt": "2026-05-17T00:00:00Z",
            }
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


def test_create_run_sends_max_inflight_rollouts() -> None:
    api_client = FakeAPIClient()
    client = RLClient(api_client)  # type: ignore[arg-type]

    run = client.create_run(
        model_name="Qwen/Qwen3.5-0.8B",
        environments=[{"id": "reverse-text"}],
        max_inflight_rollouts=96,
    )

    assert api_client.posts[0][0] == "/rft/runs"
    assert api_client.posts[0][1]["max_inflight_rollouts"] == 96
    assert run.max_inflight_rollouts == 96


def test_create_run_sends_sft_loss_and_teacher_config() -> None:
    api_client = FakeAPIClient()
    client = RLClient(api_client)  # type: ignore[arg-type]

    client.create_run(
        model_name="openai/gpt-oss-20b",
        environments=[{"id": "primeintellect/reverse-text"}],
        max_tokens=512,
        loss="sft",
        teacher={
            "model": {"name": "openai/gpt-oss-120b"},
            "sampling": {
                "max_tokens": 2048,
                "reasoning_effort": "medium",
            },
        },
    )

    assert api_client.posts[0][0] == "/rft/runs"
    payload = api_client.posts[0][1]
    assert payload == {
        "model": {"name": "openai/gpt-oss-20b"},
        "environments": [{"id": "primeintellect/reverse-text"}],
        "rollouts_per_example": 8,
        "max_steps": 100,
        "batch_size": 128,
        "secrets": [],
        "loss": "sft",
        "teacher": {
            "model": {"name": "openai/gpt-oss-120b"},
            "sampling": {
                "max_tokens": 2048,
                "reasoning_effort": "medium",
            },
        },
        "max_tokens": 512,
    }


def test_create_run_sends_opd_loss_and_teacher_config() -> None:
    api_client = FakeAPIClient()
    client = RLClient(api_client)  # type: ignore[arg-type]

    client.create_run(
        model_name="openai/gpt-oss-20b",
        environments=[{"id": "primeintellect/reverse-text"}],
        loss="opd",
        teacher={"model": {"name": "openai/gpt-oss-120b"}},
    )

    assert api_client.posts[0][0] == "/rft/runs"
    payload = api_client.posts[0][1]
    assert payload["loss"] == "opd"
    assert payload["teacher"] == {"model": {"name": "openai/gpt-oss-120b"}}


def test_create_run_omits_default_rl_loss() -> None:
    api_client = FakeAPIClient()
    client = RLClient(api_client)  # type: ignore[arg-type]

    client.create_run(
        model_name="Qwen/Qwen3.5-0.8B",
        environments=[{"id": "reverse-text"}],
    )

    assert api_client.posts[0][0] == "/rft/runs"
    assert "loss" not in api_client.posts[0][1]
    assert "teacher" not in api_client.posts[0][1]
