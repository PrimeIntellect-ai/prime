from __future__ import annotations

from typing import Any

import pytest
from prime_cli.api.rl import RLClient, RLRun
from prime_cli.core import APIError

RUN_PAYLOAD = {
    "id": "run-1",
    "userId": "user-1",
    "createdAt": "2026-08-25T00:00:00Z",
    "updatedAt": "2026-08-25T00:00:00Z",
}


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


class FakeRunActionClient:
    """Answers every request with ``response`` and records the exact call."""

    def __init__(self, response: dict[str, Any] | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request(
        self, method: str, endpoint: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.calls.append((method, endpoint, json))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


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


def test_create_run_sends_v1_env_payload_unchanged() -> None:
    api_client = FakeAPIClient()
    client = RLClient(api_client)  # type: ignore[arg-type]
    v1_env = {
        "name": "alphabet-sort",
        "taskset": {"id": "alphabet-sort-v1", "min_turns": 3},
        "harness": {"id": "default", "runtime": {"type": "subprocess"}},
        "group_size": 8,
    }
    eval_config = {
        "environments": [
            {
                "taskset": {"id": "alphabet-sort-v1", "split": "test"},
                "harness": {"id": "default"},
                "group_size": 4,
            }
        ]
    }

    client.create_run(
        model_name="Qwen/Qwen3.5-0.8B",
        environments=[v1_env],
        eval_config=eval_config,
    )

    assert api_client.posts[0][0] == "/rft/runs"
    payload = api_client.posts[0][1]
    assert payload["environments"] == [v1_env]
    assert payload["eval"] == eval_config


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


@pytest.mark.parametrize(
    ("act", "response", "expected_call", "expected_result"),
    [
        (
            lambda client: client.checkpoint_run("run-1"),
            {"run_id": "run-1", "request_id": "req-1"},
            ("POST", "/rft/runs/run-1/checkpoint", None),
            "req-1",
        ),
        (
            lambda client: client.pause_run("run-1"),
            {"run": {**RUN_PAYLOAD, "status": "PAUSING"}},
            ("PUT", "/rft/runs/run-1/pause", None),
            RLRun.model_validate({**RUN_PAYLOAD, "status": "PAUSING"}),
        ),
        (
            lambda client: client.restart_run("run-1"),
            {"run": {**RUN_PAYLOAD, "status": "PENDING"}},
            ("PUT", "/rft/runs/run-1/restart", None),
            RLRun.model_validate({**RUN_PAYLOAD, "status": "PENDING"}),
        ),
        (
            lambda client: client.restart_run(
                "run-1",
                secrets={"WANDB_API_KEY": "wk", "HF_TOKEN": "hf", "OPENAI_API_KEY": "ok"},
            ),
            {"run": {**RUN_PAYLOAD, "status": "PENDING"}},
            (
                "PUT",
                "/rft/runs/run-1/restart",
                {"wandbApiKey": "wk", "hfToken": "hf", "secrets": {"OPENAI_API_KEY": "ok"}},
            ),
            RLRun.model_validate({**RUN_PAYLOAD, "status": "PENDING"}),
        ),
    ],
    ids=["checkpoint", "pause", "restart", "restart_with_secrets"],
)
def test_run_action(act, response, expected_call, expected_result) -> None:
    api_client = FakeRunActionClient(response)

    result = act(RLClient(api_client))

    assert (result, api_client.calls) == (expected_result, [expected_call])


def test_run_action_conflict() -> None:
    conflict = APIError("HTTP 409: The run changed state; retry")
    api_client = FakeRunActionClient(conflict)

    with pytest.raises(APIError) as raised:
        RLClient(api_client).pause_run("run-1")

    assert (type(raised.value), raised.value.__cause__) == (APIError, conflict)
