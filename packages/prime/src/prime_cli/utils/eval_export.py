import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

ExportFormat = Literal["verifiers", "inspect"]

FAILED_SAMPLE_STATUSES = {"failed", "failure", "timeout", "timed_out", "cancelled", "canceled"}
TERMINAL_INCOMPLETE_EVAL_STATUSES = {"FAILED", "TIMEOUT", "CANCELLED", "CANCELED"}
ACTIVE_EVAL_STATUSES = {"PENDING", "RUNNING", "PROCESSING"}


def default_export_path(run_id: str, export_format: ExportFormat) -> Path:
    extension = "jsonl" if export_format == "verifiers" else "eval"
    return Path(f"{run_id}.{extension}")


def normalize_export_format(raw_format: str) -> ExportFormat:
    if raw_format not in ("verifiers", "inspect"):
        raise ValueError("format must be one of: verifiers, inspect")
    return raw_format


def is_active_evaluation(evaluation: dict[str, Any]) -> bool:
    return str(evaluation.get("status", "")).upper() in ACTIVE_EVAL_STATUSES


def _json_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    messages = []
    for item in value:
        if isinstance(item, dict):
            role = item.get("role")
            if not isinstance(role, str) or not role:
                role = "assistant"
            normalized = {"role": role, "content": _json_string(item.get("content"))}
            if item.get("tool_calls"):
                normalized["tool_calls"] = item["tool_calls"]
            if item.get("tool_call_id"):
                normalized["tool_call_id"] = item["tool_call_id"]
            messages.append(normalized)
            continue
        messages.append({"role": "assistant", "content": _json_string(item)})
    return messages


def _prompt_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return _messages(sample.get("prompt") or sample.get("input"))


def _completion_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    completion = _messages(sample.get("completion"))
    if completion:
        return completion

    output = sample.get("output")
    if isinstance(output, dict):
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                choice_messages = _messages([first_choice.get("message")])
                if choice_messages:
                    return choice_messages
        if "message" in output:
            message = output["message"]
            return (
                _messages([message])
                if isinstance(message, dict)
                else _messages([{"role": "assistant", "content": message}])
            )
        if "content" in output:
            return _messages([{"role": "assistant", "content": output["content"]}])

    if output is not None:
        return _messages([{"role": "assistant", "content": output}])

    answer = sample.get("answer")
    return _messages([{"role": "assistant", "content": answer}]) if answer is not None else []


def _target_answer(sample: dict[str, Any]) -> str:
    for key in ("target", "reference", "ground_truth", "expected_answer"):
        if key in sample:
            return _json_string(sample[key])

    info = sample.get("info")
    if isinstance(info, dict):
        for key in ("target", "reference", "ground_truth", "answer"):
            if key in info:
                return _json_string(info[key])

    return _json_string(sample.get("answer"))


def _reward(sample: dict[str, Any]) -> Optional[float]:
    for key in ("reward", "score"):
        value = sample.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    scores = sample.get("scores")
    if isinstance(scores, dict):
        reward_score = scores.get("reward")
        if isinstance(reward_score, dict):
            value = reward_score.get("value")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _reward_breakdown(sample: dict[str, Any]) -> dict[str, float]:
    for key in ("rewards", "metrics"):
        value = sample.get(key)
        if isinstance(value, dict):
            return {
                str(metric_name): float(metric_value)
                for metric_name, metric_value in value.items()
                if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool)
            }
    return {}


def _rollout_index(sample: dict[str, Any]) -> int:
    for key in ("rollout_index", "rollout_number", "epoch"):
        value = sample.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value - 1 if key == "epoch" else value)
    return 0


def _example_id(sample: dict[str, Any], fallback_index: int) -> str:
    if "example_id" in sample:
        return str(sample["example_id"])
    if "id" in sample:
        return str(sample["id"])
    return str(fallback_index)


def _is_failed_sample(sample: dict[str, Any]) -> bool:
    if sample.get("error") or sample.get("error_message"):
        return True
    status = sample.get("status")
    return isinstance(status, str) and status.lower() in FAILED_SAMPLE_STATUSES


def filter_export_samples(
    samples: Iterable[dict[str, Any]],
    *,
    include_failed: bool,
    min_reward: Optional[float],
    max_reward: Optional[float],
) -> list[dict[str, Any]]:
    filtered = []
    for sample in samples:
        if not include_failed and _is_failed_sample(sample):
            continue

        reward = _reward(sample)
        if min_reward is not None and (reward is None or reward < min_reward):
            continue
        if max_reward is not None and (reward is None or reward > max_reward):
            continue
        filtered.append(sample)
    return filtered


def _environment_name(evaluation: dict[str, Any]) -> str:
    metadata = evaluation.get("metadata")
    if isinstance(metadata, dict):
        for key in ("env", "env_id", "environment"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value

    environment_names = evaluation.get("environment_names")
    if isinstance(environment_names, list) and environment_names:
        return str(environment_names[0])

    environments = evaluation.get("environments")
    if isinstance(environments, list) and environments:
        first_environment = environments[0]
        if isinstance(first_environment, dict):
            name = first_environment.get("name")
            owner_slug = first_environment.get("owner_slug")
            if owner_slug and name:
                return f"{owner_slug}/{name}"
            if name:
                return str(name)

    return str(evaluation.get("name") or "evaluation")


def _model_name(evaluation: dict[str, Any]) -> str:
    metadata = evaluation.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("model"), str):
        return metadata["model"]
    return str(evaluation.get("model_name") or evaluation.get("inference_model") or "")


def _env_version(evaluation: dict[str, Any]) -> str:
    for key in ("semantic_version", "version_id"):
        value = evaluation.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _created_at(evaluation: dict[str, Any]) -> str:
    value = (
        evaluation.get("started_at") or evaluation.get("created_at") or evaluation.get("createdAt")
    )
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _completed_at(evaluation: dict[str, Any]) -> str:
    value = (
        evaluation.get("completed_at")
        or evaluation.get("updated_at")
        or evaluation.get("updatedAt")
    )
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return _created_at(evaluation)


def build_verifiers_rows(
    evaluation: dict[str, Any], samples: list[dict[str, Any]], run_id: str
) -> list[dict[str, Any]]:
    model = _model_name(evaluation)
    env = _environment_name(evaluation)
    env_version = _env_version(evaluation)
    timestamp = _completed_at(evaluation)

    rows = []
    for index, sample in enumerate(samples):
        row = {
            "prompt": _prompt_messages(sample),
            "answer": _target_answer(sample),
            "reward": _reward(sample),
            "rollout_index": _rollout_index(sample),
            "example_id": _example_id(sample, index),
            "completion": _completion_messages(sample),
            "run_id": run_id,
            "model": model,
            "env": env,
            "env_version": env_version,
            "timestamp": timestamp,
        }
        rewards = _reward_breakdown(sample)
        if rewards:
            row["rewards"] = rewards
        rows.append(row)
    return rows


def write_verifiers_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _stderr(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def build_inspect_log(
    evaluation: dict[str, Any], samples: list[dict[str, Any]], run_id: str
) -> dict[str, Any]:
    model = _model_name(evaluation)
    env = _environment_name(evaluation)
    env_version = _env_version(evaluation)
    raw_eval_config = evaluation.get("eval_config")
    eval_config: dict[str, Any] = raw_eval_config if isinstance(raw_eval_config, dict) else {}
    raw_metadata = evaluation.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    rewards = [reward for sample in samples if (reward := _reward(sample)) is not None]
    mean_reward = _mean(rewards)
    stderr_reward = _stderr(rewards)
    status = (
        "error"
        if str(evaluation.get("status", "")).upper() in TERMINAL_INCOMPLETE_EVAL_STATUSES
        else "success"
    )

    return {
        "version": 2,
        "status": status,
        "eval": {
            "run_id": run_id,
            "task": env,
            "task_version": env_version,
            "task_id": "",
            "task_file": None,
            "model": model,
            "model_args": {},
            "config": {
                "max_retries": eval_config.get("max_retries") or metadata.get("max_retries"),
                "timeout": eval_config.get("timeout_minutes") or metadata.get("timeout_minutes"),
                "rollouts": eval_config.get("rollouts_per_example")
                or metadata.get("rollouts_per_example"),
            },
            "dataset": {
                "name": evaluation.get("dataset") or metadata.get("dataset") or env,
                "location": evaluation.get("dataset") or metadata.get("dataset") or env,
            },
            "created": _created_at(evaluation),
        },
        "plan": {
            "name": "prime-hosted-eval",
            "steps": ["generate"],
            "finish": "end_turn",
        },
        "results": {
            "total_samples": len(samples),
            "completed_samples": len(
                [sample for sample in samples if not _is_failed_sample(sample)]
            ),
            "scores": [
                {
                    "name": "reward",
                    "scorer": "prime_reward",
                    "metrics": {
                        "mean": {"name": "mean", "value": mean_reward},
                        "stderr": {"name": "stderr", "value": stderr_reward},
                    },
                }
            ],
        },
        "stats": {
            "started_at": _created_at(evaluation),
            "completed_at": _completed_at(evaluation),
            "model_usage": {},
        },
        "samples": [
            _build_inspect_sample(sample, index, run_id, model, env, env_version)
            for index, sample in enumerate(samples)
        ],
    }


def _build_inspect_sample(
    sample: dict[str, Any], index: int, run_id: str, model: str, env: str, env_version: str
) -> dict[str, Any]:
    reward = _reward(sample)
    completion = _completion_messages(sample)
    assistant_message = completion[0] if completion else {"role": "assistant", "content": ""}
    example_id = _example_id(sample, index)
    rollout_index = _rollout_index(sample)

    return {
        "id": sample.get("example_id", sample.get("id", index)),
        "epoch": rollout_index + 1,
        "input": _prompt_messages(sample),
        "target": _target_answer(sample),
        "output": {
            "model": model,
            "choices": [{"message": assistant_message, "stop_reason": "end_turn"}],
        },
        "scores": {
            "reward": {
                "value": reward,
                "answer": sample.get("answer"),
                "explanation": None,
            }
        },
        "metadata": {
            "env": env,
            "env_version": env_version,
            "run_id": run_id,
            "example_id": example_id,
            "rollout_index": rollout_index,
        },
        "error": sample.get("error") or sample.get("error_message"),
    }


def write_inspect_eval(path: Path, log: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("log.json", json.dumps(log, ensure_ascii=False, indent=2))
