"""Read verifiers v1 eval output (`config.toml` + a `Trace`-per-line `results.jsonl`).

A v1 run dir is `outputs/<taskset>--<model>--<harness>/<uuid>` (or `…--legacy` for a v0 env
run through the bridge) holding `config.toml` + `results.jsonl`, where each line is a serialized
`verifiers.v1.trace.Trace`. This differs from the v0 layout (`outputs/evals/<env>--<model>/<uuid>`
with `results.jsonl` of `RolloutOutput` + a precomputed `metadata.json`).

The Lab viewer renders the v0 record shape, so `v1_trace_to_record` adapts a `Trace` to that shape
(`prompt` / `completion` / `reward` / `metrics` / `info` / `error`), and `v1_run_metadata`
synthesizes the run-level `metadata.json` the viewer expects from `config.toml` + the trace rewards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

V1_CONFIG_NAME = "config.toml"
RESULTS_NAME = "results.jsonl"


def is_v1_eval_dir(run_dir: Path) -> bool:
    """A v1 run dir has a `config.toml` next to `results.jsonl` (v0 writes `metadata.json`)."""
    return (run_dir / V1_CONFIG_NAME).is_file() and (run_dir / RESULTS_NAME).is_file()


def load_config(run_dir: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11
        import tomli as tomllib  # type: ignore[no-redef]
    try:
        with (run_dir / V1_CONFIG_NAME).open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def is_v1_trace(record: dict[str, Any]) -> bool:
    """A serialized v1 `Trace` carries a `nodes` graph and no top-level `prompt` (which every v0
    `RolloutOutput` has)."""
    return "nodes" in record and "prompt" not in record


def _main_branch_messages(nodes: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Split the trace's first root→leaf branch into (prompt, completion) messages: prompt is the
    supplied prefix (`sampled` is False) up to the first sampled node, completion is everything from
    the first sampled node on (assistant turns plus any tool results between them)."""
    if not nodes:
        return [], []
    parents = {node.get("parent") for node in nodes}
    leaf = next((i for i in range(len(nodes)) if i not in parents), len(nodes) - 1)

    chain: list[dict[str, Any]] = []
    idx: Any = leaf
    seen = set()
    while isinstance(idx, int) and 0 <= idx < len(nodes) and idx not in seen:
        seen.add(idx)
        chain.append(nodes[idx])
        idx = nodes[idx].get("parent")
    chain.reverse()

    prompt: list[dict] = []
    completion: list[dict] = []
    started = False
    for node in chain:
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        if node.get("sampled"):
            started = True
        (completion if started else prompt).append(message)
    return prompt, completion


def _error_record(error: Any) -> dict[str, Any] | None:
    if not isinstance(error, dict):
        return None
    return {"error": error.get("type", ""), "message": error.get("message", "")}


def v1_trace_to_record(trace: dict[str, Any]) -> dict[str, Any]:
    """Adapt a serialized v1 `Trace` to the v0 `RolloutOutput`-shaped record the viewer reads."""
    task = trace.get("task") if isinstance(trace.get("task"), dict) else {}
    prompt, completion = _main_branch_messages(trace.get("nodes") or [])
    # The viewer flattens reward/@metric values into one `metrics` table; rewards is the weighted
    # per-@reward breakdown, metrics the unweighted side metrics.
    metrics: dict[str, Any] = {}
    metrics.update(trace.get("rewards") or {})
    metrics.update(trace.get("metrics") or {})

    record: dict[str, Any] = {
        "example_id": task.get("idx", 0),
        "prompt": prompt,
        "completion": completion,
        "reward": trace.get("reward", 0.0),
        "metrics": metrics,
        "info": trace.get("info") or {},
        "is_completed": trace.get("is_completed"),
        "is_truncated": trace.get("is_truncated"),
        "stop_condition": trace.get("stop_condition"),
    }
    answer = task.get("answer")
    if answer is not None:
        record["answer"] = answer
    error = _error_record(trace.get("error"))
    if error is not None:
        record["error"] = error
    return record


def v1_run_identity(config: dict[str, Any]) -> tuple[str, str, str | None]:
    """(env_id, model, harness) for a v1 run, from its `config.toml`."""
    taskset = config.get("taskset") if isinstance(config.get("taskset"), dict) else {}
    harness = config.get("harness") if isinstance(config.get("harness"), dict) else {}
    env_id = taskset.get("id") or config.get("id") or "-"
    return str(env_id), str(config.get("model") or "-"), harness.get("id")


def v1_config_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """The run-level metadata derivable from `config.toml` alone (no `results.jsonl` scan) — used at
    discovery so listing many runs stays cheap. Reward aggregates are filled in lazily per selected
    run by `v1_run_metadata` / the viewer's stats worker."""
    env_id, model, harness = v1_run_identity(config)
    num_tasks = config.get("num_tasks")
    metadata: dict[str, Any] = {
        "env_id": env_id,
        "env": env_id,
        "model": model,
        "harness": harness,
        "framework": "verifiers",
        "format": "v1",
        "rollouts_per_example": config.get("num_rollouts", 1),
    }
    if num_tasks is not None:
        metadata["num_examples"] = num_tasks
    if isinstance(config.get("sampling"), dict):
        metadata["sampling_args"] = config["sampling"]
    client = config.get("client") if isinstance(config.get("client"), dict) else {}
    if client.get("base_url"):
        metadata["base_url"] = client["base_url"]
    return metadata


def v1_run_metadata(config: dict[str, Any], results_path: Path) -> dict[str, Any]:
    """Synthesize the run-level `metadata.json` the viewer expects from `config.toml` plus the
    rewards/metrics scanned from `results.jsonl`."""
    env_id, model, harness = v1_run_identity(config)
    rewards: list[float] = []
    metric_totals: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    task_idxs: set[Any] = set()

    try:
        with results_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    trace = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(trace, dict):
                    continue
                reward = trace.get("reward")
                if isinstance(reward, (int, float)):
                    rewards.append(float(reward))
                task = trace.get("task")
                if isinstance(task, dict) and "idx" in task:
                    task_idxs.add(task["idx"])
                merged = {**(trace.get("rewards") or {}), **(trace.get("metrics") or {})}
                for name, value in merged.items():
                    if isinstance(value, (int, float)):
                        metric_totals[name] = metric_totals.get(name, 0.0) + float(value)
                        metric_counts[name] = metric_counts.get(name, 0) + 1
    except OSError:
        pass

    num_tasks = config.get("num_tasks")
    metadata: dict[str, Any] = {
        "env_id": env_id,
        "env": env_id,
        "model": model,
        "harness": harness,
        "framework": "verifiers",
        "format": "v1",
        "num_examples": num_tasks if num_tasks is not None else len(task_idxs),
        "rollouts_per_example": config.get("num_rollouts", 1),
    }
    if rewards:
        metadata["avg_reward"] = sum(rewards) / len(rewards)
    if metric_totals:
        metadata["avg_metrics"] = {
            name: metric_totals[name] / metric_counts[name] for name in metric_totals
        }
    if isinstance(config.get("sampling"), dict):
        metadata["sampling_args"] = config["sampling"]
    client = config.get("client") if isinstance(config.get("client"), dict) else {}
    if client.get("base_url"):
        metadata["base_url"] = client["base_url"]
    return metadata
