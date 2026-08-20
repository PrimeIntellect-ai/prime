"""Projection of native traces onto the platform's v0 eval-sample format.

Moved here from ``verifiers.v1.utils.platform``. This is knowledge about a
*platform wire format*, so it belongs in the client for that wire — not in an
eval framework that prime-rl then has to reach across a repo boundary to import
(``from verifiers.v1.push import trace_to_sample``, which had already drifted
from the module's real path).

Everything is duck-typed. Verifiers ``Trace``/``Episode`` and prime-rl
``Rollout`` satisfy it structurally, and none of them is imported: the leaf
package that both producers depend on cannot depend back on either.

The projection exists for the *current* viewer, which reads the flat sample
table. Once the Viewer API reads traces natively this module stops being on
the default path — which is why it is a standalone function rather than
something woven through the run lifecycle.
"""

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ._http import encode_json

logger = logging.getLogger(__name__)

# Repeated /samples posts append; this is the platform's per-request ceiling.
MAX_SAMPLES_PAYLOAD_BYTES = 25 * 1024 * 1024
# The bytes an empty {"samples":[]} envelope costs, counted against every batch.
ENVELOPE_BYTES = len(b'{"samples":[]}')


def json_bytes(value: Any) -> int:
    """Serialized size of ``value`` under the encoding actually sent."""
    return len(encode_json(value))


def _dump(items: Iterable[Any]) -> List[Dict[str, Any]]:
    return [item.model_dump(mode="json", exclude_none=True) for item in items]


def is_episode(record: Any) -> bool:
    """Whether a record is an episode (a group of traces) rather than a trace."""
    return hasattr(record, "traces") and not hasattr(record, "branches")


def summary_trace_index(episode: Any) -> int:
    """Index of the trace whose flat projection represents the episode.

    The first trainable trace, else the first trace. Shared by the projection
    and the rollout counter so the two can never disagree about which trace
    (and therefore which ``task.data.idx``) an episode is numbered under.
    """
    return next(
        (index for index, trace in enumerate(episode.traces) if trace.agent.trainable),
        0,
    )


def trace_to_sample(
    trace: Any, rollout_number: int = 1, episode_id: Optional[str] = None
) -> Dict[str, Any]:
    """One trace -> the platform's sample dict (the v0 eval-sample format).

    The hub table stays flat — one row per trace; its episode is denormalized
    onto the row (``episode_id`` from the envelope, plus the trace's own
    ``agent``/``trainable``), so a multi-trace rollout's grouping travels with
    each row without a nested schema. No prompt/completion split (meaningless
    mid-branch): ``completion`` is the final branch's messages, ``trajectory``
    one message list per branch.
    """
    task = trace.task.data.model_dump(mode="json", exclude_none=True)
    branches = trace.branches
    sample: Dict[str, Any] = {
        "sample_id": trace.id,
        "example_id": trace.task.data.idx,
        "rollout_number": rollout_number,
        "episode_id": episode_id,
        "agent": trace.agent.name,
        "trainable": trace.agent.trainable,
        "task": task,
        "prompt": [],
        "completion": _dump(branches[-1].messages) if branches else [],
        "answer": task.get("answer"),
        # Keyed `tool_defs` because the v0 sample format already carries it there.
        "tool_defs": _dump(trace.tools) if trace.tools else None,
        "reward": trace.reward,
        "timing": trace.timing.model_dump(mode="json", exclude_none=True),
        "is_completed": trace.is_completed,
        "is_truncated": trace.is_truncated,
        "metrics": trace.metrics,
        "error": trace.last_error.model_dump(mode="json", exclude_none=True)
        if trace.last_error
        else None,
        "stop_condition": trace.stop_condition,
        "trajectory": [
            {
                "messages": _dump(branch.messages),
                "num_input_tokens": branch.num_input_tokens,
                "num_output_tokens": branch.num_output_tokens,
            }
            for branch in branches
        ],
        "token_usage": trace.usage.model_dump(mode="json", exclude_none=True)
        if trace.usage
        else None,
        "info": dict(trace.info) or None,
    }
    # Flatten sub-rewards to top-level keys the way v0 does (raw scores, as v0's
    # per-function outputs were); env metrics stay nested.
    for name, reward in trace.rewards.items():
        if reward is not None:
            sample.setdefault(name, reward.score)
    return sample


def trace_record_to_sample(
    trace: Mapping[str, Any], rollout_number: int = 1, episode_id: Optional[str] = None
) -> Dict[str, Any]:
    """Project a serialized trace without importing its producer package.

    Verifiers persists its message graph as ``nodes``/``calls`` rather than the
    derived ``branches`` property used by :func:`trace_to_sample`. Older records
    may carry branches directly, so both representations are accepted. Sparse
    trace mappings still produce a visible row as long as they have an ID;
    fields the legacy viewer cannot derive remain empty instead of making the
    entire record disappear.
    """
    trace_id = trace.get("id")
    if not trace_id:
        raise TypeError("serialized trace records must contain a non-empty 'id'")

    task_container = _as_mapping(trace.get("task"))
    task = dict(_as_mapping(task_container.get("data")))
    agent = _as_mapping(trace.get("agent"))
    branches = _serialized_branches(trace)
    rewards = _as_mapping(trace.get("rewards"))
    errors = trace.get("errors")
    last_error = trace.get("last_error")
    if last_error is None and isinstance(errors, list) and errors:
        last_error = errors[-1]
    stop_condition = trace.get("stop_condition")

    sample: Dict[str, Any] = {
        "sample_id": trace_id,
        "example_id": task.get("idx"),
        "rollout_number": rollout_number,
        "episode_id": episode_id,
        "agent": agent.get("name"),
        "trainable": agent.get("trainable", True),
        "task": task,
        "prompt": [],
        "completion": branches[-1]["messages"] if branches else [],
        "answer": task.get("answer"),
        "tool_defs": _mapping_list(trace.get("tools")) or None,
        "reward": trace["reward"] if "reward" in trace else _total_reward(rewards),
        "timing": dict(_as_mapping(trace.get("timing"))) or None,
        "is_completed": trace.get("is_completed", False),
        "is_truncated": trace.get("is_truncated", _is_truncated(stop_condition, trace)),
        "metrics": dict(_as_mapping(trace.get("metrics"))),
        "error": dict(last_error) if isinstance(last_error, Mapping) else last_error,
        "stop_condition": stop_condition,
        "trajectory": branches,
        "token_usage": dict(_as_mapping(trace.get("usage"))) or _aggregate_usage(trace),
        "info": dict(_as_mapping(trace.get("info"))) or None,
    }
    for name, reward in rewards.items():
        if reward is None:
            continue
        score = reward.get("score") if isinstance(reward, Mapping) else reward
        sample.setdefault(name, score)
    return sample


def record_to_samples(
    record: Mapping[str, Any], rollout_numbers: Optional[Dict[Any, int]] = None
) -> List[Dict[str, Any]]:
    """Project one serialized trace or episode to legacy viewer samples."""
    counts = rollout_numbers if rollout_numbers is not None else {}
    if "traces" not in record:
        task = _as_mapping(_as_mapping(record.get("task")).get("data"))
        idx = task.get("idx")
        rollout_key = idx if idx is not None else record.get("id")
        counts[rollout_key] = number = counts.get(rollout_key, 0) + 1
        return [trace_record_to_sample(record, rollout_number=number)]

    episode_id = record.get("id")
    if not episode_id:
        raise TypeError("serialized episode records must contain a non-empty 'id'")
    raw_traces = record.get("traces")
    if not isinstance(raw_traces, list):
        raise TypeError("serialized episode 'traces' must be a list")
    traces: List[Mapping[str, Any]] = []
    for trace in raw_traces:
        if not isinstance(trace, Mapping):
            raise TypeError("serialized episode traces must be mappings")
        traces.append(trace)
    if not traces:
        return []

    summary_index = next(
        (
            index
            for index, trace in enumerate(traces)
            if _as_mapping(trace.get("agent")).get("trainable", True)
        ),
        0,
    )
    summary_task = _as_mapping(_as_mapping(traces[summary_index].get("task")).get("data"))
    idx = summary_task.get("idx")
    rollout_key = idx if idx is not None else episode_id
    counts[rollout_key] = number = counts.get(rollout_key, 0) + 1
    sample = trace_record_to_sample(traces[summary_index], number, str(episode_id))
    sample["sample_id"] = episode_id
    sample["info"] = {
        **(sample["info"] or {}),
        "native_wrapper": dict(record),
        "native_trace_index": summary_index,
    }
    if ENVELOPE_BYTES + json_bytes(sample) <= MAX_SAMPLES_PAYLOAD_BYTES:
        return [sample]

    logger.warning(
        "Episode %s exceeds the platform sample limit; uploading projected traces",
        episode_id,
    )
    return [trace_record_to_sample(trace, number, str(episode_id)) for trace in traces]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _serialized_branches(trace: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_branches = trace.get("branches")
    if isinstance(raw_branches, list):
        return [
            {
                "messages": _mapping_list(_as_mapping(branch).get("messages")),
                "num_input_tokens": _as_mapping(branch).get("num_input_tokens", 0),
                "num_output_tokens": _as_mapping(branch).get("num_output_tokens", 0),
            }
            for branch in raw_branches
            if isinstance(branch, Mapping)
        ]

    raw_nodes = trace.get("nodes")
    if not isinstance(raw_nodes, list):
        return []
    nodes = [_as_mapping(node) for node in raw_nodes]
    parents = {node.get("parent") for node in nodes if isinstance(node.get("parent"), int)}
    leaves = [index for index in range(len(nodes)) if index not in parents]
    calls = trace.get("calls")
    calls_by_node = (
        {
            call.get("node"): call
            for call in calls
            if isinstance(call, Mapping) and isinstance(call.get("node"), int)
        }
        if isinstance(calls, list)
        else {}
    )

    branches: List[Dict[str, Any]] = []
    for leaf in leaves:
        path: List[int] = []
        seen = set()
        node_index: Any = leaf
        while (
            isinstance(node_index, int) and 0 <= node_index < len(nodes) and node_index not in seen
        ):
            seen.add(node_index)
            path.append(node_index)
            node_index = nodes[node_index].get("parent")
        path.reverse()
        branch_calls = [calls_by_node[index] for index in path if index in calls_by_node]
        input_tokens, output_tokens = _branch_token_counts(branch_calls)
        branches.append(
            {
                "messages": [
                    dict(message)
                    for index in path
                    if isinstance((message := nodes[index].get("message")), Mapping)
                ],
                "num_input_tokens": input_tokens,
                "num_output_tokens": output_tokens,
            }
        )
    return branches


def _branch_token_counts(calls: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    previous_total = 0
    for call in calls:
        usage = _as_mapping(call.get("usage"))
        current_input, current_output = _usage_counts(usage)
        input_tokens += max(0, current_input - previous_total)
        output_tokens += current_output
        previous_total = int(usage.get("total_tokens", current_input + current_output) or 0)
    return input_tokens, output_tokens


def _aggregate_usage(trace: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    calls = trace.get("calls")
    if not isinstance(calls, list):
        return None
    usages = [usage for call in calls if (usage := _as_mapping(_as_mapping(call).get("usage")))]
    if not usages:
        return None
    if any("prompt_tokens" in usage or "completion_tokens" in usage for usage in usages):
        result: Dict[str, Any] = {
            "prompt_tokens": sum(int(usage.get("prompt_tokens", 0) or 0) for usage in usages),
            "completion_tokens": sum(
                int(usage.get("completion_tokens", 0) or 0) for usage in usages
            ),
        }
        for key in ("cached_input_tokens", "reasoning_tokens", "cost"):
            values = [usage[key] for usage in usages if usage.get(key) is not None]
            if values:
                result[key] = sum(values)
        return result

    input_tokens = 0
    output_tokens = 0
    for usage in usages:
        current_input, current_output = _usage_counts(usage)
        input_tokens += current_input
        output_tokens += current_output
    if not input_tokens and not output_tokens:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _usage_counts(usage: Mapping[str, Any]) -> tuple[int, int]:
    if "input_tokens" in usage:
        input_tokens = int(usage.get("input_tokens", 0) or 0)
    else:
        input_tokens = int(usage.get("prompt_tokens", 0) or 0) + int(
            usage.get("cached_input_tokens", 0) or 0
        )
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    return input_tokens, output_tokens


def _total_reward(rewards: Mapping[str, Any]) -> float:
    total = 0.0
    for reward in rewards.values():
        if reward is None:
            continue
        if isinstance(reward, Mapping):
            total += float(reward.get("score", 0.0) or 0.0) * float(
                reward.get("weight", 1.0) or 0.0
            )
        else:
            total += float(reward)
    return total


def _is_truncated(stop_condition: Any, trace: Mapping[str, Any]) -> bool:
    if stop_condition in {
        "max_turns",
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "context_length",
    }:
        return True
    calls = trace.get("calls")
    if not isinstance(calls, list):
        return False
    last_successful = next(
        (
            call
            for call in reversed(calls)
            if isinstance(call, Mapping) and call.get("error") is None
        ),
        None,
    )
    return bool(last_successful and last_successful.get("finish_reason") == "length")


def episode_to_samples(episode: Any, rollout_number: int) -> List[Dict[str, Any]]:
    """One episode -> the sample rows the platform should store for it.

    Normally a single row: the native episode in ``info.native_wrapper`` is
    authoritative and carries every trace, while one trainable trace (or the
    first) supplies the flat summary older consumers read, identified by
    ``native_trace_index``. An episode too large for one request falls back to
    one projected row per trace, which loses the native wrapper but keeps the
    run visible rather than dropping it.
    """
    if not episode.traces:
        return []
    summary_index = summary_trace_index(episode)
    summary_trace = episode.traces[summary_index]
    sample = trace_to_sample(summary_trace, rollout_number, episode.id)
    sample["sample_id"] = episode.id
    sample["info"] = {
        **(sample["info"] or {}),
        "native_wrapper": episode.to_record(),
        "native_trace_index": summary_index,
    }
    if ENVELOPE_BYTES + json_bytes(sample) <= MAX_SAMPLES_PAYLOAD_BYTES:
        return [sample]

    logger.warning(
        "Episode %s exceeds the platform sample limit; uploading projected traces",
        episode.id,
    )
    return [trace_to_sample(trace, rollout_number, episode.id) for trace in episode.traces]


def build_samples(
    episodes: Sequence[Any], rollout_numbers: Optional[Dict[Any, int]] = None
) -> List[Dict[str, Any]]:
    """Project episodes to platform samples.

    ``rollout_numbers`` carries the per-example counter across calls. Streaming
    producers pass the same dict every time so rollout numbering stays
    consistent when episodes arrive in batches instead of one final list; a
    one-shot caller can ignore it.
    """
    counts = rollout_numbers if rollout_numbers is not None else {}
    samples: List[Dict[str, Any]] = []
    for episode in episodes:
        if not episode.traces:
            continue
        idx = episode.traces[summary_trace_index(episode)].task.data.idx
        counts[idx] = number = counts.get(idx, 0) + 1
        samples.extend(episode_to_samples(episode, number))
    return samples


def batch_samples(samples: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split samples into request-sized batches.

    Raises ``ValueError`` on a sample too large to send alone: silently dropping
    it would report a successful run that is missing rows.
    """
    batches: List[List[Dict[str, Any]]] = []
    batch: List[Dict[str, Any]] = []
    payload_bytes = ENVELOPE_BYTES
    for index, sample in enumerate(samples):
        sample_bytes = json_bytes(sample)
        if ENVELOPE_BYTES + sample_bytes > MAX_SAMPLES_PAYLOAD_BYTES:
            raise ValueError(
                f"sample {index} is too large to upload "
                f"({ENVELOPE_BYTES + sample_bytes} > {MAX_SAMPLES_PAYLOAD_BYTES} bytes)"
            )
        # The +1 is the comma that joins this sample to the previous one.
        next_bytes = payload_bytes + (1 if batch else 0) + sample_bytes
        if batch and next_bytes > MAX_SAMPLES_PAYLOAD_BYTES:
            batches.append(batch)
            batch = []
            next_bytes = ENVELOPE_BYTES + sample_bytes
        batch.append(sample)
        payload_bytes = next_bytes
    if batch:
        batches.append(batch)
    return batches
