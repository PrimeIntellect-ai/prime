"""Projections of native episodes onto the platform's sample tables: the v0
eval-sample format (moved here from ``verifiers.v1.utils.platform``) and the
training viewer's Parquet table (moved here from prime-rl). Duck-typed: verifiers
``Trace``/``Episode`` satisfy it structurally and are not imported."""

import json
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ._http import encode_json

logger = logging.getLogger(__name__)

#: The platform's per-request ceiling on ``POST /samples``.
MAX_SAMPLES_PAYLOAD_BYTES = 25 * 1024 * 1024
ENVELOPE_BYTES = len(b'{"samples":[]}')


def json_bytes(value: Any) -> int:
    return len(encode_json(value))


def _dump(items: Iterable[Any]) -> List[Dict[str, Any]]:
    return [item.model_dump(mode="json", exclude_none=True) for item in items]


def _nullify_non_finite(value: Any) -> Any:
    """NaN and infinity are not JSON. The tables show them as ``null``
    rather than losing the row (or the batch) they sit in."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _nullify_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nullify_non_finite(item) for item in value]
    return value


def _json_column(value: Any) -> str:
    return json.dumps(_nullify_non_finite(value), allow_nan=False)


def summary_trace_index(episode: Any) -> int:
    """The first trainable trace, else the first trace: the one whose flat
    projection (and ``task.data.idx``) represents the episode."""
    return next(
        (index for index, trace in enumerate(episode.traces) if trace.agent.trainable),
        0,
    )


def trace_to_sample(
    trace: Any, rollout_number: int = 1, episode_id: Optional[str] = None
) -> Dict[str, Any]:
    """One trace as a v0 sample row. ``completion`` is the final branch's
    messages, ``trajectory`` one message list per branch."""
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
    # Sub-rewards are flattened to top-level keys the way v0 did; env metrics stay nested.
    for name, reward in trace.rewards.items():
        if reward is not None:
            sample.setdefault(name, reward.score)
    return _nullify_non_finite(sample)


def episode_to_samples(episode: Any, rollout_number: int) -> List[Dict[str, Any]]:
    """Normally one row: the summary trace's projection, with the whole native
    episode in ``info.native_wrapper``. An episode too large for one request
    falls back to one projected row per trace."""
    if not episode.traces:
        return []
    summary_index = summary_trace_index(episode)
    sample = trace_to_sample(episode.traces[summary_index], rollout_number, episode.id)
    sample["sample_id"] = episode.id
    sample["info"] = _nullify_non_finite(
        {
            **(sample["info"] or {}),
            "native_wrapper": episode.to_record(),
            "native_trace_index": summary_index,
        }
    )
    if ENVELOPE_BYTES + json_bytes(sample) <= MAX_SAMPLES_PAYLOAD_BYTES:
        return [sample]
    logger.warning(
        "Episode %s exceeds the platform sample limit; uploading projected traces", episode.id
    )
    return [trace_to_sample(trace, rollout_number, episode.id) for trace in episode.traces]


def build_samples(
    episodes: Sequence[Any], rollout_numbers: Optional[Dict[Any, int]] = None
) -> List[Dict[str, Any]]:
    """Project episodes to v0 samples. A streaming producer passes the same
    ``rollout_numbers`` dict every call so numbering stays consistent."""
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
    """Split samples into request-sized batches; a sample too large to send
    alone raises rather than being silently dropped."""
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
        next_bytes = payload_bytes + (1 if batch else 0) + sample_bytes  # +1: the joining comma
        if batch and next_bytes > MAX_SAMPLES_PAYLOAD_BYTES:
            batches.append(batch)
            batch = []
            next_bytes = ENVELOPE_BYTES + sample_bytes
        batch.append(sample)
        payload_bytes = next_bytes
    if batch:
        batches.append(batch)
    return batches


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def train_sample_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("run_id", pa.string()),
            ("step", pa.int64()),
            ("tag", pa.string()),
            ("problem_id", pa.int64()),
            ("sample_id", pa.int64()),
            ("prompt", pa.string()),
            ("completion", pa.string()),
            ("trajectory", pa.string()),
            ("answer", pa.string()),
            ("env_name", pa.string()),
            ("task", pa.string()),
            ("info", pa.string()),
            ("reward", pa.float64()),
            ("advantage", pa.float64()),
            ("metrics", pa.string()),
            ("timing", pa.string()),
            ("num_input_tokens", pa.int64()),
            ("num_output_tokens", pa.int64()),
            ("created_at", pa.timestamp("us", tz="UTC")),
        ]
    )


def episodes_to_parquet_bytes(
    episodes: Sequence[Any], run_id: str, step: int, *, sample_id_offset: int = 0
) -> Optional[bytes]:
    """One training step's episodes as the viewer's Parquet table, one row per
    episode (the v0 projection plus the RFT-only columns), numbered from
    ``sample_id_offset``. ``None`` when no episode has a trajectory."""
    import io
    from datetime import datetime, timezone

    import pyarrow as pa
    import pyarrow.parquet as pq

    advantages: Dict[Any, Optional[float]] = {}
    env_names: Dict[Any, str] = {}
    for episode in episodes:
        if not episode.traces:
            continue
        summary_trace = episode.traces[summary_trace_index(episode)]
        advantages[episode.id] = (summary_trace.info or {}).get("advantage")
        env = getattr(episode, "env", None)
        env_names[episode.id] = str(getattr(env, "id", None) or "")

    now = datetime.now(timezone.utc)
    rows: List[Dict[str, Any]] = []
    for sample_id, sample in enumerate(build_samples(episodes), start=sample_id_offset):
        trajectory = sample["trajectory"]
        if not trajectory:
            continue
        advantage = advantages.get(sample["episode_id"])
        trajectory = [{**branch, "advantage": advantage} for branch in trajectory]
        try:
            problem_id = (
                int(sample["example_id"]) if sample["example_id"] is not None else sample_id
            )
        except (TypeError, ValueError):
            problem_id = sample_id
        rows.append(
            {
                "run_id": run_id,
                "step": step,
                "tag": "",
                "problem_id": problem_id,
                "sample_id": sample_id,
                "prompt": "",
                "completion": _json_column(sample["completion"]),
                "trajectory": _json_column(trajectory),
                "answer": "",
                "env_name": env_names.get(sample["episode_id"], ""),
                "task": _json_column(sample["task"]),
                "info": _json_column(sample["info"]),
                "reward": sample["reward"],
                "advantage": advantage,
                "metrics": _json_column(sample["metrics"]),
                "timing": _json_column(sample["timing"]),
                "num_input_tokens": trajectory[-1]["num_input_tokens"],
                "num_output_tokens": trajectory[-1]["num_output_tokens"],
                "created_at": now,
            }
        )
    if not rows:
        return None

    table = pa.Table.from_pylist(rows, schema=train_sample_schema())
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="snappy", use_dictionary=True, write_statistics=True)
    return buffer.getvalue()
