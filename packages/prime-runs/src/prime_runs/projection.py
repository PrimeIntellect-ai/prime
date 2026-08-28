"""Projection of native traces onto the platform's v0 eval-sample format.

Moved here from ``verifiers.v1.utils.platform``: this is knowledge about a
platform wire format, so it lives in the client for that wire. Duck-typed —
verifiers ``Trace``/``Episode`` satisfy it structurally and are not imported.

The projection serves the *current* viewer, which reads the flat sample table;
it leaves the default path once the Viewer API reads traces natively.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

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


# ------------------------------------------------------------ training table


def parquet_available() -> bool:
    """Whether the training sample table can be encoded (``prime-runs[train]``)."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def train_sample_schema() -> Any:
    """The training sample table's Parquet schema (prime-rl's ``SAMPLE_SCHEMA``)."""
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


def episodes_to_parquet_bytes(episodes: Sequence[Any], run_id: str, step: int) -> Optional[bytes]:
    """One training step's episodes as the viewer's Parquet table, one row per
    episode. Moved here from prime-rl's ``monitors/prime.py``.

    Sample construction is shared with the eval projection (:func:`build_samples`:
    the complete native episode in ``info.native_wrapper``, a flat summary from
    one trainable trace), so a training episode and an eval sample land on the
    platform identically; the RFT-only columns (run/step/advantage/problem_id/
    env_name) are layered on here. ``None`` when no episode has a trajectory.
    """
    import io
    import json
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
    for sample_id, sample in enumerate(build_samples(episodes)):
        trajectory = sample["trajectory"]
        if not trajectory:  # no branches: an episode that errored before any message
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
                "completion": json.dumps(sample["completion"]),
                "trajectory": json.dumps(trajectory),
                "answer": "",
                "env_name": env_names.get(sample["episode_id"], ""),
                "task": json.dumps(sample["task"]),
                "info": json.dumps(sample["info"]),
                "reward": sample["reward"],
                "advantage": advantage,
                "metrics": json.dumps(sample["metrics"]),
                "timing": json.dumps(sample["timing"]),
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
