"""The v0 eval-sample projection, moved here from verifiers.

These assertions are the contract the current viewer reads. They exist so the
move is provably a relocation and not a rewrite: a run pushed through this SDK
must produce the same rows verifiers produced.
"""

import pytest
from _fakes import Reward, make_episode, make_trace

from prime_runs.projection import (
    MAX_SAMPLES_PAYLOAD_BYTES,
    batch_samples,
    build_samples,
    summary_trace_index,
    trace_to_sample,
)


def test_trace_to_sample_carries_the_flat_row():
    trace = make_trace(trace_id="t1", idx=7, reward=0.5, metrics={"tokens": 12.0})

    sample = trace_to_sample(trace, rollout_number=3, episode_id="ep-9")

    assert sample["sample_id"] == "t1"
    assert sample["example_id"] == 7
    assert sample["rollout_number"] == 3
    assert sample["episode_id"] == "ep-9"
    assert sample["agent"] == "solver"
    assert sample["trainable"] is True
    assert sample["reward"] == 0.5
    assert sample["metrics"] == {"tokens": 12.0}
    # No prompt/completion split mid-branch: completion is the final branch.
    assert sample["prompt"] == []
    assert sample["completion"] == [{"role": "assistant", "content": "branch 0"}]


def test_trace_to_sample_flattens_sub_rewards_to_top_level():
    trace = make_trace(rewards={"format": Reward(1.0), "correct": Reward(0.0), "skip": None})

    sample = trace_to_sample(trace)

    assert sample["format"] == 1.0
    assert sample["correct"] == 0.0
    assert "skip" not in sample


def test_trace_to_sample_does_not_let_a_sub_reward_clobber_a_real_field():
    """``setdefault`` semantics: a sub-reward named ``reward`` must not win.

    Sub-rewards are flattened into the same namespace as the row's own columns,
    so an environment that names a reward function after one of them would
    otherwise silently overwrite the value the dashboard reads.
    """
    trace = make_trace(reward=0.25, rewards={"reward": Reward(9.0)})

    assert trace_to_sample(trace)["reward"] == 0.25


def test_trajectory_keeps_one_entry_per_branch():
    trace = make_trace(branches=3)

    trajectory = trace_to_sample(trace)["trajectory"]

    assert len(trajectory) == 3
    assert trajectory[1]["messages"] == [{"role": "assistant", "content": "branch 1"}]
    assert trajectory[0]["num_input_tokens"] == 10


def test_build_samples_emits_one_row_per_episode_with_the_native_wrapper():
    episode = make_episode("ep-1", [make_trace(trace_id="t1")])

    samples = build_samples([episode])

    assert len(samples) == 1
    assert samples[0]["sample_id"] == "ep-1"
    assert samples[0]["info"]["native_wrapper"] == episode.to_record()
    assert samples[0]["info"]["native_trace_index"] == 0


def test_summary_trace_is_the_first_trainable_one():
    """A judge or modeled user must not become the row the dashboard shows."""
    episode = make_episode(
        "ep-1",
        [
            make_trace(trace_id="judge", agent="judge", trainable=False, reward=0.0),
            make_trace(trace_id="solver", reward=1.0),
        ],
    )

    assert summary_trace_index(episode) == 1
    sample = build_samples([episode])[0]
    assert sample["info"]["native_trace_index"] == 1
    assert sample["reward"] == 1.0


def test_rollout_numbers_continue_across_streaming_calls():
    """Streaming uploads must number rollouts the way one final upload did.

    The old code built every sample in a single call, so its per-example counter
    lived in a local. Streaming means several calls, and without a carried
    counter every batch would restart at rollout 1.
    """
    counters: dict = {}
    first = build_samples([make_episode("ep-1", [make_trace(idx=4)])], counters)
    second = build_samples([make_episode("ep-2", [make_trace(idx=4)])], counters)
    other_example = build_samples([make_episode("ep-3", [make_trace(idx=5)])], counters)

    assert first[0]["rollout_number"] == 1
    assert second[0]["rollout_number"] == 2
    assert other_example[0]["rollout_number"] == 1


def test_episodes_without_traces_are_skipped():
    assert build_samples([make_episode("empty", [])]) == []


def test_batch_samples_splits_on_the_payload_ceiling():
    big = {"sample_id": "x", "blob": "a" * (MAX_SAMPLES_PAYLOAD_BYTES // 3)}
    batches = batch_samples([dict(big, sample_id=str(n)) for n in range(4)])

    assert len(batches) > 1
    assert sum(len(batch) for batch in batches) == 4


def test_batch_samples_refuses_a_sample_that_cannot_be_sent():
    """Dropping it would report a complete run that is missing rows."""
    oversized = {"sample_id": "x", "blob": "a" * (MAX_SAMPLES_PAYLOAD_BYTES + 1)}

    with pytest.raises(ValueError, match="too large"):
        batch_samples([oversized])


def test_batch_samples_returns_nothing_for_no_samples():
    assert batch_samples([]) == []


def test_is_episode_distinguishes_episodes_from_traces_in_either_form():
    from prime_runs.sinks import is_episode

    assert is_episode(make_episode())
    assert not is_episode(make_trace())
    assert is_episode({"id": "e", "traces": []})
    assert not is_episode({"id": "t"})
