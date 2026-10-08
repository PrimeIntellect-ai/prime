"""Run-level aggregates, moved here from verifiers."""

from _fakes import Reward, make_episode, make_trace

from prime_runs import metrics


def test_only_trainable_traces_are_scored():
    episode = make_episode(
        "ep-1",
        [
            make_trace(trace_id="solver", reward=1.0),
            make_trace(trace_id="judge", agent="judge", trainable=False, reward=0.0),
        ],
    )

    assert metrics.from_episodes([episode])["avg_reward"] == 1.0


def test_all_traces_count_when_none_are_trainable():
    episode = make_episode(
        "ep-1",
        [
            make_trace(trace_id="a", trainable=False, reward=1.0),
            make_trace(trace_id="b", trainable=False, reward=0.0),
        ],
    )

    assert metrics.from_episodes([episode])["avg_reward"] == 0.5


def test_sub_rewards_and_env_metrics_average_together():
    episodes = [
        make_episode("ep-1", [make_trace(rewards={"format": Reward(1.0)}, metrics={"turns": 4.0})]),
        make_episode("ep-2", [make_trace(rewards={"format": Reward(0.0)}, metrics={"turns": 6.0})]),
    ]

    avg = metrics.from_episodes(episodes)["avg_metrics"]

    assert avg["format"] == 0.5
    assert avg["turns"] == 5.0


def test_a_metric_present_on_only_some_traces_averages_over_those_traces():
    episodes = [
        make_episode("ep-1", [make_trace(metrics={"partial": 1.0})]),
        make_episode("ep-2", [make_trace(metrics={})]),
    ]

    assert metrics.from_episodes(episodes)["avg_metrics"]["partial"] == 1.0


def test_avg_error_counts_episodes_not_traces():
    episodes = [
        make_episode("ok", [make_trace(), make_trace(trace_id="t2")], ok=True),
        make_episode("broken", [], ok=False),
    ]

    assert metrics.from_episodes(episodes)["avg_error"] == 0.5


def test_empty_run_produces_zeros_rather_than_dividing_by_zero():
    assert metrics.from_episodes([]) == {"avg_reward": 0.0, "avg_metrics": {}, "avg_error": 0.0}
