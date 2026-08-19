"""Run-level aggregates over native episodes.

Opt-in, not automatic. The SDK does not decide what a run's headline number is
— that judgement belongs next to the producer, which knows which agents are
being scored and what counts as an error. This module ships the aggregation
verifiers already used, so callers migrating off ``verifiers.v1.utils.platform``
keep byte-identical dashboard numbers, and anyone else can pass their own dict
to ``run.finish(summary=...)``.

Duck-typed like :mod:`prime_runs.projection`: no producer package is imported.
"""

from typing import Any, Dict, Optional, Sequence


def from_episodes(
    episodes: Sequence[Any], traces: Optional[Sequence[Any]] = None
) -> Dict[str, Any]:
    """Run-level aggregates in the shape the eval dashboard reads.

    Rewards and metrics aggregate over the trainable traces only — fixed agents
    (a judge, a modeled user) often carry no rewards and would dilute every mean
    with structural zeros — falling back to all traces when none are trainable,
    the same rule the dashboard applies. ``avg_error`` is the share of EPISODES
    that aren't ok: a hook failure counts even when its traces are clean or it
    left none behind.
    """
    if traces is None:
        traces = [trace for episode in episodes for trace in episode.traces]
    scored = [trace for trace in traces if trace.agent.trainable] or list(traces)

    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for trace in scored:
        scores = {
            name: reward.score for name, reward in trace.rewards.items() if reward is not None
        }
        metrics = {name: value for name, value in trace.metrics.items() if value is not None}
        for name, value in {**scores, **metrics}.items():
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1

    n = len(scored)
    avg_error = sum(not episode.ok for episode in episodes) / len(episodes) if episodes else 0.0
    return {
        "avg_reward": sum(trace.reward for trace in scored) / n if n else 0.0,
        "avg_metrics": {name: sums[name] / counts[name] for name in sums},
        "avg_error": avg_error,
    }
