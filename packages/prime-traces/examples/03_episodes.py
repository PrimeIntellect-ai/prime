#!/usr/bin/env python3
"""3 — Episodes: several agents, one unit of work.

An episode is the envelope around a group of traces that belong together. It
carries its own verdict, so an environment-level failure is visible even when
every individual trace succeeded -- which is the case a flat list of traces
cannot express.

Run it:

    PRIME_API_KEY=... PRIME_TRACES_URL=https://dev-prime-traces.pintel.dev \
        uv run python examples/03_episodes.py
"""

import uuid

from _sample import new_run_id, sample_episode

from prime_traces import LineFormat, TracesClient

RUN_ID = new_run_id("episodes")
ENVIRONMENT_ID = f"env_{uuid.uuid4().hex[:8]}"


def main() -> None:
    client = TracesClient()

    # ------------------------------------------------------------------
    # Upload: same call, different line format.
    # ------------------------------------------------------------------
    # Each *line* is one complete episode with its members nested inside, so
    # an episode is never split across batches and its members are stored
    # atomically with it.
    episodes = [sample_episode(RUN_ID, environment_id=ENVIRONMENT_ID) for _ in range(2)]
    client.upload_records(
        iter(episodes),
        line_format=LineFormat.EPISODE,
        context={"source": "example"},
    )
    episode_id = episodes[0]["id"]
    print(f"uploaded 2 episodes to {RUN_ID}\n")

    # ------------------------------------------------------------------
    # List episodes.
    # ------------------------------------------------------------------
    # The episode's own fields only -- identity, verdict, provenance. Nothing
    # here requires reading the members.
    page = client.list_episodes(run_id=RUN_ID)
    print(f"list_episodes -> {len(page.items)}")
    for e in page.items:
        print(
            f"  {e.episode_id[:20]:22} outcome={e.outcome:8} "
            f"has_error={e.has_error} env={e.environment_id}"
        )

    # `run_id` and `environment_id` are inherited: the run from the members,
    # the environment from the envelope's `env.id`. Worth knowing that
    # environment_id is only populated this way -- a bare trace upload leaves
    # it empty, so the filter is useful for episodes and inert for flat traces.

    # ------------------------------------------------------------------
    # One episode, with the member rollup.
    # ------------------------------------------------------------------
    detail = client.get_episode(episode_id)
    agg = detail.traces
    print(f"\nget_episode({episode_id[:20]}…)")
    print(f"  episode says   outcome={detail.outcome} has_error={detail.has_error}")
    print(
        f"  members say    traces={agg.trace_count} tokens={agg.total_tokens} "
        f"duration={agg.total_duration_ms}ms any_error={agg.any_trace_error}"
    )
    print(f"  agents         {agg.agent_names}")

    # The two error signals are deliberately separate. `detail.has_error` is
    # the envelope's own verdict; `agg.any_trace_error` is whether any member
    # failed. An environment hook that blew up after every trace succeeded
    # shows as True / False -- collapsing them would hide exactly that case.

    # ------------------------------------------------------------------
    # The members themselves.
    # ------------------------------------------------------------------
    members = client.list_episode_traces(episode_id)
    print(f"\nlist_episode_traces -> {len(members.items)}")
    for t in members.items:
        reward = "unscored" if t.score.reward is None else f"{t.score.reward:.2f}"
        print(f"  {t.trace_id[:20]:22} reward={reward:>8} episode={t.episode_id[:12]}…")

    # A member is an ordinary trace: readable, filterable and fetchable through
    # every call in example 2.
    by_env = client.list(environment_id=ENVIRONMENT_ID, limit=100)
    print(f"\nsame rows via list(environment_id=…): {len(by_env.items)}")

    # ------------------------------------------------------------------
    # Cleanup -- and one caveat.
    # ------------------------------------------------------------------
    # `delete_run` removes the member traces but leaves the episode rows
    # behind, and no API can remove them (ENG-5239). Re-running this example is
    # safe, but each run leaves two episode rows with trace_count=0.
    client.delete_run(RUN_ID)
    orphans = [e.episode_id for e in client.list_episodes(run_id=RUN_ID).items]
    print(f"\ndeleted {RUN_ID}; {len(orphans)} episode row(s) survive the delete")


if __name__ == "__main__":
    main()
