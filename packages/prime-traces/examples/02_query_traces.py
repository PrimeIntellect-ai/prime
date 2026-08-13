#!/usr/bin/env python3
"""2 — Finding traces and reading them back.

The consumer's view: something happened during a run and you want to see it.
This is the surface people touch repeatedly, from notebooks and scripts, so it
is the one where naming and defaults matter most.

Run it:

    PRIME_API_KEY=... PRIME_TRACES_URL=https://dev-prime-traces.pintel.dev \
        uv run python examples/02_query_traces.py
"""

import json
import tempfile
from pathlib import Path

from _sample import new_run_id, sample_trace

from prime_traces import TracesClient

RUN_ID = new_run_id("query")


def seed(client: TracesClient) -> None:
    """A run with a spread of rewards and one failure, so filters have something to bite on."""
    records = [sample_trace(RUN_ID, index=i, reward=i / 10) for i in range(10)]
    records.append(sample_trace(RUN_ID, index=99, reward=0.0, outcome="error", ok=False))
    client.upload_records(iter(records), context={"source": "example"})


def main() -> None:
    client = TracesClient()
    seed(client)
    print(f"seeded {RUN_ID}\n")

    # ------------------------------------------------------------------
    # One page, newest first.
    # ------------------------------------------------------------------
    page = client.list(run_id=RUN_ID, limit=5)
    print(f"list(limit=5) -> {len(page.items)} items, more={page.next_cursor is not None}")
    for t in page.items:
        # Every summary field is a projection of the stored document. A field
        # the producer never recorded reads as None rather than an empty string
        # you would have to know to special-case.
        reward = "unscored" if t.score.reward is None else f"{t.score.reward:.2f}"
        print(f"  {t.trace_id[:20]:22} {t.task_id:14} reward={reward:>8} tokens={t.total_tokens}")

    # ------------------------------------------------------------------
    # Filters compose; `iter` handles pagination.
    # ------------------------------------------------------------------
    # `list` gives you one page and a cursor when you want to drive paging
    # yourself. `iter` takes the same filters and walks every page, which is
    # what you want in a loop.
    print("\nfailures only:")
    for t in client.iter(run_id=RUN_ID, has_error=True):
        print(f"  {t.trace_id[:20]:22} outcome={t.score.outcome}")

    print("\nreward >= 0.8, best first:")
    for t in client.iter(run_id=RUN_ID, reward_min=0.8, sort="reward"):
        print(f"  {t.trace_id[:20]:22} reward={t.score.reward:.2f}")

    # Driving the cursor by hand, for the rare caller that needs to checkpoint
    # between pages rather than hold one iterator open.
    print("\npaging manually:")
    cursor, seen = None, 0
    while True:
        p = client.list(run_id=RUN_ID, limit=4, cursor=cursor)
        seen += len(p.items)
        print(f"  page of {len(p.items)} (total {seen})")
        cursor = p.next_cursor
        if not cursor:
            break

    # ------------------------------------------------------------------
    # One trace: summary, then the document itself.
    # ------------------------------------------------------------------
    trace_id = page.items[0].trace_id
    summary = client.get(trace_id)
    print(f"\nget({trace_id[:20]}…)")
    print(f"  model     {summary.model.provider}/{summary.model.id}")
    print(f"  agent     {summary.agent_name}")
    print(f"  duration  {summary.duration_ms} ms")
    print(f"  context   {summary.context}")

    # The exact bytes the producer uploaded, unchanged. Summaries are a
    # projection of this, never a replacement for it.
    raw = client.get_raw(trace_id)
    print(f"\nget_raw -> {len(raw)} bytes, keys: {sorted(json.loads(raw))[:6]}…")

    # For anything large, stream it to disk instead of holding it in memory.
    # The write goes to a sibling temporary file and is swapped into place, so
    # an interrupted download leaves any existing file untouched.
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "trace.json"
        written = client.download_raw(trace_id, dest)
        print(f"download_raw -> {written} bytes at {dest.name}")

    client.delete_run(RUN_ID)
    print(f"\ndeleted {RUN_ID}")


if __name__ == "__main__":
    main()
