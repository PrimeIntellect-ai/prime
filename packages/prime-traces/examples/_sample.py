"""Fixture records shared by the examples.

Kept out of the examples themselves so each one shows the SDK surface and
nothing else. A real producer never writes these by hand — verifiers `Trace` /
`Episode` and prime-rl `Rollout` objects already expose `to_record()`, which is
the input `upload_records` is built around.

Only two fields are load-bearing for ingestion:

* a non-empty string ``id``
* a numeric ``timing.start`` inside the accepted window (365 days back,
  300 seconds forward -- producers upload completed work, so an old run is
  ordinary and a future one never is)

Everything else is optional and simply projects into summary columns when
present. A line missing either is rejected with ``invalid_trace`` /
``created_at_out_of_window``, and the whole request stores nothing.
"""

import time
import uuid

MODEL = "deepseek-v4-flash"


def new_run_id(label: str = "example") -> str:
    """A fresh run ID, so repeated runs of an example never collide."""
    return f"run_{label}_{uuid.uuid4().hex[:8]}"


def sample_trace(
    run_id: str,
    *,
    index: int = 0,
    reward: float = 1.0,
    outcome: str = "done",
    ok: bool = True,
    tokens: int = 1_200,
    started_at: float | None = None,
) -> dict:
    """One Verifiers-compatible trace record."""
    start = started_at if started_at is not None else time.time() - 60 + index
    return {
        "version": 4,
        "id": f"tr_{uuid.uuid4().hex[:16]}",
        "run": {"id": run_id},
        "task": {"type": "ExampleTask", "data": {"name": f"example-{index:04d}"}},
        "agent": {
            "name": "solver",
            "config": {"model": MODEL, "client": {"base_url": "https://api.pinference.ai/api/v1"}},
        },
        "calls": [{"model": MODEL, "usage": {"total_tokens": tokens}}],
        "rewards": {"correctness": {"score": reward, "weight": 1.0}},
        "metrics": {},
        "stop_condition": outcome,
        "ok": ok,
        "errors": [] if ok else [{"type": "ToolError", "message": "search timed out"}],
        # `timing.start` is the producer's wall clock and becomes `created_at`;
        # `timing.scoring.end` is the last phase, so `duration_ms` comes from
        # the two together.
        "timing": {"start": start, "scoring": {"end": start + 12.5}},
        "info": {},
    }


def sample_episode(run_id: str, *, environment_id: str, members: int = 3) -> dict:
    """One episode envelope wrapping several member traces.

    The envelope is Prime-owned and thin: it carries the episode's identity and
    verdict, and nests the member traces under ``traces``. Member ``run_id`` and
    the envelope's ``env.id`` are what the episode row inherits.
    """
    return {
        "version": 1,
        "id": f"ep_{uuid.uuid4().hex[:16]}",
        "env": {"id": environment_id},
        "outcome": "solved",
        "ok": True,
        "errors": [],
        "info": {"note": "three agents cooperating on one task"},
        "traces": [
            sample_trace(run_id, index=i, reward=0.5 * i, tokens=250) for i in range(members)
        ],
    }
