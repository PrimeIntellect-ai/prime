# Prime Runs SDK

Track eval and training runs on the Prime Intellect platform.

```bash
pip install prime-runs
```

## Quick start

```python
import prime_runs as pr

run = pr.init(
    name="gsm8k-qwen3-8b",
    environments=["gsm8k"],
    model="Qwen/Qwen3-8B",
    framework="verifiers",
    config={"num_rollouts": 4, "max_tokens": 2048},
)
print(run.url)   # https://app.primeintellect.ai/dashboard/evaluations/eval-...

for episode in rollouts:          # episodes carry run.id — see "Identity" below
    run.log_traces([episode])
    run.log({"reward": episode.reward})

run.finish(summary=pr.metrics.from_episodes(episodes))
```

`init()` opens the run and returns a handle carrying its ID and dashboard URL.
Records stream out on a background thread while the run proceeds — the dashboard
fills in as rollouts land, rather than all at once at the end. `finish()` closes
the run out.

Prefer a `with` block, which also reports a terminal status on the paths where
you never reach `finish()`:

```python
with pr.init(name="gsm8k-qwen3-8b", environments=["gsm8k"]) as run:
    ...
```

## Identity

`init()` is called **before** the first rollout, and the ID it returns is *the*
run ID everywhere — including inside every trace document you write, and
including the local archive. Stamp it once at rollout time:

```python
run = pr.init(...)
trace.record_run(EvalRunInfo(id=run.id))     # verifiers
```

Nothing is re-stamped afterwards and no record of yours is rewritten. The
ingestion service extracts `run.id` from the trace body into an indexed column
with a delete-by-run path, so "every trace for this run" is a fast query rather
than a scan over upload metadata.

`init()` also exports `PRIME_RUN_ID`, so forked workers and subprocess launchers
join the run their parent opened instead of each opening their own.

## Config

One parameter, in whatever form you have it — the same shape `environments=`
already has.

```python
pr.init(config="eval.toml")   # the file the run was launched from
pr.init(config={"n": 4})      # a mapping
pr.init(config=cfg)           # a pydantic model
```

| you pass | what is stored |
| --- | --- |
| a path | the file, byte for byte, under `config_source` |
| a mapping | exactly as given |
| a pydantic model | only the fields somebody set (`exclude_unset=True`) |
| `cfg.model_dump()` | every field, defaults included |

**A path** is the common case: both producers are launched from one user-authored
file (`uv run eval @ eval.toml`, `uv run rl @ train.toml`), and that file *is* the
run's configuration. It is kept verbatim — comments, key order and section
grouping included, none of which survive a dict round-trip. A `str` or `Path` is
always a path, never inline text; use `pr.ConfigSource(text=...)` if you already
hold the contents. It is stored, not parsed: parsing would buy a second
representation of something the platform can already read, at the cost of a TOML
dependency in a package that deliberately has two.

**A pydantic model** is dumped with `exclude_unset=True` because a resolved dump
of a deep config tree is hundreds of lines nobody chose, and a reader scrolling it
cannot tell which three values were the experiment. The full dump is still one
explicit `cfg.model_dump()` away — the shorter call gives the more useful answer.

Values land in the run's config either way, so a run launched from a file that
also wants a derived value adds it with `run.update_config({...})`.

Nothing is redacted. A config file holding a secret puts that secret on the run's
page — keep credentials in the environment, not in the file.

## Modes

| mode | what happens |
| --- | --- |
| `online` | the run lives on the platform (default when an API key is present) |
| `offline` | the run lives in a local directory, ready to sync later |
| `disabled` | every call is a no-op, with the same object shape |

An offline run is a real run: a real ID, a status, a config, a summary, a metrics
stream, and records written in the JSONL wire format the traces service accepts.
That is why producers do not need a `--no-push` branch — the call sites are
identical either way, and a missing API key degrades to offline rather than
skipping the run.

Set the mode explicitly, or through `$PRIME_RUNS_MODE`.

## What the run handle does for you

- **Streams instead of buffering.** Records go out as they are produced, so a
  run with a hundred thousand episodes does not hold them all in memory.
- **Contains its own errors.** With the default `on_error="warn"`, nothing the
  platform raises escapes into your loop. Use `on_error="raise"` in tests and CI,
  where a silent upload failure is the bug.
- **Applies backpressure.** The upload queue is bounded; if a producer durably
  outruns the uploader, records are dropped and counted (`run.dropped_records`)
  rather than stalling the run.
- **Waits for its own uploads.** `finish()` gives queued records the same budget
  a single upload gets (300s, `finish_timeout=`) and says so in a warning if
  they do not drain, rather than finalizing over records still in flight.
- **Survives forks.** A forked child gets a fresh uploader, a fresh connection
  pool and fresh file handles instead of writing the parent's — which would
  interleave two processes into one HTTP stream and flush the parent's buffered
  records a second time. It also joins the parent's run rather than opening
  its own.
- **Reports a terminal status.** Context manager, `atexit` and signal handlers
  all route to the same idempotent `finish()`. A run the producer decided had
  failed is `failed`; one stopped from outside its control flow — Ctrl-C,
  SIGTERM, an exit that never reached `finish()` — is `crashed`. Neither is
  left running forever.
- **Knows about ranks.** Rank 0 owns creation and finalization; other ranks join
  through `PRIME_RUN_ID` and upload their own records.

## Using it from async code

There is no `AsyncRun`, unlike the async clients in `prime-traces`,
`prime-evals` and `prime-sandboxes`. The uploader thread is what replaces it:
`log()`, `log_traces()` and `update_config()` are queue puts, not requests, so
calling them straight from a coroutine does no network I/O on the event loop.

Three calls do block, and all three are worth knowing about:

| call | blocks on | when it matters |
| --- | --- | --- |
| `init()` | create + environment resolution | once, at startup |
| `finish()` | draining the queue, then finalize | once, at shutdown |
| `log_traces()` | up to `put_timeout` (5s) **only if the queue is full** | a producer durably outrunning the uploader |

The first two are run boundaries — `await asyncio.to_thread(run.finish)` if a
stall there would matter. The third is the one to watch in a hot rollout loop:
the block is the backpressure, and past it the batch is dropped and counted in
`run.dropped_records`. Raise `queue_size=` before reaching for a thread.

## Configuration

Resolved from environment variables first, then `~/.prime/config.json`:

| setting | env var | default |
| --- | --- | --- |
| API key | `PRIME_API_KEY` | — |
| team | `PRIME_TEAM_ID` | — |
| platform API | `PRIME_API_BASE_URL` | `https://api.primeintellect.ai` |
| dashboard | `PRIME_FRONTEND_URL` | `https://app.primeintellect.ai` |
| traces service | `PRIME_TRACES_URL` | resolved by `prime-traces` |
| offline runs | `PRIME_RUNS_DIR` | `./prime-runs` |

Or pass them to `init()` directly (`api_key=`, `base_url=`, `team_id=`, `dir=`).

## Backends and sinks

Two independent axes:

- **Backends** own run *lifecycle* — `EvalsBackend` (`/api/v1/evaluations/*`),
  `OfflineBackend` (a local directory). Selected by `kind`.
- **Sinks** own sample *transport* — `TracesSink` (primary; streaming,
  episode-aware, content-addressed and therefore idempotent on retry) and
  `EvalSamplesSink` (the flat v0 sample table today's viewer reads).

Both sinks run during the transition, because Prime Traces is in closed beta and
a traces-only client would leave non-allowlisted accounts with an empty
dashboard. When the Viewer API reads traces natively, the default sink list drops
one entry — and no producer changes.

Turn either off with `pr.init(traces=False)` / `pr.init(samples=False)`.

## Also here

`prime_runs.projection` holds `trace_to_sample` / `build_samples`, the projection
from native traces onto the platform's v0 eval-sample format. It lives here
because it is knowledge about a platform wire format, not about any one eval
framework. `prime_runs.metrics.from_episodes` is the run-level aggregation the
eval dashboard reads — opt-in, because what a run's headline number means is a
judgement that belongs next to the producer.

Both are duck-typed: verifiers `Trace`/`Episode` and prime-rl `Rollout` satisfy
them structurally, and no producer package is imported. This is a leaf package by
design — the `prime` CLI depends on `verifiers`, so verifiers can never depend on
`prime`.

## How this differs from the other prime SDKs

`prime-sandboxes`, `prime-traces`, `prime-evals` and `prime-tunnel` are all built
the same way: a `core/` subpackage holding an `APIClient` and a `Config`, pydantic
models for the responses, and a sync/async client pair as the thing you import.
Three deliberate departures here, so the difference reads as a choice rather than
an oversight:

- **The client is private.** `init()` is the surface, not a client object, so the
  HTTP layer lives in `_http.py` rather than `core/client.py` and `PlatformClient`
  is not exported. Config still is, and is the same class as everywhere else —
  `~/.prime/config.json`, env wins, same variable names.
- **Responses are not modeled.** The platform returns more fields than any
  producer reads; freezing them in pydantic would make every backend addition a
  breaking SDK release. Backends take the two or three fields they need and
  return a `RunHandle`. That is why the local types are dataclasses and why
  pydantic is not a dependency.
- **No async client.** See [Using it from async code](#using-it-from-async-code)
  — the background uploader covers the case an async client would exist for.

## Related packages

- [prime-traces](../prime-traces) — the traces service client this SDK streams
  through, and the direct API for querying or exporting what a run produced.
- [prime](../prime) — the CLI and full SDK. Depends on this package's consumers,
  never the other way around.

## Status

Eval runs are supported. Training runs (`kind="train"`, over
`/api/v1/rft/external-runs`) are next; `pr.init(kind="train")` raises a clear
error until then.

One platform gap is worth knowing about: there is currently no producer-facing
way to mark an evaluation **failed**. The SDK calls the status endpoint it needs,
treats its absence as expected, and records the terminal state in the run's
metadata as a fallback — a failed run will keep showing as running on the
dashboard, and the SDK says so in a warning.
