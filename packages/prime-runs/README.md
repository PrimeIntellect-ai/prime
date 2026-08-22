# Prime Runs SDK

Track eval runs on the Prime Intellect platform.

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
    config="eval.toml",          # the file the run was launched from
)
print(run.url)   # https://app.primeintellect.ai/dashboard/evaluations/eval-...

for episode in rollouts:          # episodes carry run.id — see "Identity" below
    run.log_episodes([episode])   # bare traces go through log_traces()

run.finish(summary=pr.metrics.from_episodes(episodes))
```

`init()` opens the run and returns a handle carrying its ID and dashboard URL.
Records stream out on a background thread while the run proceeds, so the
dashboard fills in as rollouts land. `finish()` closes the run out; a `with`
block does that for you, including when the body raises. Run-level outputs go
in `finish(summary=...)`, or incrementally through `update_summary()`;
`metrics.from_episodes()` returns them in the shape the dashboard reads
(`metrics.RunSummary`).

## Identity

`init()` is called **before** the first rollout, and the ID it returns is *the*
run ID everywhere — including inside every trace document you write:

```python
run = pr.init(...)
trace.record_run(EvalRunInfo(id=run.id))     # verifiers
```

The ingestion service indexes `run.id` from the trace body, so "every trace for
this run" is a fast query. Bare dicts without a `run` key are stamped for you;
producer objects are passed through untouched.

## Config

`config=` takes the path to the file the run was launched from, or a mapping.

| you pass | what is stored |
| --- | --- |
| a path | the file, byte for byte, under `config_source` |
| a mapping | exactly as given |

The file is the run's real configuration — comments, key order and section
grouping included — so it is stored verbatim, not parsed. A `str` or `Path` is
always a path; use `pr.ConfigSource(text=...)` for contents already in memory.
To send structured values *and* the file, put the file under
`pr.CONFIG_SOURCE_KEY` in the mapping:

```python
config = {**cfg.model_dump(exclude_unset=True),
          pr.CONFIG_SOURCE_KEY: pr.ConfigSource.from_file("eval.toml").to_dict()}
```

Nothing is redacted — keep credentials in the environment, not in the file.

## Modes

| mode | what happens |
| --- | --- |
| `online` | the run lives on the platform (default when an API key is present) |
| `disabled` | every call is a no-op, with the same object shape |

Set the mode explicitly, or through `$PRIME_RUNS_MODE`. A missing API key
disables the run with a warning — it never silently writes somewhere else.

## What the run handle does for you

- **Streams instead of buffering.** Records go out as they are produced.
- **Contains its own errors.** With the default `on_error="warn"`, nothing the
  platform raises escapes into your loop. Use `on_error="raise"` in tests and
  CI, where a silent upload failure is the bug; failures surface from `flush()`
  and `finish()`. Platform errors are the `prime_traces` exception family
  (`pr.APIError` and friends), so one set of `except` clauses covers both SDKs.
- **Applies backpressure.** The upload queue is bounded; a producer that
  durably outruns the uploader has records dropped and counted
  (`run.dropped_records`) rather than stalled. Per-sink losses are in
  `run.failed_records`.
- **Waits for its own uploads.** `finish()` gives queued records the same
  budget a single upload gets (300s) and warns if they do not drain.
- **Reports a terminal status.** The context manager and an `atexit` hook both
  route to the same idempotent `finish()`. A run you said failed is `failed`;
  one that stopped without saying — Ctrl-C inside a `with` block, an exit that
  never reached `finish()` — is `crashed`.

## From async code

`log_traces()` / `log_episodes()` are a queue put, not a request, so they are
safe to call from a coroutine; it blocks only if the queue is full (up to 5s), which is the
backpressure. `init()` and `finish()` do network I/O — wrap them in
`asyncio.to_thread` if a stall there would matter.

## Configuration

Resolved from environment variables first, then `~/.prime/config.json`:

| setting | env var | default |
| --- | --- | --- |
| API key | `PRIME_API_KEY` | — |
| team | `PRIME_TEAM_ID` | — |
| platform API | `PRIME_API_BASE_URL` | `https://api.primeintellect.ai` |
| dashboard | `PRIME_FRONTEND_URL` | `https://app.primeintellect.ai` |
| traces service | `PRIME_TRACES_URL` | resolved by `prime-traces` |

Or pass `api_key=`, `base_url=`, `team_id=` to `init()`.

## Transports

An online run writes every record to two sinks: Prime Traces (the system of
record — streaming, episode-aware, content-addressed and therefore idempotent
on retry) and the flat v0 sample table today's viewer reads. Both run because
Prime Traces is gated to an allowlist; an account outside it has the traces sink
turn itself off at the first upload — not counted as a failure, since nothing was
lost. When the viewer reads traces natively the default sink list drops one entry
and no producer changes.

`prime_runs.projection` holds `trace_to_sample` / `build_samples`, the v0
projection moved here from verifiers. `prime_runs.metrics.from_episodes` is the
run-level aggregation the eval dashboard reads. Both are duck-typed; no producer
package is imported — this is a leaf package, because the `prime` CLI depends on
`verifiers` and verifiers depends on this.

## Status

Eval runs only. Training runs will arrive with a backend over
`/api/v1/rft/external-runs`, designed against prime-rl's actual needs.

There is currently no producer-facing way to mark an evaluation **failed**. The
SDK records the terminal state under `metadata.prime_runs` and warns that the
run will keep showing as running on the dashboard.
