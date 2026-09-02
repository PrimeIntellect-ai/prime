# Prime Runs SDK

Track evaluation and training runs on the Prime Intellect platform: `init()`
opens the run, records stream out while it proceeds, and `finish()` closes it
out with a terminal status.

## Install

```bash
uv add prime-runs            # or: pip install prime-runs
uv add 'prime-runs[train]'   # training runs: adds pyarrow for the sample table
```

## Eval runs

```python
import prime_runs as pr

run = pr.init(
    name="gsm8k-qwen3-8b",
    environments=["gsm8k"],      # hub names (get-or-create) or owner/name slugs
    model="Qwen/Qwen3-8B",
    framework="verifiers",
    config="eval.toml",          # the launched file, stored byte for byte
)
print(run.url)                   # https://app.primeintellect.ai/dashboard/evaluations/...

for episode in rollouts:
    run.log_episodes([episode])  # a queue put; bare traces: log_traces()

run.finish(summary=pr.metrics.from_episodes(episodes))
```

`init()` is called before the first rollout. Every record the run uploads is
keyed to it: the SDK sets `run.id` and `run.type` on the uploaded copy of each
trace and episode, over whatever run id the producer recorded locally, and
keeps the rest of that block (`name`, `work`). A producer never needs to know
the platform's id; `run.url` is the handle. A `with run:` block finishes for
you: an exception marks the run `failed`, Ctrl-C `cancelled`, and a process
that exits without finishing is reported `crashed` by an atexit hook.

`config=` takes the path to the launched file (kept verbatim under
`config_source`, comments and all) or a mapping stored as given; put a file
under `pr.CONFIG_SOURCE_KEY` in the mapping to send both. Nothing is redacted.

## Training runs

```python
run = pr.init(
    kind="train",
    name="qwen3-8b-gsm8k-rl",
    model="Qwen/Qwen3-8B",                    # the base model
    environments=["primeintellect/gsm8k"],    # hub ids, passed through
    training=pr.TrainingSpec(max_steps=1000, batch_size=64, rollouts_per_example=8),
    config=train_config.model_dump(),
    team_id="team_...",                       # external runs belong to a team
)

for step, (episodes, metrics) in enumerate(training_loop):
    run.log_episodes(episodes)          # episodes carry run.work.step (TrainRunInfo)
    run.log_metrics(metrics, step=step)

run.finish()
```

- The platform enables external runs per team; a team outside the allowlist
  gets a `ForbiddenError` from `init()`.
- `init(kind="train", id=os.environ["RUN_ID"])` attaches to an external run a
  launcher already created: nothing is registered, the platform keeps the run's
  failure marking, and a clean `finish()` still completes it.
- Metrics are one row per `log_metrics` call, on their own uploader. The sample
  table gets one Parquet object per upload, every 10th step, keyed by the step
  an episode was dispatched at; a step logged in several calls gets several
  objects, and the viewer shows their union.
- The status vocabulary is `completed | failed`; `cancelled` and `crashed`
  arrive as `failed` with the reason in `error_message`.

## How it behaves

- **Streams.** Records go out on a background thread as they are logged;
  whatever queues up during one request goes out as the next.
- **Contains its errors.** With the default `on_error="warn"` nothing the
  platform raises escapes into your loop; `on_error="raise"` surfaces the first
  failure from `flush()` or `finish()`, for tests and CI. Platform errors are
  the `prime_traces` exception family.
- **Degrades.** A transient failure costs its batch, three in a row retire the
  sink (a training run pauses it for five minutes instead), and a full queue
  drops records rather than stalling the run. Losses are counted in
  `run.dropped_records` and `run.failed_records`.
- **Drains on exit.** `finish()` gives queued uploads up to `finish_timeout`
  (300 s) before closing the run out; an abort path can pass
  `finish(timeout=...)`.

An online run writes to Prime Traces (the system of record, gated to an
allowlist; outside it that sink turns itself off quietly) and to the sample
table today's viewer reads. `log_*()` are queue puts, safe inside a coroutine;
`init()` and `finish()` do network I/O.

## Configuration

| Source                 | Meaning                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| `PRIME_API_KEY`        | Platform API token                                                     |
| `PRIME_TEAM_ID`        | Team context; required for training runs                               |
| `PRIME_API_BASE_URL`   | Platform API; defaults to `https://api.primeintellect.ai`              |
| `PRIME_FRONTEND_URL`   | Dashboard; defaults to `https://app.primeintellect.ai`                 |
| `PRIME_TRACES_URL`     | Prime Traces service, resolved by `prime-traces`                       |
| `PRIME_RUNS_MODE`      | `online` or `disabled`; unset means online when there is an API key    |
| `~/.prime/config.json` | Shared prime CLI config (`api_key`, `team_id`, `base_url`)             |

Precedence is `init()` argument → environment variable → config file. A missing
API key disables the run with a warning.

## Not yet available

- **Failed or cancelled evaluations on the dashboard.** The evaluations API has
  no producer-facing status endpoint yet, so the terminal state is recorded
  under `metadata.prime_runs` and the run keeps showing as running.

## Related packages

- [prime-traces](https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime-traces) — Prime Traces SDK
- [prime](https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime) — Prime CLI
- [prime-evals](https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime-evals) — Evals SDK
