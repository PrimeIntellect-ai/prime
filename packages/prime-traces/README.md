# Prime Traces SDK

Upload and query training, evaluation and inference traces through the Prime
Traces service.

> **⚠️ Prime Traces is in closed beta.** Access is granted per account, and the
> service is not yet on a public URL.

## Install

```bash
uv add prime-traces   # or: pip install prime-traces
```

## Upload

`upload_records` takes JSON-compatible mappings or any object exposing
`to_record()` — which verifiers `Trace` / `Episode` and prime-rl `Rollout`
already do, so producers can hand over their own objects. Records are
serialized lazily into bounded batches, so nothing buffers the whole run or
round-trips through disk.

```python
from prime_traces import LineFormat, TracesClient

client = TracesClient()  # PRIME_API_KEY / PRIME_TRACES_URL / ~/.prime/config.json

receipts = client.upload_records(traces, context={"source": "prime-rl"})

# One complete episode per record, for multi-agent runs
receipts = client.upload_records(episodes, line_format=LineFormat.EPISODE)

# Already have a completed JSONL file, or encoded bytes?
receipts = client.upload_file("traces.jsonl", context={"source": "hosted_eval"})
receipts = client.upload_lines(encoded_lines)
```

## Query

```python
page = client.list(run_id="run_9f3k2m", reward_min=0.9, has_error=False)
for summary in page.items:
    print(summary.trace_id, summary.score.reward)

for summary in client.iter(task_id="tb2-0187"):   # paginates for you
    ...

summary = client.get(trace_id)
raw     = client.get_raw(trace_id)                 # exact stored document
client.download_raw(trace_id, "trace.json")        # streamed, for large traces

client.delete(trace_id)
client.delete_run("run_9f3k2m")
```

Summaries are projections of the stored document, which is kept verbatim —
fields the producer never recorded come back as `None`. Deleting something that
is already gone raises `NotFoundError` rather than passing silently.

Episodes are read-only:

```python
page   = client.list_episodes(run_id="run_9f3k2m")
detail = client.get_episode(episode_id)      # + member aggregate under .traces
members = client.list_episode_traces(episode_id, has_error=True)
```

## Async

`AsyncTracesClient` mirrors `TracesClient` method for method.

```python
import asyncio
from prime_traces import AsyncTracesClient

async def main():
    async with AsyncTracesClient() as client:
        await client.upload_records(traces, context={"source": "prime-rl"})

        page, episodes = await asyncio.gather(   # reads overlap
            client.list(run_id="run_9f3k2m"),
            client.list_episodes(run_id="run_9f3k2m"),
        )

        async for summary in client.iter(task_id="tb2-0187"):
            ...

asyncio.run(main())
```

## Configuration

| Source                 | Meaning                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| `PRIME_API_KEY`        | Platform API token, needs `traces:read` / `traces:write` scopes        |
| `PRIME_TRACES_URL`     | Base URL of the Prime Traces service (**required** during closed beta) |
| `PRIME_TEAM_ID`        | Optional team context, sent as `X-Prime-Team-ID`                       |
| `~/.prime/config.json` | Shared prime CLI config (`api_key`, `team_id`, `traces_url`)           |
| `PRIME_CONFIG_DIR`     | Optional explicit config directory                                     |

Precedence is constructor argument → environment variable → config file. The
config file is the nearest trusted `.prime/config.json` at or above the working
directory (a project-local config from `prime login --local`, or one approved
with `prime config trust`), falling back to `~/.prime/config.json`.

## Not yet available

- **Exports** — the service route exists but is unimplemented, so wrapping it
  would ship a method that cannot succeed.
- **Search and free-text queries.**

## Examples

The runnable [`basic_usage.py`](./examples/basic_usage.py) example covers
upload, query, downloads and error handling.

## Related packages

- [prime](https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime) — Prime CLI, including `prime traces`
- [prime-sandboxes](https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime-sandboxes) — Sandboxes SDK
- [prime-evals](https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime-evals) — Evals SDK
