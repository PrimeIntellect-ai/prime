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
raw    = client.get_episode_raw(episode_id)  # stored envelope; traces = member IDs
members = client.list_episode_traces(episode_id, has_error=True)
```

`get_episode_raw` returns the envelope the producer uploaded (`env`, `task`,
`group`, `run`, `ok`, every entry of `errors`) with `traces` narrowed to member
trace IDs; read each member with `get_raw`. It needs a server that supports
`GET /api/v1/episodes/{episode_id}?raw=true` — an older one ignores the flag
and returns the summary JSON.

## Search indexed content

Case-sensitive literal search within one run. Requires SDK 0.0.5+ and a server
supporting `GET /api/v1/runs/{run_id}/search`.

```python
page = client.search("connection refused", run_id="run_9f3k2m", role="tool")
for match in page.items:
    print(match.trace_id, match.node_idx, match.excerpt)
```

```bash
prime traces search 'connection refused' --run-id run_9f3k2m --role tool -o json
```

Async uses the same API. Optional filters: `role`, `run_step`, `has_error`,
`reward_min`, `reward_max`, and `field` (`content`, `reasoning_content`, `tool_calls`).
String content is decoded; structured content and tool calls use recorded JSON.

Search uses server text indexes, so the query needs at least three characters.
Each call returns one page without downloading raw traces. Continue with
`cursor=page.next_cursor` (CLI: `--cursor`) and unchanged filters until null.
Cursors are node positions, not snapshots: a trace replaced mid-search resumes
at the same position in its new copy.

The first page's `coverage` reports what was not searchable: `unindexed_trace_ids`
(a bounded sample) and `partial_index`. It is null on later pages, and on a first
page when the server could not compute it within budget (unknown, not complete). Restart after
pending uploads finish indexing; nodes beyond the indexing cap remain excluded.

A search that overruns the server's read budget fails with `search_limit_exceeded`
(HTTP 400). Retrying unchanged will not help; use a more specific query or add filters.

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
| `PRIME_TRACES_URL`     | Base URL of the Prime Traces service; defaults to `https://prime-traces.pintel.dev` |
| `PRIME_TEAM_ID`        | Optional team context, sent as `X-Prime-Team-ID`                       |
| `~/.prime/config.json` | Shared prime CLI config (`api_key`, `team_id`, `traces_url`)           |

Precedence is constructor argument → environment variable → config file.

## Not yet available

- **Exports** — the service route exists but is unimplemented, so wrapping it
  would ship a method that cannot succeed.
- **Cross-run search, regex and arbitrary JSON-path queries.**

## Examples

The runnable [`basic_usage.py`](./examples/basic_usage.py) example covers
upload, query, downloads and error handling.

## Related packages

- [prime](https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime) — Prime CLI, including `prime traces`
- [prime-sandboxes](https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime-sandboxes) — Sandboxes SDK
- [prime-evals](https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime-evals) — Evals SDK
