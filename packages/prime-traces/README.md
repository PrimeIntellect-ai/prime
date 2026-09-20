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

## Search indexed content

Requires SDK 0.0.5 or later and a deployment with `GET /api/v1/traces/search`.
Search is a case-sensitive literal substring, scoped to one run:

```python
page = client.search("connection refused", run_id="run_9f3k2m", role="tool", limit=50)
for match in page.items:
    print(match.trace_id, match.node_idx, match.excerpt)

# One call searches one bounded page. An empty page can still have a cursor.
if page.next_cursor:
    next_page = client.search(
        "connection refused", run_id="run_9f3k2m", role="tool", limit=50,
        cursor=page.next_cursor,
    )
```

`AsyncTracesClient.search()` has the same parameters and response. Optional
filters are `field`, `role`, `run_step`, `has_error`, `reward_min`, and `reward_max`.
Choose `field="content"` (default), `"reasoning_content"`, or `"tool_calls"`.
Queries contain 1–256 characters of nonblank text; `limit` is 1–100.

The CLI exposes the same single-page contract:

```bash
prime traces search 'connection refused' --run-id run_9f3k2m --role tool
prime traces search 'connection refused' --run-id run_9f3k2m --role tool -o json
prime traces search 'connection refused' --run-id run_9f3k2m --role tool --cursor '<next_cursor>'
```

Each call examines a bounded window of the existing node index and returns
small excerpts without downloading raw traces. Stop only when `next_cursor`
is null; the SDK and CLI never automatically scan an entire run. Keep all
filters unchanged when resuming. Pages expose `scanned_nodes` and
`examined_traces` for progress. `unindexed_trace_ids` and `partial_index` indicate
incomplete coverage for that page, even if its cursor is null: restart after
pending uploads finish indexing. Nodes beyond the service's indexing cap are
not searchable. Cursors are not snapshots; changes during a scan can affect
results, and replacing the resumed trace requires restarting.

There is one result per matching node, at its first occurrence, ordered by
trace ID and node index. String content is JSON-decoded; structured content and
tool calls are searched in their recorded JSON representation. `representation`
is `text` or `json`. `match_start`, `match_end` (exclusive), and `excerpt_start`
are zero-based Unicode code-point offsets in that representation. The match's
coordinates within its excerpt are `match_start - excerpt_start` through
`match_end - excerpt_start`. Results identify their `upload_id` and `generation`.

This is literal content search, without regex, relevance ranking, arbitrary
JSON-path predicates, or cross-run search. An older server returns
`NotFoundError`; the CLI explains that the search API is required.

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
