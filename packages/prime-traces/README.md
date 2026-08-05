# Prime Traces SDK

Upload, query and export training, evaluation and inference traces through the
Prime Traces service.

## Features

- **Content-addressed uploads** - Batches are identified by the SHA-256 of
  their exact bytes, so interrupted uploads are safe to rerun and never store
  twice
- **Deterministic batching** - JSONL files are split at byte thresholds without
  rewriting a single line
- **Typed reads** - Cursor-paginated summaries over extracted columns, raw
  document retrieval, and streaming exports
- **Type-safe** - Full type hints and Pydantic models
- **No CLI dependencies** - Pure SDK, usable in producers and services

## Installation

```bash
uv add prime-traces
```

or with pip:

```bash
pip install prime-traces
```

## Quick Start

```python
from prime_traces import TracesClient, LineFormat

client = TracesClient()  # PRIME_API_KEY / ~/.prime/config.json

# One bare Verifiers trace per line:
receipts = client.upload_file("traces.jsonl", context={"source": "hosted_eval"})

# One complete episode per line (multi-agent runs):
receipts = client.upload_file(
    "episodes.jsonl",
    line_format=LineFormat.EPISODE,
    context={"source": "hosted_eval", "suite_commit": "a1f39c2"},
)
```

Uploads are content-addressed: each request is identified by the SHA-256 of its
exact uncompressed JSONL bytes and sent with an `Idempotency-Key`. Rerunning an
interrupted upload re-reads the file, reproduces the same bytes and keys, and
the service replays committed receipts without storing anything twice. A 400
rejection stops the upload with a bounded error code (`ErrorCode`); 429/503 and
gateway 502/504 are retried with the same bytes, honoring `Retry-After`.

## Query

```python
page = client.list(run_id="run_9f3k2m", reward_min=0.9, has_error=False)
for summary in page.items:
    print(summary.trace_id, summary.score)

for summary in client.iter(task_id="tb2-0187"):  # paginates for you
    ...

summary = client.get("8d3f1a2b...")
raw = client.get_raw("8d3f1a2b...")          # exact stored trace document
client.download_raw("8d3f1a2b...", "t.json")  # streamed, for large traces

client.delete("8d3f1a2b...")
client.delete_run("run_9f3k2m")

# Stream a filtered export to disk (same filter vocabulary as list):
client.export("high_reward.jsonl", run_id="run_9f3k2m", reward_min=0.9)
```

Episodes are read-only resources:

```python
for episode in client.list_episodes(run_id="run_9f3k2m").items:
    print(episode.episode_id, episode.outcome)

detail = client.get_episode(episode_id)   # + member aggregate under .traces
print(detail.error.type, detail.traces.trace_count)

client.list_episode_traces(episode_id)    # member trace summaries, paginated
```

Response shapes mirror the service's pinned models: pages are
`{items, next_cursor}`, a trace summary nests `model` / `score` / `execution`,
an episode nests `error` and (on point lookup) the member-trace aggregate
under `traces`, and unrecorded fields come back as `null`. The one still
provisional shape is `export()`'s filter parameters — the service route does
not declare them yet.

## Configuration

| Source                 | Meaning                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `PRIME_API_KEY`        | Platform API token (needs `traces:read` / `traces:write` scopes)                                                                            |
| `PRIME_TEAM_ID`        | Optional team context, sent as `X-Prime-Team-Id`                                                                                            |
| `PRIME_TRACES_URL`     | Base URL of the Prime Traces service; defaults to the platform API base URL. For the service's local compose stack: `http://localhost:8083` |
| `~/.prime/config.json` | Shared prime CLI config (`api_key`, `team_id`, `traces_url`)                                                                                |

## Not implemented yet (open v0 contract decisions)

- The exports _job_ API (`POST /traces/exports`, `GET /traces/exports/{job_id}`)
  — unimplemented in the service in v0 until export results have somewhere
  to land. The streaming `GET /traces/export` is what `export()` wraps.
- `/search` and free-text queries — deferred with the `trace_components`
  projection.
- The `environment_id` filters (traces and episodes) — pending a populated
  extracted column.
- Typed dot-path predicates (`traces.query`) — needs the server-side field
  registry.
- An async client — the other prime SDKs ship sync/async pairs, and the main
  producers (verifiers, prime-rl) are async; add once the sync surface
  settles rather than freezing a duplicated API now.

## Documentation

For detailed documentation, visit the
[Prime Traces SDK documentation](https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime-traces).

## Related Packages

- [prime](https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime) - Prime CLI (`prime traces ...` commands)
- [prime-sandboxes](https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime-sandboxes) - Sandboxes SDK
- [prime-evals](https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime-evals) - Evals SDK
