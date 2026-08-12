# Prime Traces SDK

Upload, query and export training, evaluation and inference traces through the
Prime Traces service.

## Features

- **Content-addressed uploads** - Batches are identified by the SHA-256 of
  their exact bytes, so interrupted uploads are safe to rerun and never store
  twice
- **Deterministic batching** - JSONL files are split at byte thresholds without
  rewriting a single line
- **Typed reads** - Cursor-paginated summaries over extracted columns and raw
  document retrieval
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

### Upload from memory

`upload_records` accepts JSON-compatible mappings as well as objects exposing
`to_record()`. Verifiers `Trace` / `Episode` and prime-rl `Rollout` objects
provide that method, so producers can upload completed records without writing
an intermediate JSONL file:

```python
from prime_traces import LineFormat, TracesClient

client = TracesClient()  # PRIME_API_KEY / ~/.prime/config.json

# Iterable[vf.Trace] or Iterable[prime_rl.orchestrator.types.Rollout]
receipts = client.upload_records(
    traces,
    context={"source": "prime-rl", "run_id": "run_9f3k2m"},
)

# Iterable[vf.Episode] for multi-agent runs
receipts = client.upload_records(
    episodes,
    line_format=LineFormat.EPISODE,
    context={"source": "verifiers"},
)
```

Records are serialized lazily and fed into bounded batches, so this neither
buffers the complete iterable nor round-trips through the filesystem. Callers
that already have encoded JSONL bytes can use `upload_lines` directly.

### Upload a completed JSONL file

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

client.delete("8d3f1a2b...")        # NotFoundError if the owner has no such trace
client.delete_run("run_9f3k2m")     # one mutation, synchronous, no job handle
```

Deletion is not a no-op on absent rows: the service checks existence first and
answers 404, so repeating a delete that already succeeded raises
`NotFoundError`. (The design docs specify it as idempotent; this tracks the
service as built.) Failures known to occur before delivery, 429 responses, and
service-coded 503 refusals are retried. Ambiguous response-path failures and
gateway 502/503/504 responses are surfaced as `AmbiguousDeleteError` without
replaying the deletion, because a retry could delete a trace written after the
first request.

Point reads and deletes currently reject trace IDs containing `/`. ASGI decodes
an encoded slash before matching the service's `/{trace_id}` route, so those IDs
cannot be addressed until the service accepts a path-valued route parameter.

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
under `traces`, and unrecorded fields come back as `null`.

## Configuration

| Source                 | Meaning                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `PRIME_API_KEY`        | Platform API token (needs `traces:read` / `traces:write` scopes)                                                                            |
| `PRIME_TEAM_ID`        | Optional team context, sent as `X-Prime-Team-ID`                                                                                            |
| `PRIME_TRACES_URL`     | Base URL of the Prime Traces service; defaults to the platform API base URL. For the service's local compose stack: `http://localhost:8083` |
| `~/.prime/config.json` | Shared prime CLI config (`api_key`, `team_id`, `traces_url`)                                                                                |

## Not implemented yet (open v0 contract decisions)

- Exports, in any form. The service publishes `GET /traces/export` and the two
  job routes, but all three handlers raise `NotImplementedError` — answered as
  500, not the 501 they document — and the streaming route declares no query
  parameters, so there is no filter vocabulary to bind to. Wrapping it now
  would ship a method that cannot succeed.
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
