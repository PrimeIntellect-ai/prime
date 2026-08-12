# prime-traces

Prime Intellect Traces SDK — upload and query training, evaluation and
inference traces through the Prime Traces service.

## Install

```bash
pip install prime-traces
```

## Upload a completed JSONL file

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
service as built.) A delete whose response is lost in transit is still safe —
the retry's 404 is absorbed. The exception is a delete using the optional
`created_at` precision hint: its 404 may mean the hint did not match while the
trace still exists, so the SDK preserves that error rather than claiming a
successful deletion it cannot prove.

Response shapes mirror the service's pinned models: pages are
`{items, next_cursor}` and a summary nests `model` / `score` / `execution`,
with unrecorded fields as `null`.

## Configuration

| Source                 | Meaning                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `PRIME_API_KEY`        | Platform API token (needs `traces:read` / `traces:write` scopes)                                                                            |
| `PRIME_TEAM_ID`        | Optional team context, sent as `X-Prime-Team-ID`                                                                                            |
| `PRIME_TRACES_URL`     | Base URL of the Prime Traces service; defaults to the platform API base URL. For the service's local compose stack: `http://localhost:8083` |
| `~/.prime/config.json` | Shared prime CLI config (`api_key`, `team_id`, `traces_url`)                                                                                |

## Not implemented yet

Deferred to follow-up PRs once the service pins the corresponding responses:

- Exports, in any form. The service publishes `GET /traces/export` and the two
  job routes, but all three handlers raise `NotImplementedError` — answered as
  500, not the 501 they document — and the streaming route declares no query
  parameters, so there is no filter vocabulary to bind to.
- Episode reads (`GET /episodes[...]`) — read-only resources; episodes are
  written only as a side effect of episode-grouped uploads.
- `/search` and free-text queries — deferred with the `trace_components`
  projection.
- The `environment_id` list filter — pending its extracted column.
- Typed dot-path predicates (`traces.query`) — needs the server-side field
  registry.
- An async client — the other prime SDKs ship sync/async pairs, and the main
  producers (verifiers, prime-rl) are async; add once the sync surface
  settles rather than freezing a duplicated API now.
