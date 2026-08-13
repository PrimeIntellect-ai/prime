# Prime Traces examples

Four Python scripts and one shell walkthrough, in the order a user meets the
product. They are written to be *read* as much as run — each one shows a single
job and stays quiet about everything else.

All of them create their own data and delete it afterwards, so they are safe to
run repeatedly. (One exception, noted in `03_episodes.py`: episode rows survive
`delete_run` and cannot be removed by any API — ENG-5239.)

| | | |
|---|---|---|
| `01_producer_upload.py` | Getting a finished run in | `upload_records`, `upload_file`, batch context, idempotent replay |
| `02_query_traces.py` | Finding and reading traces | `list`, filters, `iter`, manual cursors, `get`, `get_raw`, `download_raw` |
| `03_episodes.py` | Multi-agent work as one unit | `LineFormat.EPISODE`, `list_episodes`, `get_episode`, `list_episode_traces` |
| `04_deletes_and_errors.py` | Cleanup and failure | `delete`, `delete_run`, the typed exception vocabulary |
| `cli_walkthrough.sh` | The same journey, from a terminal | `prime traces upload\|list\|get\|delete` |

## Running them

```bash
export PRIME_API_KEY=...                                    # a token with the `traces` scope
export PRIME_TRACES_URL=https://dev-prime-traces.pintel.dev # dev deployment

cd packages/prime-traces
uv run python examples/01_producer_upload.py
./examples/cli_walkthrough.sh                                # needs jq
```

`_sample.py` holds the fixture records so the examples themselves contain only
SDK calls. A real producer never writes those by hand — verifiers `Trace` /
`Episode` and prime-rl `Rollout` objects already expose `to_record()`.

## Two things that will bite you first

**The private-beta gate is per *owner*, and a team replaces the owner.** The
service resolves `owner_id = team_id if team_id else user_id`. If your
configured team is not on the allowlist, a perfectly good token returns
`403 service_not_enabled` — which reads like a credentials problem and is not.

- SDK: `PRIME_TEAM_ID=` (empty) clears it, or pass `TracesClient(team_id="")`.
- CLI: an empty `PRIME_TEAM_ID` is **ignored** — it falls through to the config
  file. Use `prime switch personal`.

That the same variable behaves differently in the two is on the discussion list
in [`SURFACE_NOTES.md`](./SURFACE_NOTES.md).

**The service idles.** The first request after a quiet period can exceed the
gateway timeout and return a bare `504 upstream request timeout` with none of
the service's error shape. Retry once before believing it.

## What these are for

`SURFACE_NOTES.md` collects the open questions about the user-facing surface —
naming, symmetry between SDK and CLI, and the sharp edges these examples ran
into. The examples are the evidence; the notes are the agenda.
