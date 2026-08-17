# Surface notes

Open questions about the user-facing surface, collected while writing and
running the examples in this directory against the deployed dev service. Each
one is a thing a reader tripped on or had to explain — not a bug. The SDK and
CLI behaved correctly throughout.

Ordered by how much they shape a first impression.

---

## 1. `TracesClient()` can 403 on a correct token

Zero-argument construction inherits `team_id` from `~/.prime/config.json`, and a
team **replaces** the owner (`owner_id = team_id if team_id else user_id`).
During the private beta, a user whose personal ID is allowlisted but whose team
is not gets `403 service_not_enabled` from the first line of every example.

The message is accurate but the situation is invisible: nothing the user typed
mentioned a team.

**Worth deciding:** should the 403 name the owner it evaluated? "Prime Traces is
not enabled for `<team>`" turns a mystery into an instruction. Same for the CLI,
which knows the team name.

## 2. `PRIME_TEAM_ID=` means two different things

Verified in both:

| | empty `PRIME_TEAM_ID` |
|---|---|
| SDK `Config.team_id` | `""` → teamless, no header sent |
| CLI `Config.team_id` | ignored → **falls through to the config file** |

The CLI requires a non-empty value (`env_val is not None and env_val.strip()`);
the SDK accepts any set value. So the obvious way to say "just me" works in one
half of the product and silently does the opposite in the other.

**My read:** pick one. Empty-means-teamless is the more useful of the two, and
the CLI already has `prime switch personal` for the persistent case.

## 3. `team_id=""` as the force-teamless sentinel

`TracesClient(team_id="")` means "no team", `team_id=None` means "read the
config". An empty string carrying meaning distinct from `None` is easy to get
wrong — and `team_id=None` looks like it should mean "no team".

**Worth deciding:** an explicit `use_team=False`, or a sentinel object, or
documenting this loudly. It mirrors `InferenceClient`, so changing it is a
cross-SDK decision rather than a local one.

## 4. Deleting something twice is an error

`delete()` raises `NotFoundError` on a repeat, deviating from the design docs,
which specify deletion as idempotent. Every caller whose intent is "make sure
this is gone" has to write:

```python
try:
    client.delete(trace_id)
except NotFoundError:
    pass
```

The docstring is candid about this. But the common intent needing a `try` is
worth a second look.

**Worth deciding:** idempotent 204, or keep 404 and ship the helper.

## 5. `created_at` is a hint that can cause a false 404

`delete(trace_id, created_at=...)` is a pure performance hint — except a hint
that matches no stored copy 404s even though the trace exists. An optional
argument whose only failure mode is "wrong answer if you get it slightly wrong"
is a sharp edge on an otherwise simple call.

**Worth deciding:** take it from the summary object instead of a bare string
(`delete(summary)`), so it cannot be wrong, or drop it from the public surface.

## 6. `list()` vs `iter()`

`iter()` is what most callers want in a loop; `list()` returns one page plus a
cursor. The names suggest the opposite weighting — `list` reads like "give me
the list", and it gives you 20 of them.

**Worth deciding:** nothing may need to change, but the default `limit=20`
silently truncating an unsuspecting `list()` caller is the failure this naming
invites.

## 7. Episodes are an enum argument, not a method

`upload_records(records, line_format=LineFormat.EPISODE)` vs an
`upload_episodes(episodes)`. One code path is a real virtue and the format
genuinely is a wire header — but the enum is less discoverable, and the two
inputs are different shapes with different validation.

**Worth deciding:** a thin `upload_episodes()` wrapper costs nothing and reads
better at the call site.

## 8. SDK and CLI filters have diverged

`prime traces list` is missing filters the SDK exposes: `environment_id`,
`model_provider`, `is_truncated`, and `context`. Nothing signals the gap — the
CLI just cannot express those queries.

**Worth deciding:** whether the CLI intentionally trails the SDK, or should be
generated from the same filter list so it cannot drift again.

## 9. `environment_id` is inert for bare traces

The column is only populated from an episode envelope's `env.id`. A bare trace
upload always stores `""`, so `list(environment_id=...)` silently returns
nothing for the flat path. The filter looks general and is not.

**Worth deciding:** populate it from the trace record too, or document it as
episode-only.

## 10. `APIError.code` is typed `str`, not `ErrorCode`

The module docstring says "producers are expected to branch on the code, not the
message" — but the attribute is `Optional[str]`, so no editor offers the enum,
and the natural `exc.code is ErrorCode.TRACE_NOT_FOUND` is **always False**
(`==` works; `ErrorCode` is a str-enum). We hit this writing example 4.

**My read:** type it `Optional[ErrorCode]`. Cheap, and it makes the documented
usage the discoverable one.

## 11. `delete_run` reports nothing

Returns `None`, so a caller cannot say "deleted 42 traces" or distinguish a run
that never existed from one already empty.

**Worth deciding:** return a count if the service can cheaply provide one.

## 12. Missing pieces, for completeness

- **No episode delete anywhere** (ENG-5239) — `delete_run` orphans episode rows
  permanently.
- **No async client.** Every sibling prime SDK ships a sync/async pair.
- **No `export`.** Deferred deliberately until the service contract exists.
- **No CLI episode subcommands**, though the SDK has three.
- **The CLI group is hidden** until production routing lands.

---

## What went well, and should not be traded away

Worth saying out loud, because a list of nitpicks reads more negative than the
surface deserves:

- **`upload_records` taking anything with `to_record()`** means producers hand
  over their own objects, and the SDK depends on neither verifiers nor prime-rl.
- **Idempotent replay** makes the recovery story for an interrupted upload "run
  it again", with no bookkeeping on the producer's side.
- **`on_batch`** gives progress reporting without the SDK owning a progress bar.
- **Summaries are honest projections** — a field the producer never recorded
  reads `None`, not an empty string you must know to special-case.
- **`AmbiguousDeleteError`** refuses to guess on the caller's behalf. It is the
  single best thing in the error hierarchy and justifies having one.
- **Two separate error signals on an episode** (`has_error` vs
  `any_trace_error`) preserve the case a rollup would erase.
