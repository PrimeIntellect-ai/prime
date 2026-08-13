#!/usr/bin/env bash
# The same journey as the Python examples, from the terminal.
#
# The CLI is the surface people meet first and judge fastest, so this is worth
# reading as prose: does each command say what it does, and is the output
# something you would want to look at?
#
#   PRIME_API_KEY=... PRIME_TRACES_URL=https://dev-prime-traces.pintel.dev \
#       ./examples/cli_walkthrough.sh
#
# The command group is currently hidden (it does not appear in `prime --help`)
# until production routing lands, but it runs today for anyone who sets
# PRIME_TRACES_URL.

set -euo pipefail

RUN_ID="run_cli_$(date +%s)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

say() { printf '\n\033[1m── %s\033[0m\n' "$1"; }

# ---------------------------------------------------------------------------
say "Configuration"
# ---------------------------------------------------------------------------
# Precedence is PRIME_TRACES_URL > config file > the platform base URL. The
# override is stored per context, so `prime --context dev traces …` and
# `--context prod` talk to different deployments with different credentials.
#
#   prime config set-traces-url https://dev-prime-traces.pintel.dev
#   prime config set-traces-url -        # clear it, follow the base URL
#
prime config view | grep -i -E 'traces|base url' || true

# ---------------------------------------------------------------------------
say "Upload"
# ---------------------------------------------------------------------------
python3 - "$WORK/traces.jsonl" "$RUN_ID" <<'PY'
import json, sys, time, uuid
path, run_id = sys.argv[1], sys.argv[2]
now = time.time()
with open(path, "w") as fh:
    for i in range(6):
        fh.write(json.dumps({
            "version": 4, "id": f"tr_{uuid.uuid4().hex[:16]}",
            "run": {"id": run_id},
            "task": {"type": "ExampleTask", "data": {"name": f"example-{i:04d}"}},
            "agent": {"name": "solver", "config": {"model": "deepseek-v4-flash"}},
            "calls": [{"model": "deepseek-v4-flash", "usage": {"total_tokens": 1000 + i}}],
            "rewards": {"correctness": {"score": i / 5, "weight": 1.0}},
            "metrics": {}, "stop_condition": "done", "ok": True, "errors": [],
            "timing": {"start": now - 60 + i, "scoring": {"end": now - 47 + i}},
            "info": {},
        }) + "\n")
PY

# `--context key=value` is repeatable and travels with the batch. Progress is
# reported per batch, since a large file becomes several.
prime traces upload "$WORK/traces.jsonl" -c source=cli_walkthrough -c owner="$USER"

# Safe to rerun: batches are identified by the hash of their exact bytes, so an
# interrupted upload is recovered by running the same command again.
say "Upload again (idempotent replay — nothing is stored twice)"
prime traces upload "$WORK/traces.jsonl" -c source=cli_walkthrough -c owner="$USER"

# ---------------------------------------------------------------------------
say "List"
# ---------------------------------------------------------------------------
prime traces list --run-id "$RUN_ID"

say "List with filters"
prime traces list --run-id "$RUN_ID" --reward-min 0.6 --sort reward

# `--output json` makes every command pipeable. The epilog on each `--help`
# documents the JSON shape, so you can write a jq expression without guessing.
say "List as JSON, piped through jq"
prime traces list --run-id "$RUN_ID" --limit 3 --output json \
  | jq -r '.items[] | "\(.trace_id)  reward=\(.score.reward)"'

TRACE_ID="$(prime traces list --run-id "$RUN_ID" --limit 1 --output json | jq -r '.items[0].trace_id')"

# ---------------------------------------------------------------------------
say "Get one trace"
# ---------------------------------------------------------------------------
prime traces get "$TRACE_ID"

# `--raw` returns the exact stored document rather than the summary.
say "Get the raw document"
prime traces get "$TRACE_ID" --raw | head -c 200; echo '…'

# With `--dest` it streams to a file instead of stdout — the right shape for
# documents too large to want in a terminal.
say "Stream the raw document to a file"
prime traces get "$TRACE_ID" --raw --dest "$WORK/trace.json" --output json
wc -c < "$WORK/trace.json"

# ---------------------------------------------------------------------------
say "Delete"
# ---------------------------------------------------------------------------
# Exactly one of TRACE_ID or --run-id; passing both or neither exits 1.
prime traces delete "$TRACE_ID" --yes
prime traces list --run-id "$RUN_ID" --output json | jq '.items | length'

say "Delete the whole run"
prime traces delete --run-id "$RUN_ID" --yes
prime traces list --run-id "$RUN_ID" --output json | jq '.items | length'

# Deleting something already gone reports "not found" and exits 1 rather than
# succeeding silently.
say "Repeat the delete (expected to fail)"
prime traces delete "$TRACE_ID" --yes || echo "  exit $? — as documented"

printf '\n\033[1mdone\033[0m — %s cleaned up\n' "$RUN_ID"
