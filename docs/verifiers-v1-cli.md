# Verifiers V1 CLI boundary

Prime uses Typer for command routing and treats the local Verifiers evaluator as a workspace
process. `prime eval run` forwards every argument unchanged and replaces itself with:

```text
<workspace-python> -c 'from verifiers.v1.cli.eval.main import main; main()' <argv...>
```

Verifiers parses the Pydantic configuration once, generates the run identity, executes the
evaluation, and writes its native artifacts. Prime does not probe, pre-resolve, or reinterpret
local evaluation options.

## Command ownership

- `prime eval run` is the native local Verifiers CLI. A V1 taskset is positional or selected
  with `--taskset.id`; a V0 environment is selected with `--id` and runs through Verifiers'
  V0 adapter.
- `prime eval submit` is the Prime-owned hosted V0 API command.
- `prime eval push` uploads a completed native or legacy run.
- Prime Lab discovers both native `config.toml` + Trace `results.jsonl` artifacts and legacy
  `metadata.json` + `results.jsonl` artifacts.

The Verifiers Prime plugin owns workspace interpreter selection and the other delegated command
mappings; it also exports the V1 eval and init modules for consumers once released. Prime contains
no fallback command map.

## Native artifacts

A normal local run writes `config.toml`, append-only Trace `results.jsonl`, and `eval.log`.
A dry run writes only `config.toml`. The native run directory name is the run ID; no additional
manifest or sidecar is required.

The Prime Evals samples endpoint does not yet accept schema-versioned Trace JSONL directly, so
`prime eval push` performs the lossy projection into the platform sample view at that API
boundary.
