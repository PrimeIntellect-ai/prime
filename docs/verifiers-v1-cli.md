# Verifiers V1 CLI boundary

Prime treats Verifiers as a workspace process, not a Python plugin. `prime eval run` validates
the workspace interpreter's protocol and then replaces itself with:

```text
<workspace-python> -m verifiers.v1.cli.eval.main run <argv...>
```

Prime forwards `<argv...>` unchanged for V1. Verifiers owns parsing, taskset/harness-specific
help, execution, logging, signals, and native run artifacts.

The compatibility routes remain available:

- `--save-results` / `-s` (and the associated `--skip-upload` / `--env-path` options) use the
  existing V0 local adapter and metadata artifact.
- `--hosted` uses the existing hosted parser and `/hosted-evaluations` payload, including V0 TOML
  configs, environment resolution, access flags, secrets, grouping, logs, and cancellation.
- Existing `push`, `get`, `stop`, and implicit `prime eval <target>` behavior remains available.
  The existing `view`/deprecated `tui` behavior is unchanged.

Prime therefore reserves the ambiguous short option `-s` for V0 compatibility; V1 callers use
the equivalent long-form `--shuffle` option shown by upstream help.

## Native V1 platform blockers

Prime does not expose a speculative V1 submit command until `POST /hosted-evaluations` accepts a
resolved V1 invocation. The existing `prime eval run ... --hosted` compatibility path remains
operational. A native worker must execute Verifiers protocol version 1, preserve its `run_id`,
and return `config.toml`, Trace `results.jsonl`, and `eval.log` for a normal run. A dry run returns
only `config.toml`.

The Prime Evals samples endpoint does not yet expose a schema-versioned Trace JSONL ingestion
operation. `prime eval push` therefore performs one lossy projection into the sample view at
that API boundary. A native endpoint should accept `trace_schema_version: 1` plus Trace JSONL and
associate it with the resolved config and run ID without requiring that projection.

Prime Lab discovers and reads both native config/Trace artifacts and V0
`metadata.json`/results artifacts. Its existing multi-eval config form continues to launch through
the compatible hosted route. The native run directory name is its run ID.
