# Verifiers V1 CLI boundary

Prime's registry router treats Verifiers-owned commands as workspace processes. It forwards
every argument unchanged and replaces itself with:

```text
<workspace-python> -m <verifiers-module> <argv...>
```

The Verifiers command registry supplies the module. Prime selects the workspace interpreter,
materializes the selected account as environment variables, and otherwise does not probe,
pre-resolve, rewrite paths, render help, or reinterpret native options.

Hub references are Prime-owned. Install `owner/name[@version]` with `prime env install`, then
pass the resulting local module name to `prime eval run`, `prime env validate`, or
`prime env serve`. Verifiers never reads Prime profile files or downloads packages.

## Command ownership

- `prime eval run` is the native local Verifiers CLI. A V1 taskset is positional or selected
  with `--taskset.id`; a V0 environment is selected with `--id` and runs through Verifiers'
  V0 adapter.
- `prime env init` scaffolds a V1 environment, or a V0 environment with `--v0`.
- `prime env validate` runs a V1 taskset's model-free validation hooks.
- `prime env serve` serves a V1 taskset, or a V0 environment selected with `--id`.
- `prime eval submit` is the Prime-owned hosted V0 API command.
- `prime eval push` uploads a completed native or legacy run.
- `prime env install`, `build`, `push`, and `pull` are Prime-owned package and platform workflows.
- `prime gepa run` remains a V0 adapter until GEPA has a V1 CLI.
- Prime Lab discovers both native `config.toml` + Trace `results.jsonl` artifacts and legacy
  `metadata.json` + `results.jsonl` artifacts.

## Native artifacts

A normal local run writes `config.toml`, append-only Trace `results.jsonl`, and `eval.log`.
A dry run writes only `config.toml`. The native run directory name is the run ID; no additional
manifest or sidecar is required.

The Prime Evals samples endpoint does not yet accept schema-versioned Trace JSONL directly, so
`prime eval push` performs the lossy projection into the platform sample view at that API
boundary.
