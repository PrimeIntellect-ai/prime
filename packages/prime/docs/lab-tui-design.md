# Lab TUI V1 Design

This document is the reviewer brief for the `prime lab` terminal UI. It describes
the current V1 shape, the contracts that matter for review, and the validation
expected before merge.

## Status

Lab TUI is a V1 candidate. The product surface is implemented around a single
golden path:

- open `prime lab`;
- choose or continue from Welcome;
- browse Environments, Training, Evaluations, or Settings;
- use Agent only as a full-screen assistant surface with native Lab controls;
- launch jobs through editable native config cards, not copied shell commands.

Ship readiness means focused Lab tests are green, package checks are green, the
dev screen sketches match the current UI, and any untested integration limits are
called out plainly.

## Product Principles

- Local-first, platform-backed. Local workspace state and cached rows render
  immediately, then authenticated platform rows hydrate in the background.
- Deterministic before agentic. Setup, Sync, Doctor, cached loading, config
  validation, and launch runners are normal product code. Agents request native
  controls; they do not own validation or execution.
- One visible path per job. Training and eval launches flow through one config
  editor/launcher path, whether started from a row, config file, Welcome, or
  Agent.
- Terminal-native density. Primary pages optimize for scanning, selection, and
  keyboard movement. Welcome and Agent can be more spacious, but browser-style
  landing-page patterns are out of scope.
- Current-state docs only. Code comments and docs describe how Lab works now,
  not how it changed during the branch.

## Shell Model

The persistent shell has three panes:

- section nav: `Environments`, `Training`, `Evaluations`, `Settings`;
- selector pane: scoped rows, grouped trees, or Settings lists;
- inspector pane: details for the highlighted object.

Global actions:

- `w`: Welcome;
- `c`: Agent;
- `s`: Settings, hidden from the footer but available as a quick jump;
- `/`: filter the active selector;
- `Enter`: open, confirm, or enter the selected control;
- `Esc`: back from child screens or clear active transient UI.

The footer should stay compact and contextual. It should not list every arrow key
when the focused control makes arrow behavior obvious. Agent has no Welcome
shortcut in its footer; `Esc` is the back path there.

The top bar and status bar both expose active identity:

- auth indicator;
- team/user identity;
- Prime profile when known;
- active workspace path;
- configured agent status.

Warnings appear in the status bar as `1 warning` or `N warnings`. Hovering or
clicking the status bar opens a small popover with the full issue text and, in
the main shell, a Doctor hint.

## Navigation Rules

The navigation contract is designed to avoid "stuck" states:

- section tree highlights switch the active section immediately;
- `Enter` on a row opens the row;
- `Left` from a row returns to the nearest visible control, then section nav;
- `Right` from nav or toggles enters the next useful selector;
- `Tab` and `Shift+Tab` remain fallback pane traversal;
- Settings subcolumns use `Left`/`Right` to switch categories, `Up` to stay put,
  and `Down`/`Enter` to enter the category's row list;
- Evaluations grouped by environment use a real tree: `Left` moves to parent,
  `Right` expands or enters the next child, and `Enter` opens/toggles.

When a selector is empty, Lab renders one disabled-looking explanatory row rather
than an empty pane. Empty copy should point to the likely next action: clear the
filter, switch scope, sign in, or wait for loading.

## Data Scoping

Rows are scoped before filtering so filter choices do not expose hidden public
rows in the default account view.

Environments:

- default scope is `Account + local`;
- the public Hub list is available through the `Public` scope;
- local and platform records merge by environment identity when possible;
- source badges show local, account-owned/team-owned, and public state.

Training:

- always represents Hosted Training runs for the active account/team;
- rows include run id/name, date, status, model, and environment summary;
- detail prefetch loads metrics and rollout samples without blocking browsing;
- opening a run screen loads logs and metrics progressively.

Evaluations:

- default scope is hosted evals for the active account/team plus local eval
  outputs from the current workspace;
- `Hosted` and `Local` scopes are available when the user wants one source;
- `By run` and `By env` are views over the same scoped item set;
- hosted/local rows share the same row shape and badges;
- score details live in metadata and detail panes, not as an incompatible row
  status.

Settings:

- the fourth main section, not a hidden Welcome-only screen;
- groups workspaces, profiles, local assets, and setup actions;
- exposes Setup, Sync, and Doctor as native actions;
- active workspace/profile/team are visible in the header and row details.

## Agent Surface

Agent is a full-screen chat stage opened with `c` or the Agent row. It includes a
one-line warning:

> Experimental: Agent mode is experimental. Review generated configs and launch
> details before running jobs.

Agent UX rules:

- prompt placeholder names the selected agent and says `Enter to send`;
- `Shift+Enter` inserts a newline;
- `/` opens command actions;
- `?` inserts starter prompts;
- `@` inserts Lab references;
- empty `Up`/`Down` traverse prompt history;
- large pasted text is collapsed into a placeholder while preserving the full
  submitted payload.

Agent messages can request native widgets:

- `choose` for ambiguous decisions;
- `search_environments` for Hub environment lookup;
- `train_model` for resolved hosted-training launch fields;
- `edit_config` for eval/RL/GEPA config edits;
- `launch_run` for a selected config launch;
- `preview_action`, `show_patch`, and `inspect_rollouts` for side effects and
  review surfaces.

Agents receive the current Hosted Training model ids in dynamic tool schemas and
developer instructions. The `train_model` tool rejects unavailable model ids
before a widget is rendered. The launcher dropdown is generated from the same
account-scoped model list.

Training launches created from Agent write generated configs under
`.prime/lab/configs/rl/`, remove any agent-proposed run name from the TOML, and
use the platform-generated name. Once the launch output reveals a run id, Lab
opens the training run screen and enables `View run` on the chat card for return
navigation.

Supported agent paths are adapter-specific:

- Codex uses app-server dynamic tools;
- Claude, Cursor, Amp, OpenCode, Hermes, and Factory Droid use MCP or ACP
  surfaces where available;
- Pi uses a project-local extension;
- Factory Droid live validation can require a logged-in Factory account and
  credits, so static config validation is the expected fallback when that
  account is unavailable.

## Cache And Loading

`LabDataSource.load_initial()` renders local workspace data and cached platform
rows first. The background load then merges fresh platform rows into the same
sections.

Important guarantees:

- cached platform details enrich fresh rows without overriding fresh status;
- local envs and local evals stay visible when authenticated platform requests
  fail;
- cache keys include workspace, platform base URL, active profile, and team
  context;
- no-auth platform sections render auth placeholders while still showing local
  evals;
- warnings accumulate in the snapshot instead of replacing the page with a hard
  failure when partial data is still useful.

The current implementation uses laddered loads for larger platform lists:
5 -> 10 -> 20 -> ... -> requested limit. This gives fast first paint while still
settling on the requested row count.

## Robustness Expectations

Core flows should be stress-tested with these cases:

- app opens in a fresh, non-Lab directory and points the user at setup/doctor;
- app opens in a configured Lab workspace with cached rows and no network;
- authenticated platform sections fail independently without blanking local
  state;
- filters are applied inside the current scope only;
- Settings can be reached from section nav, `s`, Welcome, and status/header
  affordances;
- warning popovers render multiline validation errors;
- Agent widgets handle ambiguous choices, follow-up text, unavailable models,
  launch stop/unmount, and training run handoff;
- local eval viewer opens from both `By run` and `By env`;
- training run rows prefetch details without forcing a run screen open;
- all launch commands run with workspace cwd.

## Performance Notes

The largest V1 performance wins already in place are:

- local/cached first paint;
- background snapshot hydration;
- detail prefetch only for highlighted training rows;
- local eval metric loading only when selected;
- chat transcript shape diffing so streaming text does not remount every widget;
- bounded inline launch log retention.

Post-V1 cleanup should focus on section-specific loaders and freshness
timestamps, plus a small action registry that can feed launch buttons, Settings
actions, status hints, and Agent widgets from one typed source.

## Validation Checklist

Run these before marking the PR ready:

```bash
env UV_CACHE_DIR=/private/tmp/uv-cache uv run --project packages/prime prime lab --help
env UV_CACHE_DIR=/private/tmp/uv-cache uv run --project packages/prime prime lab doctor --help
env UV_CACHE_DIR=/private/tmp/uv-cache uv run --project packages/prime python packages/prime/dev/render_lab_screens.py --screen all --width 120
env UV_CACHE_DIR=/private/tmp/uv-cache uv run --project packages/prime pytest packages/prime/tests/test_lab_view.py -q
env UV_CACHE_DIR=/private/tmp/uv-cache uv run --project packages/prime pytest packages/prime/tests -q
env UV_CACHE_DIR=/private/tmp/uv-cache uv run --project packages/prime ruff check packages/prime/src/prime_lab_app packages/prime/tests/test_lab_view.py
env UV_CACHE_DIR=/private/tmp/uv-cache uv run --project packages/prime ruff format --check packages/prime/src/prime_lab_app packages/prime/tests/test_lab_view.py
git diff --check
```

For manual review, render the sketch file and compare it against a live
`prime lab` run in:

- a fresh directory;
- a configured Lab workspace;
- an offline or platform-warning state;
- an Agent session with at least one choice widget and one launch widget.

## V1 Limits

- Agent mode is intentionally labeled experimental.
- Factory Droid cannot be fully end-to-end validated without a logged-in account
  and credits.
- Public platform browsing depends on current API availability; local and cached
  rows remain the fallback.
- The screen sketch script is a visual aid, not a snapshot oracle.
