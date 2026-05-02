# Lab V1 Design

> Overnight implementation directive: continue working in incremental slices without stopping for review until the planned Lab TUI scope described here is fully implemented, tested, and reviewed, or until a real blocker requires user input.

## Summary

`prime lab` is the primary terminal interface for Lab. It brings local workspace state, authenticated platform data, environment source browsing, config editing, launch flows, rollout inspection, metrics, logs, and coding-agent help into one coherent research environment.

`prime lab view` remains an alias. `prime lab setup`, `prime lab sync`, and `prime lab doctor` are Prime-owned services that can run from the CLI and from the TUI.

Cross-repo platform/API and Verifiers cleanup dependencies are tracked in Linear as `WB-2`.

## Product Principles

- One golden path: every common workflow has one obvious command and one obvious TUI action.
- Local-first, platform-backed: the app paints local and cached state immediately, then hydrates authenticated platform state in the background.
- Deterministic before agentic: schema checks, deterministic remediation, generated diffs, and explicit launch previews come before coding-agent assistance.
- Agent-native where useful: coding agents explain, edit, summarize, and assist open-ended research work without replacing deterministic setup, validation, launch, or browsing.
- Source hash as identity: environment source identity is based on a deterministic content hash that excludes generated artifacts.
- Canonical config inside, nice config outside: the runtime works with normalized config while users edit compact TOML that reflects the recommended writing style.
- No dead panels: platform-only surfaces use real buttons or native data when available.
- Reusable widgets: action bars, filters, source browsers, Markdown/code viewers, config editors, logs, rollout viewers, palette tokens, and footer descriptors are shared across screens.

## Current Product Surface

### Shell And Navigation

- The TUI launches by default from `prime lab`.
- The global bottom status bar shows compact auth, team, workspace, and coding-agent connection state.
- Primary sections are Home, Environments, Training, Evaluations, and Agent.
- Footer actions come from screen bindings/action descriptors rather than handwritten footer text.
- Mouse, arrow keys, Enter, and Space are the main navigation primitives.
- `prime lab view` preserves the explicit viewer alias.

### Launch Surface And Home

Lab always opens on a sparse launch surface, then enters Home as the workspace control surface.

- Launch paint is intentionally sparse: Prime Intellect branding, Lab title, compact status, one animated visual, and quickstart actions.
- Launch quickstart actions are product flows, not navigation labels:
  - Explore Environments opens the merged local/platform environment browser.
  - Train Models opens a native training config editor seeded from a recommended template.
  - Run Evaluations opens a native hosted-evaluation config editor seeded from a recommended template.
  - Build with Agent opens the configured coding-agent command surface; if no agent is configured, it opens the embedded setup flow.
- Dense workspace rows stay hidden while platform state is loading unless the user is filtering or interacting.
- Active workspace and remembered inactive workspaces are loaded from global Lab state.
- Add, forget, and switch workspace actions are native TUI screens.
- Setup, sync, doctor, and coding-agent actions render as proper buttons when relevant.
- Setup Lab workspace runs inside the TUI without exiting to the terminal.
- Workspace switching changes the app context and command cwd for Lab actions. It does not mutate the parent shell cwd.

## Current User Journeys

### First Launch

```text
prime lab
  -> launch surface
  -> local workspace/cache status paints immediately
  -> platform sections hydrate in the background
  -> Enter opens Home, or a quickstart action opens a flow directly
```

The launch surface should stay visually calm even while data is still loading. It should never show selector-only controls such as Filter or More Rows, and quickstart buttons should feel like deliberate calls to action rather than a secondary nav menu.

### Build Environment

```text
Launch: Build Environment
  -> agent flow
  -> choose prompt template or write custom prompt
  -> agent inspects workspace and proposes/scaffolds environment source
  -> Lab shows created files, doctor checks, and optional eval config
  -> user runs local/hosted eval or pushes source
```

This flow is intentionally creative and agent-native. Deterministic checks still own validation: source hygiene, README/pyproject presence, ignored artifacts, config references, and push readiness.

### Run Evaluation

```text
Launch: Run Evaluation
  -> native eval config editor
  -> choose template or custom environment/model values
  -> validate and preview nice TOML
  -> launch hosted eval
  -> stream logs/results
  -> open Data viewer for rollouts
```

The next improvement is a first-class template picker before the editor. The current implementation opens the editor with a recommended starter config.

### Launch Training

```text
Launch: Launch Training
  -> native training config editor
  -> choose template or custom model/environment values
  -> validate and preview nice TOML
  -> launch training
  -> follow new-run logs after the CLI prints the run id
  -> open Training run page for metrics, logs, config, and rollout data
```

Every training config display should expose a small Modify and Run action. Training selectors, run overview, environment pages, and project timelines should all converge on this same editor/launch surface.

### Explore Environments

```text
Launch: Explore Environments
  -> merged local/platform environment selector
  -> open Hub-like environment page
  -> browse README/source/actions
  -> create eval/training config from the environment
  -> install, refresh source, or view platform
```

The selector and detail sidebar should stay identical for local-only, platform-only, and merged local/platform records except for source/badge differences.

### Agent Research Flow

```text
Agent
  -> full-screen chat surface
  -> user asks for a research action
  -> agent can request a typed UI widget
  -> user confirms/edits in native widget
  -> deterministic command/API executes
  -> result links back to the relevant Lab page
```

Example:

```text
"tweak the alphabet-sort run to use bigger batch size"
  -> agent resolves "alphabet-sort run" or asks for clarification
  -> Lab renders a mini run/config selector
  -> user selects source config/run
  -> Lab renders config edit widget with batch_size focused
  -> user confirms launch
  -> Lab streams logs and attaches the new run to the current project
```

Agent-composed widgets should be declarative data specs rendered by Lab, not arbitrary Textual objects emitted by the agent. The agent proposes intent, candidates, defaults, and validation notes; Lab owns the widgets, side effects, and confirmation gates.

### Chat Experience

The agent page should feel closer to the launch surface than to the browser pages.

Current direction:

- No permanent sidebar.
- A centered chat stage owns the page.
- A subtle animated backdrop strip stretches with the stage width, giving the surface identity without competing with content.
- The composer is an Enter-to-send command bar; visible controls stay minimal.
- Slash commands expose secondary actions such as agent switching and prompt starters without permanent chrome.
- Agent, workspace, transport, and connection state sit in ambient header/status text, not a separate control panel.
- Turns render like a refined terminal log: clear user/agent/system demarcation, preserved whitespace, markup-safe content, and enough structure for inline widgets.

Design options considered:

- Launch-like command surface: sparse hero prompt, large composer, recent/template actions below. Best for starting creative work.
- Terminal transcript: dense scrollback with soft separators and low chrome. Best for long-running agent sessions.
- Widget-first workbench: transcript plus inline config selectors, diff previews, launch cards, and result cards. Best for agent-directed Lab actions.
- Ambient research canvas: backdrop field remains visible behind/around cards. Best for brand feel, but must stay dim and low-frequency to avoid distracting from text.

Chosen V1 shape: a centered stage that starts as a launch-like command surface and evolves into a terminal transcript with inline widget cards. The backdrop should be a narrow dim strip, not a full-screen animation, once chat content exists.

### Projects

Projects represent a research effort rather than a single asset.

Project timeline events should include:

- environment source hash created or pushed
- config saved
- eval launched/completed
- training launched/completed
- run cloned or modified
- note/decision added
- agent action proposed/applied

Project tags should attach to configs, environment versions, evals, training runs, and actions. Explicit project membership takes precedence over tag inference. Tags are still useful for cross-cutting labels such as benchmark family, customer, model family, or experiment class.

The first project UI can be a simple timeline grouped by day with filters for Environments, Evaluations, Training, Configs, Notes, and Agent actions. Later, projects can back collections and shared research reports.

### Environments

Environments use a canonical merged record that combines local and platform state.

Merged sources:

- local `environments/*` directories
- local `pyproject.toml`
- local `README.md`
- local files
- `.prime/.env-metadata.json`
- platform Environments Hub list/detail/version/status/action payloads

Rows and sidebars show coherent status:

- `LOCAL` appears for workspace source.
- `PUBLIC` or `PRIVATE` appears for platform visibility.
- Local and platform records merge into one row when their identity matches.
- The selector details view avoids duplicate local/platform/status/install sections.
- Local source hashes are computed with the same generated-artifact ignore policy as environment publishing.

Environment detail screens provide:

- Hub-like header with name, owner, version, visibility, source state, and platform link.
- Code tab with a left-side file browser and compact action/about panel plus a right-side README or source preview.
- Markdown rendering for `.md` files by default.
- Raw toggle for Markdown files.
- Clickable buttons for Markdown and HTML `href` links.
- Version selection.
- Source cache under `~/.prime/lab/cache`.
- Install, train, evaluate, refresh-source, and platform actions as buttons.
- Leaderboard tab as a compact platform link surface unless native data is present.
- Discussion and action links live in the code tab action panel until native discussion/action data deserves a full screen.

### Training

Training prioritizes team/authenticated runs.

Selector behavior:

- Clicking a new run selects it and updates the sidebar.
- Enter opens the run screen.
- Clicking an already selected run opens it.
- Only one run is ready to expand at a time.
- Selected run detail and first metric rows are prefetched and cached.

Run screen:

- Overview shows progress, metrics or reward distribution, key metadata, environments, config summary, Edit config, and View on platform.
- Data uses the shared rollout/eval viewer.
- System shows progressively streamed pretty logs.
- Metrics reveal progressively from prefetched rows while the background loader pages more data.
- Logs reveal progressively from an initial tail while the default larger tail loads.
- Chart toggles keep axes scoped to the selected metric/distribution.
- Chart layout reserves a side column for config/context instead of forcing full-width plots.
- Rerun/config editing opens the native config editor.
- Launching a config opens a native follow screen that streams command output.

### Evaluations

Evaluations combine hosted/platform evals and local eval outputs.

Selector behavior:

- By-run and by-environment views.
- Live filter with reusable filter components.
- Local/platform badges.
- Selection details aligned with the Verifiers eval TUI.
- Raw JSON is not a default selector tab.

Data viewer:

- Shared rollout viewer for local evals and training rollouts.
- Prompt/completion history rendering.
- Separate system and user initial prompt sections.
- Task and state details.
- Sample list with reward/score coloring.
- Markup-safe arbitrary text rendering.
- XML-like content and `[word]` tokens render as user text, not Rich markup.
- User-visible newlines are preserved.

### Config Editing And Launch

The config editor is native TUI UI.

- Clone from local TOML config.
- Clone from environment action.
- Clone from training run config summary.
- Edit model, environments, max steps, rollouts per example, batch size, and max tokens.
- `seq_len` is visible as a read-only reference.
- Config save writes a copy under `.prime/lab/configs/<kind>/`.
- Preview shows validation, command, field diff, and nice TOML.
- Launch opens a native follow screen with terminal-like output.
- Training launches detect the emitted `prime rl logs <run-id> -f` hint and hand off to live run logs with retry/backoff while the run starts.
- Environment install also uses the native follow screen.

Canonical config handling uses three representations:

- User TOML: compact authored representation with recommended field order and omitted defaults.
- Canonical config: normalized internal object with defaults, resolved environment references, typed fields, and stable diffing.
- Launch payload: API/CLI command input generated from canonical config plus auth/team/workspace context.

### Agent

Lab owns a coding-agent runtime abstraction.

Supported targets:

- Codex
- OpenCode
- Pi Coding Agent
- Hermes Agent
- Claude Code as one-shot exec
- Custom command fallback as one-shot exec

Runtime behavior:

- Server-mode agents start automatically when Lab launches and an agent is configured.
- Connection state is visible in the global status bar.
- Codex app-server stdio and Hermes ACP stdio are native streaming chat transports.
- OpenCode ACP HTTP and Pi RPC currently start as server transports and use one-shot prompt execution as the chat fallback until native client contracts are wired.
- Claude Code and custom commands run as one-shot prompt-to-completion execution.
- Chat screens use the runtime abstraction instead of full terminal takeover.
- Agent sync installs Prime-owned skills/docs guidance into configured agent locations.
- The chat surface should move toward the launch-screen feel: more spacious, fewer sidebars, status/action chrome pushed to the edges, and interactive prompt/config/result cards inside the transcript.

Tested runtime contracts:

- One-shot exec agents receive a prompt and append prompt/completion messages without terminal takeover.
- Codex app-server stdio initializes, starts a thread, streams assistant deltas, and completes a turn.
- Codex app-server threads receive the Lab dynamic widget tool contract and can request `lab.render_widget` payloads for native choice/config/action widgets.
- Hermes ACP stdio initializes, creates a session, streams session updates, and completes a prompt.
- OpenCode ACP HTTP endpoint startup is detected and falls back to one-shot prompt execution until a native HTTP client is added.
- Pi RPC startup creates a workspace-scoped session directory and falls back to one-shot prompt execution until a native RPC client is added.
- A configured workspace agent auto-starts when the TUI launches and reports status in the global status bar.

Agent use cases:

- explain a failing workspace check
- propose a config edit
- edit environment source
- summarize rollout failures
- prepare a rerun plan
- apply deterministic remediation after user confirmation
- synchronize local agent skills with the active Lab workspace
- retrieve platform and Prime CLI docs from local context

## Agent-Composed Widget Contract

Agent-composed UI should use a small stable schema that Lab can render without trusting agent code.

Initial widget types:

- choice picker: list of typed candidates, single or multi-select
- config editor: config kind, canonical config payload, highlighted fields, read-only fields
- action preview: command/API call, side effects, validation output, confirm/cancel
- file patch summary: files, hunks, risk notes, open file buttons
- run launcher: source config/run, config diff, launch button, live logs link
- rollout insight: selected samples, failure categories, proposed next action

The agent may request a widget by returning structured JSON or MCP/tool-call payload. Lab validates the payload, renders a native widget, and routes the result back to the agent/runtime. The Textual UI remains a skin over typed domain objects so the same widget specs can be printed in tests, rendered in the TUI, or serialized into project timelines.

## Workspace, Auth, And Teams

Average user model:

- one personal account
- one or more team accounts
- several workspaces
- stable workspace/team pairings

Global state lives under `~/.prime/lab/`:

```text
~/.prime/lab/state.json
~/.prime/lab/workspaces.json
~/.prime/lab/cache/
```

State tracks remembered workspace paths, active workspace, inactive workspace paths, last selected auth profile, last selected team per workspace, agent preference per workspace, and cache metadata.

Auth tokens stay in the existing Prime auth/config locations. Lab state stores references, not token copies.

The status bar uses a compact shape:

```text
auth ok | PI Applied Research | ~/dev/verifiers | agent: codex connected
```

## Source Identity And Cache

V1 source identity uses `compute_content_hash()` from the environment publishing path.

The hash covers the environment source tree and excludes generated or irrelevant artifacts:

- `.git`
- hidden directories and hidden files unless explicitly whitelisted
- `.prime`
- `dist`
- `build`
- `outputs`
- `__pycache__`
- `*.pyc`
- `*.egg-info`
- common cache directories
- large generated artifacts

Global source cache:

```text
~/.prime/lab/cache/sources/<content_hash>/
  manifest.json
  files/
```

Environment pointers:

```text
~/.prime/lab/cache/environments/<owner>/<name>/<version>.json
~/.prime/lab/cache/environments/<owner>/<name>/<version_id>.json
```

The current implementation already stores environment source trees globally under owner/name/version and writes manifests for downloaded archives. The V1 cache contract keeps those stable owner/name/version pointers, but the source tree itself should become hash-addressed so the hash is the source of truth and duplicate versions can share one blob.

Workspace pins:

```text
<workspace>/.prime/lab/pins.json
```

The source cache is global because the same environment source can appear across workspaces and teams. Workspace pins stay local because each workspace can select a different active version.

## Cache Invariants

Lab caches are local-first and monotonic.

- Row caches hydrate selector pages on first paint and must never shrink just because a restart or refresh begins with a smaller request limit.
- Incoming successful rows update existing cached rows by stable identity, then keep older cached rows until the per-section cap is reached.
- Placeholder rows, loading rows, and failure rows never overwrite good durable cache entries.
- Detail caches are keyed by item identity and can refresh individual item payloads without clearing adjacent selector rows.
- Source caches never install packages to browse files; they safely extract source archives and ignore generated folders such as `dist`, `build`, `outputs`, `__pycache__`, and `.pyc`.
- Workspace memory is global and should load every remembered path, not only sibling directories.
- Current durable selector row cap is 1000 items per section. Higher interactive pagination should request more from the platform without corrupting first-paint cache.

## Load Performance

Current load behavior:

- `load_initial()` paints local workspace state, local environments, local evals, and durable cached platform rows before network/platform hydration completes.
- Background hydration currently ladders section limits up to the requested cap so the UI grows progressively instead of waiting for the largest request.
- Row/detail caches prevent selector pages from shrinking on restart or refresh when a smaller or failed request returns.

Known speed pressure points:

- Ladder hydration re-fetches whole sections at each limit. At a 1000-row cap this can mean repeated list calls for environments, training, and evaluations.
- Platform sections are loaded as one snapshot unit. Slow environments, training, or eval APIs can delay the full background snapshot step.
- Detail hydration is item-keyed, but some selector sidebars still need extra fetches before they feel fully populated.
- Source cache lookup is local and fast, but platform-only source downloads still depend on archive availability and network latency.

V1 performance fixes:

- Split background hydration into per-section workers so Training can finish without waiting for Environments or Evaluations.
- Add cursor/page APIs where platform supports them; avoid repeated full list requests for each ladder step.
- Cache per-section freshness timestamps and skip redundant background requests for fresh sections unless the user explicitly refreshes.
- Record load timings in a lightweight debug event log so slow startup can be attributed to local scanning, cache reads, auth/team resolution, or platform endpoints.
- Keep the launch surface resident while hydration continues, and only repaint status/counts when values actually change.

If the app still feels slow after those changes, the next blocker is likely API shape rather than Textual rendering.

## Setup, Doctor, And Sync

`prime lab setup` creates and records a Lab workspace.

It handles:

- workspace directory creation/validation
- `.prime` state
- template configs
- local env folders
- local eval output folders
- recommended `.gitignore`
- skill/setup assets from Verifiers
- coding-agent selection
- workspace health checks

`prime lab doctor` and `prime lab doctor --fix` run deterministic health checks:

- missing `.prime`
- missing configs/environments directories
- missing recommended `.gitignore` entries
- generated outputs inside environment source
- invalid TOML
- missing local README or pyproject metadata
- stale or missing templates/docs indexes
- unpinned environment versions in configs
- missing local environment references
- missing configured agent skills

Each issue has severity, explanation, and deterministic remediation when available.

`prime lab sync` refreshes Lab-owned assets:

- Prime-owned templates
- local agent guidance
- configured agent skills
- local platform docs index
- CLI docs index
- known workspace asset metadata

Sync reports drift before changing user-owned agent files.

## Git And Platform Sync

Lab nudges users toward aligned local source, GitHub, and platform state.

Workspace and environment checks flag:

- source hash differs from latest platform version
- local source has no corresponding platform version
- generated outputs inside source trees
- stale README/source links
- missing local metadata
- unpinned config environment references

Environment screens expose:

- Install
- Train
- Evaluate
- Refresh source
- View platform version
- Copy/source-friendly install command content

## Verifiers Command Migration

Recommended ownership:

| Verifiers command | Prime CLI target | Ownership |
| --- | --- | --- |
| `vf-setup` | `prime lab setup` | Prime CLI owns user-facing setup. Verifiers keeps compatibility shims. |
| `vf-init` | `prime env init` and Lab create-env flow | Prime CLI owns the scaffold path. Verifiers keeps library internals. |
| `vf-tui` | `prime eval tui` and Lab Data tab | Shared rollout viewer logic lives in reusable Prime CLI modules. |
| `vf-install` | `prime env install` | Prime CLI owns install UX. |
| `vf-build` | `prime env push/build` | Prime CLI owns publishing. Verifiers keeps low-level build helpers. |
| `vf-eval` | `prime eval run` plus Verifiers local runner | Prime CLI owns Lab workspace/platform UX. Verifiers remains the local library runner. |
| `vf-gepa` | `prime eval/gepa` wrapper | Prime CLI exposes it when Lab config/project flows need it. |
| `vf-rl`, `vf-train`, `vf-vllm` | Verifiers RL package | Kept outside Lab commands unless they become a clear Prime CLI flow. |

## Platform API Assessment

This pass inspected `~/dev/platform` backend and frontend routes. Read/write primitives exist for environments, training, hosted evals, logs, metrics, samples, and source downloads. The TUI now uses available CLI/API paths and records the stable platform contracts required for deeper native parity.

### Environments Hub

Useful existing surfaces:

- `GET /environmentshub/`
- `GET /environmentshub/{owner_name}/{env_name}/@{tag}`
- `GET /environmentshub/{owner_name}/{env_name}/versions`
- `GET /environmentshub/{owner_name}/{env_name}/status`
- `POST /environmentshub/resolve`
- `POST /environmentshub/lookup`
- create/finalize/wheel/source upload endpoints used by `prime env push`
- package/simple index and package download endpoints used by install/pull
- `GET /environmentshub/{owner_name}/{env_name}/actions`
- `GET /environmentshub/{owner_name}/{env_name}/actions/{action_id}/logs`
- `POST /environmentshub/{owner_name}/{env_name}/actions/retry`
- environment secrets and variables endpoints

Stable contracts to add on the platform side:

- Source manifest endpoint keyed by content hash.
- Single environment version payload with source archive hash, source file manifest, semantic version, version id, and wheel hash.
- Public REST API for discussions.
- Native leaderboard summary API for rows, histograms, filtering, and pagination.
- GitHub sync metadata for repository URL, branch, commit, source hash, and platform version linkage.
- Project/collection attachment metadata for environment versions and source hashes.

### Training

Useful existing surfaces:

- `GET /rft/models`
- `GET /rft/deployable-models`
- `GET /rft/runs`
- `GET /rft/runs/{run_id}`
- `POST /rft/runs`
- stop, restart, delete, bulk delete
- `GET /rft/runs/{run_id}/logs?tail_lines=N`
- `GET /rft/runs/{run_id}/logs/pretty?tail_lines=N`
- `GET /rft/runs/{run_id}/metrics`
- `GET /rft/runs/{run_id}/samples`
- `GET /rft/runs/{run_id}/progress`
- `GET /rft/runs/{run_id}/distributions`
- checkpoint and adapter endpoints
- external run monitoring ingestion endpoints

Stable contracts to add on the platform side:

- `POST /rft/runs/validate` or `POST /rft/runs/preview` returning normalized config, resolved environment versions, read-only `seq_len`, capacity/queue warnings, estimated resources, estimated tokens, and launch blockers.
- Cursor-based or streaming logs.
- Log source selectors for orchestrator, environment worker, evaluator, trainer, and related streams.
- Rerun/clone endpoint or schema helper returning canonical config.

### Evaluations

Useful existing surfaces:

- `POST /evaluations`
- `POST /evaluations/{evaluation_id}/samples`
- `POST /evaluations/{evaluation_id}/finalize`
- `GET /evaluations`
- `GET /evaluations/{evaluation_id}`
- `GET /evaluations/{evaluation_id}/samples`
- `DELETE /evaluations/{evaluation_id}`
- `POST /evaluations/bulk-delete`
- `GET /hosted-evaluations/models`
- `POST /hosted-evaluations`
- `PATCH /hosted-evaluations/{evaluation_id}/cancel`
- `GET /hosted-evaluations/{evaluation_id}/logs`
- environment evaluation routes for statistics, runs, results, and histograms

Stable contracts to add on the platform side:

- Hosted eval validation/preview endpoint mirroring training launch preview.
- Cursor-based or streaming hosted eval logs.
- Stable mapping from hosted evals to environment leaderboard rows.
- Rerun-with-edits endpoint returning canonical config and diffable defaults.

### Projects And Collections

No clear Lab project/collection API was found in the inspected routes.

Domain model:

- Projects are research effort timelines linking configs, source hashes, environment versions, evals, training runs, notes, and decisions.
- Collections are curated environment/eval sets independent of a single research timeline.

Stable contracts to add on the platform side:

- list/create/update/delete projects
- attach/detach run/eval/environment/config/source hash
- timeline events
- collection list/create/update/share

## Implementation Architecture

Layering:

```text
CLI entrypoints
  -> Lab services
  -> domain records and canonical configs
  -> data clients and caches
  -> Textual screens
  -> reusable widgets
```

Key modules:

- `prime_cli.lab_setup`: setup, sync, and doctor services callable from CLI and TUI.
- `prime_lab_view.cache`: row/detail/source/workspace cache.
- `prime_lab_view.environment_records`: local/platform environment merge layer.
- `prime_lab_view.config_screen`: native config editor and launch follower.
- `prime_lab_view.source_browser`: source tree and file rendering.
- `prime_lab_view.eval_screen`: shared rollout/eval viewer widgets.
- `prime_lab_view.agent_runtime`: server/exec adapter runtime.
- `prime_lab_view.palette`: shared color tokens and CSS helpers.

Widget rules:

- Widgets receive data and action descriptors, not hardcoded key labels.
- Widgets use palette tokens, not literal CSS colors.
- Arbitrary user/model text is sanitized before Rich rendering.
- Markdown views parse links into clickable buttons when possible.
- Platform links are buttons with human labels, not raw URL blobs.
- Config-like displays expose Edit, Save Copy, Run, or View Source when relevant.

## Code Smells And Refactor Notes

These are the active cleanup targets noticed while implementing the current slice:

- Launch action routing lived as string ids in `LaunchScreen` plus a dict in `PrimeLabView`; quickstart flow objects now live in `quickstart.py`, but the next cleanup is a typed `LabAction` registry shared by launch buttons, Home actions, footer labels, and agent widget requests.
- Config templates exist in environment actions, setup templates, and quickstart flows. They should be centralized behind canonical config factories with named templates and nice-TOML printers.
- Agent chat now uses a centered transcript plus command bar, but inline widgets are still textual summaries. The next cleanup is a typed transcript/card renderer that can mount real widget cards in response to `lab.render_widget`.
- Selector hydration is snapshot-oriented. A per-section data service would be easier to cache, time, test, and refresh independently.
- Some screens still reach into private-ish helpers across modules. Public domain helpers should live in `training_config`, `quickstart`, `source_browser`, `readme`, and future `actions` modules.
- Text-image screen captures are manually curated in docs. The renderer should become a deterministic test utility that can print any page/widget from typed state without launching the full app.
- Footer/action labels are better than before but still split between Textual bindings, screen-specific buttons, and generated action items. A single action descriptor should drive label, binding, click target, enabled state, and documentation.
- Home and Agent surfaces are converging on launch-screen visual language, while technical selectors keep dense browsing affordances.

## Test Coverage

Unit coverage includes:

- environment merge precedence and badge combinations
- source hash ignore policy
- source cache pathing and manifest writes
- safe extraction
- workspace memory and active/inactive transitions
- setup/sync/doctor services
- config parse/render/diff/launch helpers
- action/link extraction and markup escaping
- agent adapter and runtime modes
- durable row cache monotonicity across smaller refreshes

Textual coverage includes:

- `prime lab` default launch
- launch Home visual state and quickstart actions
- setup empty state
- inactive workspace selection
- environment row selection/opening
- environment source browser layout
- Markdown Raw toggle
- training click guard
- training config edit action
- run chart metric/distribution toggles
- rollout viewer prompt/message/state rendering
- config edit/save/launch preview
- agent connection status

Screen snapshot coverage is recorded in `packages/prime/docs/lab-tui-screen-captures.md`. These text-image captures validate launch composition, selector pages, environment detail layout, training detail, evaluation detail, config launch/edit flows, setup/workspace screens, and agent chat surfaces at terminal resolution.

CLI coverage includes:

- `prime lab`
- `prime lab view`
- `prime lab setup`
- `prime lab sync`
- `prime lab doctor`
- `prime env push` content hash behavior
- `prime env pull` safe source browsing path
