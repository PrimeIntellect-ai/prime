# Lab V1 Design

> Overnight implementation directive: continue working in incremental slices without stopping for review until the planned Lab TUI scope described here is fully implemented, tested, and reviewed, or until a real blocker requires user input.

## Summary

`prime lab` is the primary terminal interface for Lab. It brings local workspace state, authenticated platform data, environment source browsing, config editing, launch flows, rollout inspection, metrics, logs, and coding-agent help into one coherent research environment.

`prime lab view` remains an alias. `prime lab setup`, `prime lab sync`, and `prime lab doctor` are Prime-owned services that can run from the CLI and from the TUI.

Cross-repo platform/API and Verifiers cleanup dependencies are tracked in Linear as `WB-2`.

## V1 Readiness Snapshot

Current state: V1 candidate, ready for final user testing. The core surfaces are implemented, focused Lab tests are green, setup/doctor agree on the golden workspace shape, cache hydration preserves good local/cached rows through platform failures, and live native-agent diagnostics have been run on the configured machine.

Implemented and green in the current slice:

- `prime lab` launch surface, quickstart actions, workspace memory, setup/sync/doctor services, and `prime lab view` alias.
- Merged local/platform environment records, source cache, source browser, README rendering, local/platform badges, and environment actions.
- Training/evaluation selectors, durable row/detail cache hydration, config edit/run flows, launch command output, training log handoff, metrics/distribution toggles, and shared rollout viewer logic.
- Global chat sessions, prompt history, slash/help/reference menus, native Lab tool contracts, MCP/ACP wiring, unsupported-agent triage, and inline eval/training launcher cards.
- Dev-only text renderer under `packages/prime/dev/`; no generated screen artifacts are part of the runtime or test oracle.

Known post-V1 follow-up work:

- Live native-agent validation must be rerun against Amp, Codex, Claude Code, Cursor, Factory Droid, OpenCode, Hermes, and Pi whenever their adapter/tool contracts change. Unit tests cover contracts; they do not prove the current user machine's auth/tool wiring is healthy.
- Pi ACP remains a dead end for Lab controls because the current bridge does not forward MCP tools to the model. Lab supports Pi through a project-local Pi extension instead.
- Agent inline run-launcher cards have a logical control model, shared action preparation, shared launch execution, and a Textual card skin. Choice/config/action/patch/rollout cards should continue moving toward the same split.
- Config factories and nice-TOML printers should be centralized across quickstart flows, environment actions, config screens, and agent cards so every entry point renders the same recommended user-facing TOML.
- Background hydration should move from snapshot refreshes to per-section services with freshness timestamps and timings.
- Platform preview APIs for training/evaluation would remove local estimates and make launch validation deterministic.

## Product Principles

- One golden path: every common workflow has one obvious command and one obvious TUI action.
- Local-first, platform-backed: the app paints local and cached state immediately, then hydrates authenticated platform state in the background.
- Deterministic before agentic: schema checks, deterministic remediation, generated diffs, and explicit launch previews come before coding-agent assistance.
- Agent-native where useful: coding agents explain, edit, summarize, and assist open-ended research work without replacing deterministic setup, validation, launch, or browsing.
- Source hash as identity: environment source identity is based on a deterministic content hash that excludes generated artifacts.
- Canonical config inside, nice config outside: the runtime works with normalized config while users edit compact TOML that reflects the recommended writing style.
- No dead panels: platform-only surfaces use real buttons or native data when available.
- Reusable components: action bars, filters, source browsers, Markdown/code viewers, config editors, logs, rollout viewers, palette tokens, and footer descriptors are shared across screens.

## Current Product Surface

### Shell And Navigation

- The TUI launches by default from `prime lab`.
- The global bottom status bar shows compact auth, team, warnings, and coding-agent connection state.
- The active workspace path is a top-level clickable workspace affordance. It is the single global home for workspace location and opens workspace/profile/settings controls.
- Primary destinations are Environments, Training, Evaluations, and Settings. Agent is a global shell surface opened from the status bar or `c` binding.
- Section selectors are destinations only. Counts, loading state, auth state, and team context live in page subtitles, top chrome, bottom status, or explicit detail panels.
- Non-zero warnings are clickable in the bottom status bar and open a compact warning tray above the bar. The tray points users toward workspace settings and Doctor for deterministic checks and fixes.
- The bottom status bar is a compact shell control: auth/team, workspace, warnings, and agent state are displayed once. Non-zero warnings open the warning tray; otherwise the left side opens workspace/profile controls and the right side opens Agent.
- Footer actions come from screen bindings/action descriptors rather than handwritten footer text.
- Mouse, arrow keys, Enter, Esc, and Space are the main navigation primitives.
- `Esc` backs out of child/full-screen views. `Enter` opens, submits, or confirms. `/` activates filters or slash commands. Arrow keys move selection or tabs. `Space` is reserved for local toggles/pickers and explicit control actions such as chart selection or source parent navigation.
- `Ctrl+C` exits the app and is intentionally not advertised. Focused text inputs consume `Ctrl+C` to clear their value instead of closing Lab.
- `prime lab view` preserves the explicit viewer alias.

### Launch Surface And Workspace Controls

Lab always opens on a sparse Welcome surface. Workspace/profile/setup controls are available from the shell and contextual buttons rather than acting like the app's homepage.

- Launch paint is intentionally sparse: Prime Intellect branding, Lab title, compact status, one animated visual, and quickstart actions.
- Launch quickstart actions are product flows, not navigation labels:
  - Explore Environments opens the merged local/platform environment browser.
  - Train Models opens a native training config editor seeded from a recommended template.
  - Run Evaluations opens a native hosted-evaluation config editor seeded from a recommended template.
  - Build with Agent opens the configured coding-agent command surface; if no agent is configured, it opens the embedded setup flow.
- Footers show only primary navigation and the most important page action. Secondary accelerators like more rows/logs, platform open, copy, expand/collapse, and tab focus remain callable where useful but stay hidden behind page controls, slash commands, buttons, or future help overlays.
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
  -> Enter opens workspace controls, or a quickstart action opens a flow directly
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

The current implementation opens the editor with a recommended starter config. A richer template picker belongs in the next product slice once the canonical config factory is centralized.

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
  -> browse README/source
  -> create eval/training config from the environment
  -> sync local/platform source or open platform
```

The selector and detail sidebar should stay identical for local-only, platform-only, and merged local/platform records except for source/badge differences.

### Agent Research Flow

```text
Agent
  -> full-screen chat surface
  -> user asks for a research action
  -> agent can request a typed native control
  -> user confirms/edits in that control
  -> deterministic command/API executes
  -> result links back to the relevant Lab page
```

Example:

```text
"tweak the alphabet-sort run to use bigger batch size"
  -> agent resolves "alphabet-sort run" or asks for clarification
  -> Lab renders a mini run/config selector
  -> user selects source config/run
  -> Lab renders an inline config editor with batch_size focused
  -> user confirms launch
  -> Lab streams logs and attaches the new run to the current project
```

Agent-composed controls are declarative data specs rendered by Lab, not arbitrary Textual objects emitted by the agent. The agent proposes intent, candidates, defaults, and validation notes; Lab owns the controls, side effects, and confirmation gates.

Inline run-launcher cards are the golden path for agent-mediated eval and training launches. They prefill environment/model/run knobs from existing workspace configs when available, fall back to local environment metadata or Environments Hub ids, render model choices from `configs/endpoints.toml` before Prime Inference defaults, always save eval results, and stream launch output/logs in the chat card itself.

The transcript has a Lab-native ChatParts layer rather than relying on Textual message primitives. ChatParts are typed render intents such as text, Markdown, Lab references, and Lab-native actions. The same parts can be streamed into the TUI, printed in tests, serialized to session logs, and mounted as interactive control cards.

### Chat Experience

The agent page should feel closer to the launch surface than to the browser pages.

Current direction:

- No permanent sidebar.
- A centered chat stage owns the page.
- A subtle animated backdrop strip stretches with the stage width, giving the surface identity without competing with content.
- The composer is an Enter-to-send command bar; visible controls stay minimal.
- The composer is one line by default, expands upward only after explicit newlines, and collapses large paste payloads to a colored `[N lines pasted]` placeholder while preserving the submitted text.
- Slash commands expose secondary actions such as agent switching and prompt starters without permanent chrome.
- `?` as the first composer character opens prompt starters. `@` opens Lab references such as environments, configs, runs, evals, and files. Empty-composer Up/Down cycles recent user prompts from the current workspace+agent session.
- Agent, workspace, transport, and connection state sit in ambient header/status text, not a separate control panel.
- Turns render like a refined terminal log: purple rails with grayscale user text, green rails for agent output, compact system/error rows, preserved whitespace, markup-safe content, and enough structure for inline controls.

Design options considered:

- Launch-like command surface: sparse hero prompt, large composer, recent/template actions below. Best for starting creative work.
- Terminal transcript: dense scrollback with soft separators and low chrome. Best for long-running agent sessions.
- Control-first workbench: transcript plus inline config selectors, diff previews, launch cards, and result cards. Best for agent-directed Lab actions.
- Ambient research canvas: backdrop field remains visible behind/around cards. Best for brand feel, but must stay dim and low-frequency to avoid distracting from text.

Chosen V1 shape: a centered stage that starts as a launch-like command surface and evolves into a terminal transcript with inline control cards. The backdrop should be a narrow dim strip, not a full-screen animation, once chat content exists.

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
- The selector details view avoids duplicate local/platform/status/action sections.
- Local source hashes are computed with the same generated-artifact ignore policy as environment publishing.

Environment detail screens provide:

- Hub-like header with name, owner, version, visibility, source state, and platform link.
- Code tab with a left-side file browser and compact action/about panel plus a right-side README or source preview.
- Markdown rendering for `.md` files by default.
- Raw toggle for Markdown files.
- Clickable buttons for Markdown and HTML `href` links.
- Version selection.
- Source cache under `~/.prime/lab/cache`.
- Train, Evaluate, Sync, and Platform actions render as stacked buttons. Sync is the only local/platform source mutation button: it pulls platform-only environments into the workspace and pushes local owned environments to the platform.
- Platform discussion/action data should be surfaced through the single Platform button until native discussion/action data deserves a full screen.

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
- Environment sync uses the native follow screen.

Canonical config handling uses three representations:

- User TOML: compact authored representation with recommended field order and omitted defaults.
- Canonical config: normalized internal object with defaults, resolved environment references, typed fields, and stable diffing.
- Launch payload: API/CLI command input generated from canonical config plus auth/team/workspace context.

### Agent

Lab owns a coding-agent runtime abstraction.

Tool-backed Lab targets:

- Codex
- Amp Code via `--mcp-config` and the Prime Lab MCP bridge
- Claude via Claude Code and the Prime Lab MCP bridge
- Cursor via the Prime Lab MCP bridge
- Factory Droid Agent via project `.factory/mcp.json`
- OpenCode via ACP plus Prime Lab MCP tools
- Hermes Agent via ACP plus Prime Lab MCP tools
- Pi Coding Agent via project `.pi/extensions/prime-lab/index.ts`

Runtime behavior:

- The Agent Capability Registry owns each agent's label, binary requirements, setup repair commands, runtime transport, native tool surface, doctor status, and generated config paths.
- Server-mode agents start automatically when Lab launches and an agent is configured.
- Connection state is visible in the global status bar.
- Agent status is concise, such as `✓ Codex`; the status bar does not repeat `agent` or `connected`.
- Agent Client Protocol is Lab's preferred chat/session/event protocol for agents that expose it. ACP owns session lifecycle, prompt turns, assistant deltas, tool-call progress, slash-command announcements, cancel/close, and resumable session metadata.
- MCP remains Lab's native control/tool protocol. ACP sessions receive the Prime Lab MCP server through `session/new`, and MCP-backed headless agents receive generated native MCP config files.
- Codex app-server stdio is a native streaming chat transport with dynamic Lab tools.
- Amp receives a generated direct MCP server map through `--mcp-config` and streams Claude-compatible JSON output.
- Claude uses the Claude Code CLI path and receives a generated `--mcp-config` pointing to `prime lab mcp --workspace ...`; that stdio MCP server forwards tool calls into the running Lab TUI over workspace-scoped local IPC.
- Cursor receives a generated `.cursor/mcp.json` entry for `prime_lab` and runs headless with MCP and tool approvals enabled.
- OpenCode starts `opencode acp --cwd <workspace>` and receives Prime Lab MCP tools in the ACP session.
- Factory Droid receives a generated project `.factory/mcp.json` entry and runs headless from the selected workspace.
- Pi receives a generated project `.pi/extensions/prime-lab/index.ts` extension that registers Lab tools directly with Pi.
- Hermes starts `hermes acp --accept-hooks` in the selected workspace and receives Prime Lab MCP tools in the ACP session.
- Custom commands are not Lab-supported until they provide a native tool contract.
- Chat screens use the runtime abstraction instead of full terminal takeover.
- Agent sync installs Prime-owned skills/docs guidance into configured agent locations.
- The chat surface follows the launch-screen feel: spacious, no sidebar, status/action chrome at the edges, slash commands for secondary actions, and room for interactive prompt/config/result cards inside the transcript.
- Agent failures stay attached to the selected agent and render explicit errors. Lab does not silently switch to a different agent after a failure.
- Streaming assistant text is normalized at the adapter boundary before rendering. Full-message snapshots are deduplicated, token/delta streams append to one mutable assistant turn, and Markdown rendering happens from the current turn buffer rather than from independent partial blocks.

### Agent Sessions And Research Threads

Agent chat persistence is global and workspace-scoped.

- Lab sessions live under `~/.prime/lab/sessions/{workspace_hash}/{agent}/{session_id}/`.
- `session.json` stores workspace, agent, transport, native session id, endpoint, auth/team context, and timestamps.
- `transcript.jsonl` stores normalized user, assistant, and system messages.
- `actions.jsonl` stores Lab-native actions such as configs created, evals launched, training runs launched, source syncs, and deterministic remediation.
- `native/` stores pointers or symlinks to agent-native logs when available.
- Opening the Agent screen loads the latest compatible session for the selected workspace and agent.
- Switching agent starts or resumes that agent's own session; it never presents a model/agent switch as the same conversational identity.
- Native resume ids are passed back to transports that support them, while Lab's session id remains the stable local folder identity.
- `/clear` starts a fresh workspace+agent session and clears the visible transcript. Previous sessions remain durable and resumable through session history once that browser exists.
- The launch-style art clears once a conversation starts so the transcript can own the vertical space.

Research Projects are the future grouping layer above sessions. A project can tie together agent sessions, configs, environment source hashes, evals, training runs, notes, and team-shared research history. The current session/action metadata is shaped so projects can attach later without changing chat persistence.

Runtime contract validation matrix:

- Claude SDK chat runs through the Python SDK query stream, tracks session IDs, and renders streamed assistant text.
- Claude SDK sessions receive an in-process SDK MCP server exposing native Lab tools.
- Amp headless chat receives a generated direct Prime Lab MCP server map, streams JSON output, and forwards native MCP calls into the running Lab app.
- Claude Code headless chat receives a generated Prime Lab MCP config, streams JSON output, and forwards native MCP calls into the running Lab app.
- Cursor headless chat receives a generated workspace `.cursor/mcp.json` entry, streams JSON output, and forwards native MCP calls into the running Lab app.
- OpenCode ACP initializes, creates or resumes a session with Prime Lab MCP servers, streams `session/update` assistant chunks, and renders ACP tool-call updates as compact transcript events.
- Factory Droid discovers the generated project `.factory/mcp.json` file and lists Prime Lab MCP tools; full model-turn validation also requires a Factory account with Droid entitlement.
- Pi loads the generated project extension, registers Prime Lab tools with the model, streams JSON output, and forwards native tool calls into the running Lab app.
- Hermes ACP initializes, creates or resumes a session with Prime Lab MCP servers, and streams `session/update` assistant chunks through the same ACP normalizer.
- Codex app-server stdio initializes, starts a thread, streams assistant deltas, and completes a turn.
- Codex app-server threads receive the Lab dynamic tool contract and can request specific `lab.choose`, `lab.edit_config`, `lab.preview_action`, `lab.launch_run`, `lab.show_patch`, and `lab.inspect_rollouts` payloads.
- Agents without a native Lab tool surface are triaged as `not yet supported`; Lab does not start a process, inject fallback protocols, or silently route to another agent.
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

## Agent-Composed Control Contract

Agent-composed UI should use a small stable schema that Lab can render without trusting agent code.

Initial control types:

- choice picker: list of typed candidates, single or multi-select
- config editor: config kind, canonical config payload, highlighted fields, read-only fields
- action preview: command/API call, side effects, validation output, confirm/cancel
- file patch summary: files, hunks, risk notes, open file buttons
- run launcher: compact config fields, source config/run, launch button, inline command output and live logs
- rollout insight: selected samples, failure categories, proposed next action

Evaluation and training flows use the same native control sequence: `edit_config`, then
`preview_action`, then `launch_run`, with `config_kind = "eval"` or
`config_kind = "rl"`. The UI should not grow separate eval-only and
training-only interaction contracts.

The agent may request an interactive control only through a native tool or MCP call. Lab validates the payload, renders a native control, and routes the result back to the agent/runtime. The Textual UI remains a skin over typed domain objects so the same specs can be printed in tests, rendered in the TUI, or serialized into project timelines.

Recommended augmentation model:

- Native tool contracts are required for an agent to be considered Lab-supported. Lab injects tools such as `choose`, `edit_config`, and `launch_run` with JSON schemas, descriptions, and developer instructions at session start. The tool schema itself is the main context: it tells the model what action exists, when to use it, and what payload shape Lab can render.
- ACP-backed agents receive Lab tools as MCP servers at `session/new`. MCP-backed headless agents use the same bridge through native config files. In both cases, the running TUI owns a local workspace-scoped IPC socket, and `prime lab mcp` exposes stdio MCP tools that forward calls into that socket. This avoids pseudo-protocol parsing while keeping the UI process as the source of truth for control rendering and action logging.
- Skills are the portable guidance layer. `prime lab setup` and `prime lab sync` install the Prime-managed Lab controls skill so Amp, Codex, Cursor, Claude, Claude Code, Factory Droid, OpenCode, Hermes, and Pi share the same product guidance while their native tool APIs differ.
- `AGENTS.md` remains workspace policy: repo conventions, canonical commands, upload policy, and Prime research norms. It should not be the only source of Lab control semantics because it can drift per repo and does not provide callable schemas.
- Headless transports that do not expose custom tools are triaged as not yet supported for Lab-native chat actions. Lab should not ask agents to emit fenced JSON or other pseudo-tool protocols; the golden path is a native tool surface that Lab can validate and route.
- Agent output parsing must stay at the adapter boundary. Reasoning/thinking events are not transcript content; assistant-message deltas append to one mutable turn; final snapshots should complete that turn rather than creating a second visible answer.

Native control availability must be testable per agent:

- native tool transports receive a startup handshake listing supported Lab control schemas
- non-native transports show a precise not-yet-supported state and do not receive pseudo-protocol prompts
- `prime lab setup` and `prime lab sync` install the Prime-managed Lab controls skill into `.prime/skills`, mirror it to `~/.prime/lab/skills`, link it into the configured agent's skill folder, and write the agent's native Lab tool config
- accepted native control requests and user-triggered inline launches write `actions.jsonl` events with control id, kind, title, source/config path, command, status, and return code when available
- run-launcher requests mount compact inline config-and-launch cards in chat; the user sees the target once, tunes the visible values, and launches without leaving the transcript
- run-launcher cards use the same config-building and launch-runner services as full config pages; compact chat cards are visual skins over the same logical action, not a duplicate launch path
- choice/config/action/patch/rollout cards should follow the same logical-control/visual-skin split as the run launcher, with shared backend services rather than per-screen process logic
- `/diagnose` asks the active agent to render a no-op choice picker and records both the diagnostic start and any structured control request in the session action log

This keeps control interop strict: Amp, Codex, Claude, Claude Code, Cursor, Factory Droid, OpenCode, Hermes, and Pi are supported because each has a native Lab tool surface. One-shot custom commands remain unsupported until they provide a native tool contract.

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

The status bar uses a compact shape with colored indicators:

```text
✓ PI Applied Research · ~/dev/verifiers | ✓ Codex
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

The implementation keeps stable owner/name/version pointers and stores the source tree itself under the content hash, so source identity is hash-addressed and duplicate versions can share one blob. Downloads write pointer manifests that resolve to `sources/<content_hash>/files`.

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
- Cache per-section freshness timestamps and skip redundant background requests for fresh sections unless the cache is stale or the active context changes.
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

Doctor is the source of truth for Lab workspace and agent-readiness health. The TUI should not grow separate one-off checks for AGENTS.md, skills, docs, config drift, or agent guidance. Instead:

- launch performs a cheap cached doctor summary and surfaces non-zero WARN/FAIL counts in the warning tray
- Settings exposes explicit Doctor and Sync buttons with the same result model as the CLI
- chat startup can inject a compact readiness summary into the agent context, but user-facing remediation still routes through Doctor/Sync
- synced assets record source URL/ref/hash metadata under `.prime/lab/manifest.json`
- doctor compares local files, linked agent skill dirs, and manifest entries to flag missing or stale guidance
- sync updates Prime-owned assets only; user-authored AGENTS.md sections need explicit confirmation before overwrite

This keeps stale workspace guidance, outdated skills, missing docs, and agent setup issues on one golden path: Doctor diagnoses, Sync refreshes, Setup initializes.

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

- Train
- Evaluate
- Sync
- View platform version

`prime env pull` writes origin and fork-chain metadata in `.prime/.env-metadata.json`. A later push preserves the pulled origin while updating the active owner/name, so local forks retain their ancestry. This lets Lab explain whether Sync will pull from the platform, push to the active owner/team, or create a fork lineage.

Interactive code viewing is a likely Agent-side companion to this model, but it should not become a full editor until the Git/GitHub/EnvHub contract is explicit. The intended model is: source hash identifies platform environment versions; Git branch/commit links workspace source to reviewable code; Lab can show code state in a togglable agent sidebar or inline card; projects tie source, configs, evals, training runs, and agent sessions into a research timeline.

## Verifiers Command Migration

Recommended ownership:

| Verifiers command | Prime CLI target | Ownership |
| --- | --- | --- |
| `vf-setup` | `prime lab setup` | Prime CLI owns user-facing setup. Verifiers keeps thin aliases until those commands are removed. |
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
  -> reusable components
```

Key modules:

- `prime_cli.lab_setup`: setup, sync, and doctor services callable from CLI and TUI.
- `prime_lab_app.cache`: row/detail/source/workspace cache.
- `prime_lab_app.environment_records`: local/platform environment merge layer.
- `prime_lab_app.config_screen`: native config editor and launch follower.
- `prime_lab_app.toml_format`: shared nice-TOML formatting for config screens, quickstarts, environment actions, and chat cards.
- `prime_lab_app.source_browser`: source tree and file rendering.
- `prime_lab_app.eval_screen`: shared rollout/eval viewer components.
- `prime_lab_app.agent_capabilities`: supported-agent registry, machine requirements, setup commands, native surfaces, and doctor path expectations.
- `prime_lab_app.agent_acp`: ACP session params, MCP server injection, session capability parsing, and session/update normalization.
- `prime_lab_app.agent_runtime`: server/exec adapter runtime.
- `prime_lab_app.agent_cards`: inline chat control cards and compact config/launch skins shared by eval and training agent flows.
- `prime_lab_app.agent_mcp_bridge`: workspace-scoped IPC bridge and generated MCP config helpers.
- `prime_lab_app.palette`: shared color tokens and CSS helpers.

Component rules:

- Components receive data and action descriptors, not hardcoded key labels.
- Components use palette tokens, not literal CSS colors.
- Arbitrary user/model text is sanitized before Rich rendering.
- Markdown views parse links into clickable buttons when possible.
- Platform links are buttons with human labels, not raw URL blobs.
- Config-like displays expose Edit, Save Copy, Run, or View Source when relevant.

## Code Smells And Refactor Notes

These are the active cleanup targets noticed while implementing the current slice:

- Launch action routing lived as string ids in `LaunchScreen` plus a dict in `PrimeLabView`; quickstart flow objects now live in `quickstart.py`, but the next cleanup is a typed `LabAction` registry shared by launch buttons, Settings actions, footer labels, and agent control requests.
- Config templates exist in environment actions, setup templates, and quickstart flows. TOML presentation is now shared, but the next cleanup is centralizing the canonical config factories and named templates behind one service.
- Agent chat mounts inline launcher cards through `agent_cards` using `agent_widget_model` for payload normalization and `agent_widget_actions` for config construction, generated TOML, launch command preparation, and action logging. Run-launcher and config-editor requests already share the compact embedded config skin; the next cleanup is applying the same logical-control/service/skin split to choice, preview, patch, and rollout insight cards.
- Agent capability data is centralized and runtime startup consumes `AgentCapability` for unsupported triage. Remaining cleanup is moving native-surface preparation and dependency repair rules fully into the capability registry so runtime only starts declared transports.
- Selector hydration is snapshot-oriented. A per-section data service would be easier to cache, time, test, and refresh independently.
- Some screens still reach into private-ish helpers across modules. The agent control helpers and TOML formatting have public module surfaces now; remaining public domain helpers should live in `training_config`, `quickstart`, `source_browser`, `readme`, and future `actions` modules.
- Visual checks should use the dev-only stdout renderer under `packages/prime/dev/`; screen images are not a test oracle and should not leave generated artifacts in the repo.
- Footer/action labels are improving but still split between Textual bindings, screen-specific buttons, and generated action items. A single action descriptor should drive label, binding, click target, enabled state, and documentation.
- Welcome and Agent surfaces are converging on launch-screen visual language, while technical selectors keep dense browsing affordances.

## Test Coverage

Testing should stay surgical. Default suite coverage should protect contracts
that would create real product regressions: cache monotonicity, workspace
scoping, config generation, launch commands, agent tool routing, stream
deduplication, and a small number of critical Textual navigation paths. Visual
review is manual through the dev renderer; do not add screenshot-style tests for
every layout tweak.

Unit coverage includes:

- environment merge precedence and badge combinations
- source hash ignore policy
- source cache pathing and manifest writes
- cache-first hydration without rendering stale error placeholders over good cached rows
- safe extraction
- workspace memory and active/inactive transitions
- setup/sync/doctor services
- config parse/render/diff/launch helpers
- action/link extraction and markup escaping
- agent adapter and runtime modes
- agent session cache pathing, metadata, transcript writes, and per-agent separation
- chat composer newline expansion, large paste placeholder, and preserved submit payload
- durable row cache monotonicity across smaller refreshes

Textual coverage includes:

- `prime lab` default launch
- launch Welcome visual state and quickstart actions
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
- agent slash command selection and per-agent session switching

Manual visual checks should use the dev-only renderer:

```bash
uv run --project packages/prime python packages/prime/dev/render_lab_screens.py --screen all > /tmp/lab-screens.txt
```

The renderer prints lightweight terminal sketches from deterministic state. It is for quick product review only, not app runtime behavior and not a load-bearing test fixture.

Live native-agent validation should be run when changing adapter contracts:

- Codex app-server dynamic tools call `choose` and emit a Lab control action.
- Amp loads the generated direct MCP server map, calls `choose`, and reaches the running Lab IPC bridge.
- Claude headless chat loads the generated MCP config, has Lab MCP tools explicitly allowed, calls `mcp__prime_lab__choose`, and reaches the running Lab IPC bridge.
- Cursor headless chat loads the generated `.cursor/mcp.json`, runs with MCP approval enabled, calls `prime_lab-choose`, and reaches the running Lab IPC bridge.
- OpenCode loads workspace `opencode.json`, calls `prime_lab_choose`, and reaches the running Lab IPC bridge.
- Factory Droid loads workspace `.factory/mcp.json` and lists `mcp_prime_lab_*` tools; full live model-turn validation also requires a Factory account with Droid entitlement.
- Pi loads workspace `.pi/extensions/prime-lab/index.ts`, lists the Prime Lab extension tools, and reaches the running Lab IPC bridge.
- Hermes loads its generated MCP config, calls `choose`, and reaches the running Lab IPC bridge.

CLI coverage includes:

- `prime lab`
- `prime lab view`
- `prime lab setup`
- `prime lab sync`
- `prime lab doctor`
- `prime env push` content hash behavior
- `prime env pull` safe source browsing path
