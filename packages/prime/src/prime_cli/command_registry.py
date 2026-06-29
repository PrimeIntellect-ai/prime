"""Declarative Prime CLI command registry."""

from dataclasses import dataclass

_MODULE_ALIAS = {
    "eval": "evals",
    "secret": "secrets",
    "train": "rl",
}


@dataclass(frozen=True)
class Command:
    path: tuple[str, ...]
    summary: str
    section: str
    # Either ``run_attr`` (a Prime-owned command function taking a parsed config)
    # or ``verifiers`` (the name of a Verifiers CLI module the router execs in
    # its place). ``pre_exec`` optionally names a ``module`` callable receiving
    # the raw leaf argv that runs in-process before the Verifiers exec.
    run_attr: str | None = None
    verifiers: str | None = None
    pre_exec: str | None = None
    positionals: tuple[str, ...] = ()
    module_override: str | None = None

    @property
    def module(self) -> str:
        seg = self.module_override or _MODULE_ALIAS.get(self.path[0], self.path[0])
        return f"prime_cli.commands.{seg}"

    @property
    def config_attr(self) -> str | None:
        if self.verifiers is not None:
            return None
        return "".join(seg.title() for p in self.path for seg in p.split("-")) + "Config"


GROUPS: dict[tuple[str, ...], str] = {
    ("availability",): "Check GPU availability and pricing.",
    ("config",): "Configure the Prime CLI.",
    ("deployments",): "Manage adapter deployments.",
    ("disks",): "Manage persistent storage.",
    ("env",): "Manage Verifiers environments and Hub publishing.",
    ("env", "action"): "Manage environment actions.",
    ("env", "secret"): "Manage environment secrets.",
    ("env", "var"): "Manage environment variables.",
    ("env", "version"): "Manage environment versions.",
    ("eval",): "Run local evaluations or manage hosted evaluations.",
    ("gepa",): "Run GEPA prompt optimization with Verifiers.",
    ("images",): "Manage container images in the Prime registry.",
    ("inference",): "Use Prime Inference.",
    ("lab",): "Manage the Prime Lab workspace.",
    ("pods",): "Manage compute pods.",
    ("registry",): "Manage registry credentials and private images.",
    ("sandbox",): "Manage sandboxes.",
    ("secret",): "Manage global secrets.",
    ("teams",): "Inspect teams and memberships.",
    ("train",): "Launch and manage Hosted Training runs.",
    ("tunnel",): "Manage tunnels for exposing local services.",
}


COMMANDS: tuple[Command, ...] = (
    Command(("availability", "disks"), "List available disks", "Compute", run_attr="disks"),
    Command(
        ("availability", "gpu-types"), "List available GPU types", "Compute", run_attr="gpu_types"
    ),
    Command(("availability", "list"), "List available GPU resources", "Compute", run_attr="list"),
    Command(
        ("config", "delete"),
        "Delete a saved environment",
        "Account",
        positionals=("name",),
        run_attr="delete_env",
    ),
    Command(("config", "envs"), "List available environments", "Account", run_attr="list_envs"),
    Command(
        ("config", "remove-team-id"),
        "Remove team ID to use personal account",
        "Account",
        run_attr="remove_team_id",
    ),
    Command(("config", "reset"), "Reset configuration to defaults", "Account", run_attr="reset"),
    Command(
        ("config", "save"),
        "Save current config as environment (including API key)",
        "Account",
        positionals=("name",),
        run_attr="save_env",
    ),
    Command(
        ("config", "set-api-key"),
        "Set your API key (prompts securely if not provided)",
        "Account",
        positionals=("api_key",),
        run_attr="set_api_key",
    ),
    Command(
        ("config", "set-base-url"),
        "Set the API base URL (prompts if not provided)",
        "Account",
        positionals=("url",),
        run_attr="set_base_url",
    ),
    Command(
        ("config", "set-frontend-url"),
        "Set the frontend URL (prompts if not provided)",
        "Account",
        positionals=("url",),
        run_attr="set_frontend_url",
    ),
    Command(
        ("config", "set-inference-url"),
        "Set the inference URL (prompts if not provided)",
        "Account",
        positionals=("url",),
        run_attr="set_inference_url",
    ),
    Command(
        ("config", "set-share-resources-with-team"),
        "Set whether to automatically share new resources with all team members",
        "Account",
        positionals=("enabled",),
        run_attr="set_share_resources_with_team",
    ),
    Command(
        ("config", "set-ssh-key-path"),
        "Set the SSH private key path",
        "Account",
        positionals=("path",),
        run_attr="set_ssh_key_path",
    ),
    Command(
        ("config", "set-team-id"),
        "Set your team ID.",
        "Account",
        positionals=("team_id",),
        run_attr="set_team_id",
    ),
    Command(
        ("config", "use"),
        "Switch to a different environment",
        "Account",
        positionals=("env",),
        run_attr="use_environment",
    ),
    Command(("config", "view"), "View current configuration", "Account", run_attr="view"),
    Command(
        ("deployments", "create"),
        "Deploy a model for inference.",
        "Lab",
        positionals=("model_id",),
        run_attr="create_deployment",
    ),
    Command(
        ("deployments", "delete"),
        "Unload a model from inference.",
        "Lab",
        positionals=("model_id",),
        run_attr="delete_deployment",
    ),
    Command(
        ("deployments", "list"),
        "List adapters and their deployment status.",
        "Lab",
        run_attr="list_deployments",
    ),
    Command(("disks", "create"), "Create a new storage disk", "Compute", run_attr="create"),
    Command(
        ("disks", "get"),
        "Get detailed information about a specific disk",
        "Compute",
        positionals=("disk_id",),
        run_attr="get",
    ),
    Command(("disks", "list"), "List your persistent disks", "Compute", run_attr="list"),
    Command(
        ("disks", "terminate"),
        "Terminate a disk",
        "Compute",
        positionals=("disk_id",),
        run_attr="terminate",
    ),
    Command(
        ("disks", "update"),
        "Update a disk's name",
        "Compute",
        positionals=("disk_id",),
        run_attr="update",
    ),
    Command(
        ("env", "action", "list"),
        "List actions (CI jobs) for an environment.",
        "Lab",
        positionals=("environment",),
        run_attr="actions_list",
    ),
    Command(
        ("env", "action", "logs"),
        "Get logs for a specific action.",
        "Lab",
        positionals=("environment", "action_id"),
        run_attr="actions_logs",
    ),
    Command(
        ("env", "action", "retry"),
        "Retry an action (integration test) for an environment.",
        "Lab",
        positionals=("environment", "action_id"),
        run_attr="actions_retry",
    ),
    Command(
        ("env", "build"),
        "Build an OpenEnv-backed environment image.",
        "Lab",
        positionals=("env_id",),
        run_attr="build",
    ),
    Command(
        ("env", "delete"),
        "Delete an entire environment from the environments hub",
        "Lab",
        positionals=("env_id",),
        run_attr="delete",
    ),
    Command(
        ("env", "info"),
        "Show environment details and installation commands",
        "Lab",
        positionals=("env_id",),
        run_attr="info",
    ),
    Command(
        ("env", "init"),
        "Initialize a V1 or V0 environment with Verifiers.",
        "Lab",
        verifiers="init",
        pre_exec="init_preflight",
    ),
    Command(
        ("env", "inspect"),
        "Inspect environment source without downloading the archive locally.",
        "Lab",
        positionals=("env_id", "source_path"),
        run_attr="inspect_cmd",
    ),
    Command(
        ("env", "install"),
        "Install a verifiers environment.",
        "Lab",
        positionals=("env_ids",),
        run_attr="install",
    ),
    Command(("env", "list"), "List environments from the hub.", "Lab", run_attr="list_cmd"),
    Command(
        ("env", "pull"),
        "Pull environment for local inspection",
        "Lab",
        positionals=("env_id",),
        run_attr="pull",
    ),
    Command(
        ("env", "push"),
        "Push environment to registry",
        "Lab",
        positionals=("env_id",),
        run_attr="push",
    ),
    Command(
        ("env", "secret", "create"),
        "Create an environment-specific secret.",
        "Lab",
        positionals=("environment",),
        run_attr="env_secret_create",
    ),
    Command(
        ("env", "secret", "delete"),
        "Delete an environment-specific secret.",
        "Lab",
        positionals=("environment",),
        run_attr="env_secret_delete",
    ),
    Command(
        ("env", "secret", "link"),
        "Link a global secret to an environment.",
        "Lab",
        positionals=("global_secret_id", "environment"),
        run_attr="env_secret_link",
    ),
    Command(
        ("env", "secret", "list"),
        "List all secrets for an environment.",
        "Lab",
        positionals=("environment",),
        run_attr="env_secret_list",
    ),
    Command(
        ("env", "secret", "unlink"),
        "Unlink a global secret from an environment.",
        "Lab",
        positionals=("global_secret_id", "environment"),
        run_attr="env_secret_unlink",
    ),
    Command(
        ("env", "secret", "update"),
        "Update an environment-specific secret.",
        "Lab",
        positionals=("environment",),
        run_attr="env_secret_update",
    ),
    Command(
        ("env", "serve"),
        "Serve a V1 or V0 environment with Verifiers.",
        "Lab",
        verifiers="serve",
    ),
    Command(
        ("env", "status"),
        "Show action status for an environment.",
        "Lab",
        positionals=("env_id",),
        run_attr="status_cmd",
    ),
    Command(
        ("env", "uninstall"),
        "Uninstall an environment distribution with uv.",
        "Lab",
        positionals=("env_name",),
        run_attr="uninstall",
    ),
    Command(
        ("env", "validate"),
        "Run a taskset's model-free validation with Verifiers.",
        "Lab",
        verifiers="validate",
    ),
    Command(
        ("env", "var", "create"),
        "Create an environment variable.",
        "Lab",
        positionals=("environment",),
        run_attr="var_create",
    ),
    Command(
        ("env", "var", "delete"),
        "Delete an environment variable.",
        "Lab",
        positionals=("var_id", "environment"),
        run_attr="var_delete",
    ),
    Command(
        ("env", "var", "list"),
        "List all variables for an environment.",
        "Lab",
        positionals=("environment",),
        run_attr="var_list",
    ),
    Command(
        ("env", "var", "update"),
        "Update an environment variable.",
        "Lab",
        positionals=("var_id", "environment"),
        run_attr="var_update",
    ),
    Command(
        ("env", "version", "delete"),
        "Delete an environment version by content hash",
        "Lab",
        positionals=("env_id", "content_hash"),
        run_attr="delete_version",
    ),
    Command(
        ("env", "version", "list"),
        "List all versions of an environment",
        "Lab",
        positionals=("env_id",),
        run_attr="list_versions",
    ),
    Command(
        ("eval", "get"),
        "Show evaluation details.",
        "Lab",
        positionals=("eval_id",),
        run_attr="get_eval",
    ),
    Command(("eval", "list"), "List evaluations.", "Lab", run_attr="list_evals"),
    Command(
        ("eval", "logs"),
        "Get logs for a hosted evaluation.",
        "Lab",
        positionals=("eval_id",),
        run_attr="logs_cmd",
    ),
    Command(
        ("eval", "push"),
        "Push native or legacy evaluation data to Prime Evals.",
        "Lab",
        positionals=("config_path",),
        run_attr="push_eval",
    ),
    Command(
        ("eval", "run"),
        "Run a local V1 or V0 evaluation with Verifiers",
        "Lab",
        verifiers="eval",
    ),
    Command(("eval", "samples"), "", "Lab", positionals=("eval_id",), run_attr="get_samples"),
    Command(
        ("eval", "stop"),
        "Stop a running hosted evaluation.",
        "Lab",
        positionals=("eval_id",),
        run_attr="stop_cmd",
    ),
    Command(
        ("eval", "submit"),
        "Submit a hosted V0 evaluation",
        "Lab",
        positionals=("environment",),
        run_attr="submit_eval_cmd",
    ),
    Command(
        ("eval", "view"), "Launch the interactive evaluation viewer.", "Lab", run_attr="view_cmd"
    ),
    Command(
        ("feedback",), "Submit feedback about Prime Intellect.", "Account", run_attr="feedback"
    ),
    Command(
        ("fork",),
        "Fork a public environment into your Prime Intellect namespace.",
        "Lab",
        positionals=("environment",),
        run_attr="fork",
    ),
    Command(
        ("gepa", "run"),
        "Run Verifiers' native GEPA command.",
        "Lab",
        verifiers="gepa",
    ),
    Command(
        ("images", "delete"),
        "Delete an image from your registry.",
        "Compute",
        positionals=("image_reference",),
        run_attr="delete_image",
    ),
    Command(
        ("images", "list"),
        "List all images you've pushed to Prime Intellect registry.",
        "Compute",
        run_attr="list_images",
    ),
    Command(
        ("images", "publish"),
        "Make an image public so other Prime users can run it.",
        "Compute",
        positionals=("image_reference",),
        run_attr="publish_image",
    ),
    Command(
        ("images", "push"),
        "Build and push a Docker image to Prime Intellect registry.",
        "Compute",
        positionals=("image_reference",),
        run_attr="push_image",
    ),
    Command(
        ("images", "unpublish"),
        "Make a public image private again.",
        "Compute",
        positionals=("image_reference",),
        run_attr="unpublish_image",
    ),
    Command(
        ("inference", "chat"),
        "Send a one-shot chat message to a Prime Inference model.",
        "Compute",
        positionals=("model", "message"),
        run_attr="chat",
    ),
    Command(
        ("inference", "models"),
        "List available models from Prime Inference (/v1/models).",
        "Compute",
        run_attr="list_models",
    ),
    Command(("lab", "doctor"), "Check a Lab workspace.", "Lab", run_attr="doctor"),
    Command(("lab", "hygiene"), "Check cheap Lab git hygiene.", "Lab", run_attr="hygiene"),
    Command(("lab", "mcp"), "Run the Lab MCP server over stdio.", "Lab", run_attr="mcp"),
    Command(
        ("lab", "register-github"),
        "Write the GitHub workflow for Lab git hygiene.",
        "Lab",
        run_attr="register_github",
    ),
    Command(("lab", "setup"), "Set up a Lab workspace.", "Lab", run_attr="setup"),
    Command(
        ("lab", "sync"), "Refresh Lab skills and local agent guidance.", "Lab", run_attr="sync"
    ),
    Command(("lab", "view"), "Launch the interactive Lab viewer.", "Lab", run_attr="_launch_view"),
    Command(("login",), "Login to Prime Intellect", "Account", run_attr="login"),
    Command(("logout",), "Log out of Prime Intellect", "Account", run_attr="logout"),
    Command(
        ("pods", "create"),
        "Create a new pod with an interactive setup process",
        "Compute",
        run_attr="create",
    ),
    Command(
        ("pods", "history"),
        "List your pods history (terminated pods)",
        "Compute",
        run_attr="history",
    ),
    Command(("pods", "list"), "List your running pods", "Compute", run_attr="list"),
    Command(
        ("pods", "ssh"),
        "SSH / connect to a pod using configured SSH key",
        "Compute",
        positionals=("pod_id",),
        run_attr="connect",
    ),
    Command(
        ("pods", "status"),
        "Get detailed status of a specific pod",
        "Compute",
        positionals=("pod_id",),
        run_attr="status",
    ),
    Command(
        ("pods", "terminate"),
        "Terminate a pod",
        "Compute",
        positionals=("pod_id",),
        run_attr="terminate",
    ),
    Command(
        ("registry", "check-image"),
        "Verify that an image is accessible (optionally using registry credentials).",
        "Compute",
        positionals=("image",),
        run_attr="check_docker_image",
    ),
    Command(
        ("registry", "list"),
        "List registry credentials available to the current user.",
        "Compute",
        run_attr="list_registry_credentials",
    ),
    Command(
        ("sandbox", "create"),
        "Create a new sandbox",
        "Compute",
        positionals=("docker_image",),
        run_attr="create",
    ),
    Command(
        ("sandbox", "delete"),
        "Delete one or more sandboxes by ID, by label, or all sandboxes with --all",
        "Compute",
        positionals=("sandbox_ids",),
        run_attr="delete",
    ),
    Command(
        ("sandbox", "download"),
        "Download a file from a sandbox",
        "Compute",
        positionals=("sandbox_id", "remote_path", "local_file"),
        run_attr="download_file",
    ),
    Command(
        ("sandbox", "expose"),
        "Expose a port from a sandbox.",
        "Compute",
        positionals=("sandbox_id", "port"),
        run_attr="expose_port",
    ),
    Command(
        ("sandbox", "get"),
        "Get detailed information about a specific sandbox",
        "Compute",
        positionals=("sandbox_id",),
        run_attr="get",
    ),
    Command(("sandbox", "list"), "List your sandboxes", "Compute", run_attr="list_sandboxes_cmd"),
    Command(
        ("sandbox", "list-ports"),
        "List exposed ports for a sandbox, or all sandboxes if no ID is provided",
        "Compute",
        positionals=("sandbox_id",),
        run_attr="list_ports",
    ),
    Command(
        ("sandbox", "logs"),
        "Get logs from a sandbox",
        "Compute",
        positionals=("sandbox_id",),
        run_attr="logs",
    ),
    Command(
        ("sandbox", "reset-cache"),
        "Reset sandbox authentication cache",
        "Compute",
        run_attr="reset_cache",
    ),
    Command(
        ("sandbox", "run"),
        "Execute a command in a sandbox.",
        "Compute",
        positionals=("sandbox_id", "command"),
        run_attr="run",
    ),
    Command(
        ("sandbox", "ssh"),
        "Connect to a sandbox via SSH.",
        "Compute",
        positionals=("sandbox_id", "ssh_args"),
        run_attr="ssh_connect",
    ),
    Command(
        ("sandbox", "unexpose"),
        "Unexpose a port from a sandbox",
        "Compute",
        positionals=("sandbox_id", "exposure_id"),
        run_attr="unexpose_port",
    ),
    Command(
        ("sandbox", "upload"),
        "Upload a file to a sandbox",
        "Compute",
        positionals=("sandbox_id", "local_file", "remote_path"),
        run_attr="upload_file",
    ),
    Command(
        ("secret", "create"), "Create a new global secret.", "Account", run_attr="secret_create"
    ),
    Command(
        ("secret", "delete"),
        "Delete a global secret.",
        "Account",
        positionals=("secret_id",),
        run_attr="secret_delete",
    ),
    Command(
        ("secret", "get"),
        "Get details of a specific secret.",
        "Account",
        positionals=("secret_id",),
        run_attr="secret_get",
    ),
    Command(("secret", "list"), "List your global secrets.", "Account", run_attr="secret_list"),
    Command(
        ("secret", "update"),
        "Update an existing global secret.",
        "Account",
        positionals=("secret_id",),
        run_attr="secret_update",
    ),
    Command(
        ("switch",),
        "Switch between your personal account and team contexts",
        "Account",
        positionals=("target",),
        run_attr="switch",
    ),
    Command(
        ("teams", "list"), "List teams for the current user.", "Account", run_attr="list_teams"
    ),
    Command(("teams", "members"), "List members of a team.", "Account", run_attr="list_members"),
    Command(
        ("train", "checkpoints"),
        "List checkpoints for a run.",
        "Lab",
        positionals=("run_id",),
        run_attr="list_checkpoints",
    ),
    Command(
        ("train", "components"),
        "List pods (orchestrator + env-servers) for a run.",
        "Lab",
        positionals=("run_id",),
        run_attr="list_components",
    ),
    Command(
        ("train", "configs"),
        "List available configuration options for Hosted Training.",
        "Lab",
        run_attr="list_configs",
    ),
    Command(
        ("train", "delete"), "Delete a run.", "Lab", positionals=("run_id",), run_attr="delete_run"
    ),
    Command(
        ("train", "distributions"),
        "Get reward/advantage distribution histogram for a run.",
        "Lab",
        positionals=("run_id",),
        run_attr="get_distributions",
    ),
    Command(
        ("train", "get"),
        "Get details of a specific run.",
        "Lab",
        positionals=("run_id",),
        run_attr="get_run",
    ),
    Command(
        ("train", "init"),
        "Generate a template config file for a Hosted Training run.",
        "Lab",
        positionals=("output_path",),
        run_attr="init_config",
    ),
    Command(("train", "list"), "List your runs.", "Lab", run_attr="list_runs"),
    Command(
        ("train", "logs"),
        "Get logs for a run.",
        "Lab",
        positionals=("run_id",),
        run_attr="get_logs",
    ),
    Command(
        ("train", "metrics"),
        "Get Hosted Training metrics for a run.",
        "Lab",
        positionals=("run_id",),
        run_attr="get_metrics",
    ),
    Command(
        ("train", "models"),
        "List available models for Hosted Training.",
        "Lab",
        run_attr="list_models",
    ),
    Command(
        ("train", "progress"),
        "Get progress information, including which steps have samples and distributions.",
        "Lab",
        positionals=("run_id",),
        run_attr="get_progress",
    ),
    Command(
        ("train", "request"),
        "Request models for Hosted Training.",
        "Lab",
        run_attr="request_models",
    ),
    Command(
        ("train", "restart"),
        "Restart a running run from its latest checkpoint.",
        "Lab",
        positionals=("run_id",),
        run_attr="restart_run",
    ),
    Command(
        ("train", "rollouts"),
        "Get rollout samples for a run.",
        "Lab",
        positionals=("run_id",),
        run_attr="get_rollouts",
    ),
    Command(
        ("train", "run"),
        "Launch a Hosted Training run from a config file.",
        "Lab",
        positionals=("config_path",),
        run_attr="create_run",
    ),
    Command(("train", "stop"), "Stop a run.", "Lab", positionals=("run_id",), run_attr="stop_run"),
    Command(
        ("train", "usage"),
        "Show token usage and price for a single training run.",
        "Lab",
        positionals=("run_id",),
        run_attr="run_usage_command",
        module_override="usage",
    ),
    Command(("tunnel", "list"), "List active tunnels.", "Compute", run_attr="list_tunnels"),
    Command(
        ("tunnel", "start"),
        "Start a tunnel to expose a local port.",
        "Compute",
        run_attr="start_tunnel",
    ),
    Command(
        ("tunnel", "status"),
        "Get status of a specific tunnel.",
        "Compute",
        positionals=("tunnel_id",),
        run_attr="tunnel_status",
    ),
    Command(
        ("tunnel", "stop"),
        "Stop and delete one or more tunnels.",
        "Compute",
        positionals=("tunnel_ids",),
        run_attr="stop_tunnel",
    ),
    Command(
        ("upgrade",), "Upgrade the Prime CLI to the latest version", "Account", run_attr="upgrade"
    ),
    Command(
        ("wallet",),
        "Show wallet balance and most recent billing rows.",
        "Account",
        run_attr="wallet_command",
    ),
    Command(
        ("whoami",),
        "Show current authenticated user and update config",
        "Account",
        run_attr="whoami",
    ),
)


def command_map() -> dict[tuple[str, ...], Command]:
    commands = {command.path: command for command in COMMANDS}
    if len(commands) != len(COMMANDS):
        raise RuntimeError("duplicate Prime CLI command path")
    return commands
