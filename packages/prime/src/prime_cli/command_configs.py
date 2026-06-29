"""Pydantic config models for Prime CLI commands."""

from __future__ import annotations

import pathlib

from pydantic import AliasChoices, Field, model_validator
from pydantic_config import BaseConfig


class AvailabilityDisksConfig(BaseConfig):
    """List available disks"""

    regions: list[str] | None = Field(None, description="Filter by regions (e.g., united_states)")
    data_center_id: str | None = Field(None, description="Filter by data center ID (e.g., US-1)")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class AvailabilityGpuTypesConfig(BaseConfig):
    """List available GPU types"""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class AvailabilityListConfig(BaseConfig):
    """List available GPU resources"""

    gpu_type: str | None = Field(None, description="GPU type (e.g., H100_80GB)")
    gpu_count: int | None = Field(None, description="Number of GPUs required")
    regions: list[str] | None = Field(None, description="Filter by regions (e.g., united_states)")
    socket: str | None = Field(None, description="Filter by socket type (e.g., PCIe, SXM5, SXM4)")
    provider: str | None = Field(None, description="Filter by provider (e.g., aws, azure, google)")
    disks: list[str] | None = Field(None, description="Filter by disk ids")
    group_similar: bool = Field(True, description="Group similar configurations from same provider")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class ConfigDeleteConfig(BaseConfig):
    """Delete a saved environment"""

    name: str = Field(..., description="Name of the saved environment")


class ConfigEnvsConfig(BaseConfig):
    """List available environments"""

    pass


class ConfigRemoveTeamIdConfig(BaseConfig):
    """Remove team ID to use personal account"""

    pass


class ConfigResetConfig(BaseConfig):
    """Reset configuration to defaults"""

    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class ConfigSaveConfig(BaseConfig):
    """Save current config as environment (including API key)"""

    name: str = Field(..., description="Name for the environment")


class ConfigSetApiKeyConfig(BaseConfig):
    """Set your API key (prompts securely if not provided)"""

    api_key: str | None = Field(
        None,
        description="Your Prime Intellect API key. If not provided, you'll be prompted securely.",
    )


class ConfigSetBaseUrlConfig(BaseConfig):
    """Set the API base URL (prompts if not provided)"""

    url: str | None = Field(
        None,
        description="Base URL for the Prime Intellect API. If not provided, you'll be prompted.",
    )


class ConfigSetFrontendUrlConfig(BaseConfig):
    """Set the frontend URL (prompts if not provided)"""

    url: str | None = Field(
        None,
        description="Prime Intellect web app URL. Prompts when omitted.",
    )


class ConfigSetInferenceUrlConfig(BaseConfig):
    """Set the inference URL (prompts if not provided)"""

    url: str | None = Field(
        None,
        description="Inference URL for Prime Inference API. If not provided, you'll be prompted.",
    )


class ConfigSetShareResourcesWithTeamConfig(BaseConfig):
    """Set whether to automatically share new resources with all team members"""

    enabled: str = Field(..., description="Enable or disable auto-sharing with team: true or false")


class ConfigSetSshKeyPathConfig(BaseConfig):
    """Set the SSH private key path"""

    path: str = Field(..., description="Path to your SSH private key file")


class ConfigSetTeamIdConfig(BaseConfig):
    """Set your team ID."""

    team_id: str = Field(..., description="Your Prime Intellect team ID.")


class ConfigUseConfig(BaseConfig):
    """Switch to a different environment"""

    env: str = Field(
        ..., description="Environment name: 'production' or a custom saved environment"
    )


class ConfigViewConfig(BaseConfig):
    """View current configuration"""

    pass


class DeploymentsCreateConfig(BaseConfig):
    """Deploy a model for inference."""

    model_id: str = Field(description="Model ID to deploy")
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class DeploymentsDeleteConfig(BaseConfig):
    """Unload a model from inference."""

    model_id: str = Field(description="Model ID to unload")


class DeploymentsListConfig(BaseConfig):
    """List adapters and their deployment status."""

    team: str | None = Field(
        None, validation_alias=AliasChoices("team", "t"), description="Filter by team ID"
    )
    num: int = Field(20, validation_alias=AliasChoices("num", "n"), description="Items per page")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class DisksCreateConfig(BaseConfig):
    """Create a new storage disk"""

    id: str | None = Field(None, description="Short ID from availability list")
    size: int = Field(..., description="Size of the disk in GB")
    name: str | None = Field(None, description="Name for the disk")
    country: str | None = Field(None, description="Country location")
    cloud_id: str | None = Field(None, description="Cloud ID from availability")
    data_center_id: str | None = Field(None, description="Data center ID")
    team_id: str | None = Field(
        None, description="Team ID to use for the disk (uses config team_id if not specified)"
    )
    provider_type: str | None = Field(None, description="Provider type (e.g., lambda, runpod)")
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class DisksGetConfig(BaseConfig):
    """Get detailed information about a specific disk"""

    disk_id: str = Field(...)
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class DisksListConfig(BaseConfig):
    """List your persistent disks"""

    limit: int = Field(100, description="Maximum number of disks to list")
    offset: int = Field(0, description="Number of disks to skip")
    watch: bool = Field(
        False,
        validation_alias=AliasChoices("watch", "w"),
        description="Watch disks list in real-time",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class DisksTerminateConfig(BaseConfig):
    """Terminate a disk"""

    disk_id: str = Field(...)
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class DisksUpdateConfig(BaseConfig):
    """Update a disk's name"""

    disk_id: str = Field(...)
    name: str = Field(..., description="New name for the disk")


class EvalGetConfig(BaseConfig):
    """Show evaluation details."""

    eval_id: str = Field(..., description="The ID of the evaluation to retrieve")
    output: str = Field(
        "json", validation_alias=AliasChoices("output", "o"), description="json|pretty"
    )


class EvalListConfig(BaseConfig):
    """List evaluations."""

    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )
    num: int = Field(20, validation_alias=AliasChoices("num", "n"), description="Items per page")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    env: str | None = Field(
        None,
        validation_alias=AliasChoices("env", "env_name", "e"),
        description="Filter by environment (e.g., 'gsm8k' or 'owner/gsm8k')",
    )


class EvalLogsConfig(BaseConfig):
    """Get logs for a hosted evaluation."""

    eval_id: str = Field(..., description="Evaluation id to get logs for")
    tail: int = Field(
        1000, validation_alias=AliasChoices("tail", "n"), description="Number of lines to show"
    )
    follow: bool = Field(
        False, validation_alias=AliasChoices("follow", "f"), description="Follow log output"
    )
    poll_interval: float = Field(5.0, description="Polling interval in seconds when following logs")


class EvalPushConfig(BaseConfig):
    """Push native or legacy evaluation data to Prime Evals."""

    config_path: str | None = Field(
        None,
        description="Native V1 or legacy evaluation run directory. Auto-discovers when omitted.",
    )
    env_id: str | None = Field(
        None,
        validation_alias=AliasChoices("env_id", "env", "e"),
        description="Published environment slug (owner/name).",
    )
    run_id: str | None = Field(
        None,
        validation_alias=AliasChoices("run_id", "r"),
        description="Link to existing training run id",
    )
    eval_id: str | None = Field(
        None,
        validation_alias=AliasChoices("eval_id", "eval"),
        description="Push to existing evaluation id",
    )
    name: str | None = Field(None, description="Explicit evaluation name override")
    output: str = Field(
        "pretty", validation_alias=AliasChoices("output", "o"), description="json|pretty"
    )
    is_public: bool = Field(
        False,
        validation_alias=AliasChoices("is_public", "public"),
        description="Make the pushed evaluation public. Evaluations are private by default.",
    )


class EvalSamplesConfig(BaseConfig):
    """"""

    eval_id: str = Field(..., description="The ID of the evaluation")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    num: int = Field(100, validation_alias=AliasChoices("num", "n"), description="Items per page")
    output: str = Field(
        "json", validation_alias=AliasChoices("output", "o"), description="json|pretty"
    )


class EvalStopConfig(BaseConfig):
    """Stop a running hosted evaluation."""

    eval_id: str = Field(..., description="Evaluation id to stop")


class EvalSubmitConfig(BaseConfig):
    """Submit a hosted V0 evaluation"""

    environment: str | None = Field(
        None, description="Environment name/slug or V0 TOML config path"
    )
    env_path: str | None = Field(
        None,
        description="Environment directory used for upstream resolution.",
    )
    poll_interval: float = Field(
        10.0, description="Polling interval in seconds for hosted evaluation status"
    )
    follow: bool = Field(
        False, description="Follow hosted evaluation status and stream logs until completion"
    )
    model: str | None = Field(
        None, validation_alias=AliasChoices("model", "m"), description="Inference model"
    )
    num_examples: int | None = Field(
        None,
        validation_alias=AliasChoices("num_examples", "n"),
        description="Examples per environment",
    )
    rollouts_per_example: int | None = Field(
        None,
        validation_alias=AliasChoices("rollouts_per_example", "r"),
        description="Rollouts per example",
    )
    env_args: str | None = Field(None, description="V0 load_environment arguments as JSON")
    extra_env_kwargs: str | None = Field(
        None, description="V0 post-load environment arguments as JSON"
    )
    timeout_minutes: int | None = Field(
        None, description="Timeout in minutes for hosted evaluation"
    )
    allow_sandbox_access: bool | None = Field(
        None, description="Allow sandbox read/write access for hosted evaluations"
    )
    allow_instances_access: bool | None = Field(
        None, description="Allow instance creation and management for hosted evaluations"
    )
    allow_tunnel_access: bool | None = Field(
        None, description="Allow tunnel creation and management for hosted evaluations"
    )
    custom_secrets: str | None = Field(
        None, description='Custom secrets for hosted eval as JSON (e.g. \'{"API_KEY":"xxx"}\')'
    )
    sampling_args: str | None = Field(
        None,
        description="Sampling arguments as JSON.",
    )
    eval_name: str | None = Field(None, description="Custom name for the hosted evaluation")
    max_concurrent: int | None = Field(None, description="Maximum concurrent rollouts")
    max_retries: int | None = Field(None, description="Retries per rollout")
    state_columns: list[str] | None = Field(
        None,
        description="State columns to retain.",
    )
    independent_scoring: bool | None = Field(None, description="Score rollouts independently")
    verbose: bool | None = Field(None, description="Enable verbose evaluator logs")
    header: list[str] | None = Field(
        None, description="Extra HTTP header as 'Name: Value'; repeat as needed"
    )
    api_client_type: str | None = Field(None, description="V0 model client type")
    api_base_url: str | None = Field(None, description="V0 model API base URL")
    api_key_var: str | None = Field(
        None, description="Environment variable containing the model API key"
    )


class EvalViewConfig(BaseConfig):
    """Launch the interactive evaluation viewer."""

    limit: int = Field(
        50, validation_alias=AliasChoices("limit", "n"), description="Max evaluation rows to load"
    )
    env_dir: str | None = Field(
        None,
        validation_alias=AliasChoices("env_dir", "e"),
        description="Path to environments directory",
    )
    outputs_dir: str | None = Field(
        None,
        validation_alias=AliasChoices("outputs_dir", "o"),
        description="Path to outputs directory",
    )


class FeedbackConfig(BaseConfig):
    """Submit feedback about Prime Intellect."""

    pass


class ForkConfig(BaseConfig):
    """Fork a public environment into your Prime Intellect namespace."""

    environment: str = Field(..., description="Public environment to fork, in owner/name format")
    team: str | None = Field(
        None,
        validation_alias=AliasChoices("team", "t"),
        description="Team slug to fork into (uses configured team ID if omitted)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class ImagesDeleteConfig(BaseConfig):
    """Delete an image from your registry."""

    image_reference: str = Field(
        ...,
        description="Personal or team image reference to delete.",
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class ImagesListConfig(BaseConfig):
    """List all images you've pushed to Prime Intellect registry."""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format (table or json)",
    )
    search: str | None = Field(
        None,
        validation_alias=AliasChoices("search", "q"),
        description="Case-insensitive substring match on image name, tag, or reference",
    )
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    num: int = Field(
        50, validation_alias=AliasChoices("num", "n"), description="Items per page (max 250)"
    )


class ImagesPublishConfig(BaseConfig):
    """Make an image public so other Prime users can run it."""

    image_reference: str = Field(
        ...,
        description="Personal or team image reference to make public.",
    )


class ImagesPushConfig(BaseConfig):
    """Build and push a Docker image to Prime Intellect registry."""

    image_reference: str = Field(
        ..., description="Image reference (e.g., 'myapp:v1.0.0' or 'myapp:latest')"
    )
    context: str = Field(
        ".", validation_alias=AliasChoices("context", "c"), description="Build context directory"
    )
    dockerfile: str | None = Field(
        None, validation_alias=AliasChoices("dockerfile", "f"), description="Path to Dockerfile"
    )
    platform: str = Field(
        "linux/amd64",
        description="Target platform (defaults to linux/amd64 for Kubernetes compatibility)",
    )
    public: bool = Field(False, description="Make the image public when the build completes")
    private: bool = Field(False, description="Make the image private when the build completes")


class ImagesUnpublishConfig(BaseConfig):
    """Make a public image private again."""

    image_reference: str = Field(
        ...,
        description="Personal or team image reference to make private.",
    )


class InferenceChatConfig(BaseConfig):
    """Send a one-shot chat message to a Prime Inference model."""

    model: str = Field(..., description="Model id (see `prime inference models`)")
    message: str | None = Field(None, description="User message. If omitted, reads from stdin.")
    system: str | None = Field(
        None, validation_alias=AliasChoices("system", "s"), description="System prompt"
    )
    stream: bool = Field(False, description="Stream tokens as they arrive")
    temperature: float | None = Field(
        None, validation_alias=AliasChoices("temperature", "t"), description="Sampling temperature"
    )
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")
    output: str = Field(
        "text", validation_alias=AliasChoices("output", "o"), description="text|json"
    )


class InferenceModelsConfig(BaseConfig):
    """List available models from Prime Inference (/v1/models)."""

    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )
    search: str | None = Field(
        None,
        validation_alias=AliasChoices("search", "q"),
        description="Case-insensitive substring match on model id",
    )
    sort: str = Field(
        "id", validation_alias=AliasChoices("sort", "s"), description="Sort by: id, input, output"
    )
    order: str = Field(
        "asc",
        validation_alias=AliasChoices("order", "d"),
        description="Sort order (direction): asc, desc",
    )


class LabDoctorConfig(BaseConfig):
    """Check a Lab workspace."""

    fix: bool = Field(False, description="Apply safe local remediations.")


class LabHygieneConfig(BaseConfig):
    """Check cheap Lab git hygiene."""

    fix: bool = Field(
        False, description="Apply safe local remediations such as dirs and gitignore entries."
    )


class LabMcpConfig(BaseConfig):
    """Run the Lab MCP server over stdio."""

    workspace: pathlib.Path | None = Field(
        None, description="Workspace whose running Lab TUI should receive MCP tool calls."
    )


class LabRegisterGithubConfig(BaseConfig):
    """Write the GitHub workflow for Lab git hygiene."""

    pass


class LabSetupConfig(BaseConfig):
    """Set up a Lab workspace."""

    skip_agents_md: bool = Field(
        False,
        description="Skip workspace agent guidance files.",
    )
    skip_install: bool = Field(
        False,
        description="Skip uv project initialization and Verifiers installation.",
    )
    agents: str | None = Field(
        None,
        validation_alias=AliasChoices("agents", "agent"),
        description="Comma-separated coding agents to configure, or 'all'.",
    )
    no_interactive: bool = Field(
        False,
        description="Use setup defaults without prompts.",
    )


class LabSyncConfig(BaseConfig):
    """Refresh Lab skills and local agent guidance."""

    agents: str | None = Field(
        None,
        validation_alias=AliasChoices("agents", "agent"),
        description="Comma-separated coding agents to configure, or 'all'.",
    )
    skip_docs: bool = Field(False, description="Skip workspace guidance refresh.")
    no_agent: bool = Field(
        False,
        description="Refresh shared assets without configuring agent skill roots.",
    )

    @model_validator(mode="after")
    def validate_agent_selection(self) -> "LabSyncConfig":
        if self.agents is not None and self.no_agent:
            raise ValueError("--agent and --no-agent cannot be used together")
        return self


class LabViewConfig(BaseConfig):
    """Launch the interactive Lab viewer."""

    limit: int = Field(1000, validation_alias=AliasChoices("limit", "n"))
    env_dir: str = Field("./environments")
    outputs_dir: str = Field("./outputs")


class LoginConfig(BaseConfig):
    """Login to Prime Intellect"""

    headless: bool = Field(False, description="Don't attempt to open browser")


class LogoutConfig(BaseConfig):
    """Log out of Prime Intellect"""

    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class PodsCreateConfig(BaseConfig):
    """Create a new pod with an interactive setup process"""

    id: str | None = Field(None, description="Short ID from availability list")
    cloud_id: str | None = Field(None, description="Cloud ID from cloud provider")
    gpu_type: str | None = Field(None, description="GPU type (e.g. A100, V100)")
    gpu_count: int | None = Field(None, description="Number of GPUs")
    name: str | None = Field(None, description="Name for the pod")
    disk_size: int | None = Field(None, description="Disk size in GB")
    vcpus: int | None = Field(None, description="Number of vCPUs")
    memory: int | None = Field(None, description="Memory in GB")
    image: str | None = Field(
        None, description="Image name or 'custom_template' when using custom template ID"
    )
    custom_template_id: str | None = Field(None, description="Custom template ID")
    team_id: str | None = Field(
        None, description="Team ID to use for the pod (uses config team_id if not specified)"
    )
    disks: list[str] | None = Field(
        None, description="Attach existing disk IDs to the pod. Repeat option for multiple disks."
    )
    env: list[str] | None = Field(
        None,
        description="Environment variables to set in the pod.",
    )
    share_with_team: bool = Field(False, description="Share the pod with all team members")
    add_members: bool = Field(
        False, description="Interactively select team members to share the pod with"
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class PodsHistoryConfig(BaseConfig):
    """List your pods history (terminated pods)"""

    limit: int = Field(100, description="Maximum number of history items to list")
    offset: int = Field(0, description="Number of history items to skip")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class PodsListConfig(BaseConfig):
    """List your running pods"""

    limit: int = Field(100, description="Maximum number of pods to list")
    offset: int = Field(0, description="Number of pods to skip")
    watch: bool = Field(
        False,
        validation_alias=AliasChoices("watch", "w"),
        description="Watch pods list in real-time",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class PodsSshConfig(BaseConfig):
    """SSH / connect to a pod using configured SSH key"""

    pod_id: str = Field(...)


class PodsStatusConfig(BaseConfig):
    """Get detailed status of a specific pod"""

    pod_id: str = Field(...)
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class PodsTerminateConfig(BaseConfig):
    """Terminate a pod"""

    pod_id: str = Field(...)
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class RegistryCheckImageConfig(BaseConfig):
    """Verify that an image is accessible (optionally using registry credentials)."""

    image: str = Field(..., description="Image reference, e.g. ghcr.io/org/repo:tag")
    registry_credentials_id: str | None = Field(
        None, description="Registry credentials ID for private images"
    )


class RegistryListConfig(BaseConfig):
    """List registry credentials available to the current user."""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SandboxCreateConfig(BaseConfig):
    """Create a new sandbox"""

    docker_image: str | None = Field(
        None, description="Image to run. When using --vm, provide the VM image reference."
    )
    name: str | None = Field(
        None, description="Name for the sandbox (auto-generated if not provided)"
    )
    start_command: str | None = Field(
        "tail -f /dev/null", description="Command to run in the container"
    )
    cpu_cores: float = Field(1.0, description="Number of CPU cores")
    memory_gb: float = Field(2.0, description="Memory in GB")
    disk_size_gb: float = Field(10.0, description="Disk size in GB")
    gpu_count: int = Field(0, description="Number of GPUs")
    gpu_type: str | None = Field(
        None,
        description="GPU type/model (e.g. H100_80GB, A100_80GB). Required when --gpu-count > 0",
    )
    vm: bool = Field(
        False,
        description="Create a VM-backed sandbox. Required when requesting GPUs.",
    )
    network_access: bool = Field(
        True, description="Allow outbound internet access (enabled by default)"
    )
    timeout_minutes: int = Field(60, description="Timeout in minutes")
    idle_timeout_minutes: int | None = Field(
        None,
        description="Terminate after this many idle minutes. Disabled by default.",
    )
    team_id: str | None = Field(None, description="Team ID (uses config team_id if not specified)")
    region: str | None = Field(
        None,
        description="Sandbox cluster region. Uses the backend default when omitted.",
    )
    registry_credentials_id: str | None = Field(
        None, description="Registry credentials ID for pulling private images"
    )
    env: list[str] | None = Field(
        None,
        description="Environment variables in KEY=VALUE format. Can be specified multiple times.",
    )
    secret: list[str] | None = Field(
        None, description="Secrets in KEY=VALUE format. Can be specified multiple times."
    )
    labels: list[str] | None = Field(
        None, validation_alias=AliasChoices("labels", "l"), description="Labels for the sandbox."
    )
    guaranteed: bool = Field(
        False,
        description="Use Guaranteed QoS. Admin only; incompatible with --vm.",
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class SandboxDeleteConfig(BaseConfig):
    """Delete one or more sandboxes by ID, by label, or all sandboxes with --all"""

    sandbox_ids: list[str] | None = Field(
        None, description="Sandbox ID(s) to delete (space or comma-separated)"
    )
    all: bool = Field(
        False, validation_alias=AliasChoices("all", "a"), description="Delete all sandboxes"
    )
    labels: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("labels", "l"),
        description="Delete sandboxes having all provided labels.",
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )
    all_users: bool = Field(
        False,
        validation_alias=AliasChoices("all_users", "A"),
        description="Delete across every user in the team. Requires team admin role.",
    )
    target_user_id: str | None = Field(
        None,
        validation_alias=AliasChoices("target_user_id", "user", "u"),
        description="Target one teammate. Requires team admin; conflicts with --all-users.",
    )


class SandboxDownloadConfig(BaseConfig):
    """Download a file from a sandbox"""

    sandbox_id: str = Field(..., description="Sandbox ID to download file from")
    remote_path: str = Field(..., description="Path to file in sandbox")
    local_file: str = Field(..., description="Path where file should be saved locally")


class SandboxExposeConfig(BaseConfig):
    """Expose a port from a sandbox."""

    sandbox_id: str = Field(..., description="Sandbox ID to expose port from")
    port: int = Field(..., description="Port number to expose")
    name: str | None = Field(None, description="Optional name for the exposed port")
    protocol: str = Field(
        "HTTP", validation_alias=AliasChoices("protocol", "p"), description="Protocol: HTTP or TCP"
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SandboxGetConfig(BaseConfig):
    """Get detailed information about a specific sandbox"""

    sandbox_id: str = Field(...)
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SandboxListConfig(BaseConfig):
    """List your sandboxes."""

    team_id: str | None = Field(
        None, description="Filter by team ID (uses config team_id if not specified)"
    )
    status: str | None = Field(None, description="Filter by status")
    labels: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("labels", "l"),
        description="Filter by labels; sandboxes must have all provided labels.",
    )
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    num: int = Field(50, validation_alias=AliasChoices("num", "n"), description="Items per page")
    all: bool = Field(False, description="Show all sandboxes including terminated ones")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SandboxListPortsConfig(BaseConfig):
    """List exposed ports for a sandbox, or all sandboxes if no ID is provided"""

    sandbox_id: str | None = Field(
        None, description="Sandbox ID (omit to list all exposed ports across all sandboxes)"
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SandboxLogsConfig(BaseConfig):
    """Get logs from a sandbox"""

    sandbox_id: str = Field(...)


class SandboxResetCacheConfig(BaseConfig):
    """Reset sandbox authentication cache"""

    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class SandboxRunConfig(BaseConfig):
    """Execute a command in a sandbox."""

    sandbox_id: str = Field(...)
    command: list[str] = Field(
        ...,
        description="Command to execute. Use -- before command options.",
    )
    working_dir: str | None = Field(
        None, validation_alias=AliasChoices("working_dir", "w"), description="Working directory"
    )
    env: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("env", "e"),
        description="Environment variables in KEY=VALUE format. Can be specified multiple times.",
    )
    timeout: int | None = Field(None, description="Timeout for the command in seconds")
    user: str | None = Field(
        None,
        validation_alias=AliasChoices("user", "u"),
        description="Container username or UID, optionally USER:GROUP.",
    )


class SandboxSshConfig(BaseConfig):
    """Connect to a sandbox via SSH."""

    sandbox_id: str | None = Field(
        None, description="Sandbox ID to SSH into (interactive selection if not provided)"
    )
    ssh_args: list[str] | None = Field(
        None, description="Additional SSH arguments (e.g., -- -v for verbose)"
    )
    shell: str | None = Field(
        None,
        validation_alias=AliasChoices("shell", "s"),
        description="Shell to use (e.g., bash, zsh, sh). Auto-detected if not specified.",
    )


class SandboxUnexposeConfig(BaseConfig):
    """Unexpose a port from a sandbox"""

    sandbox_id: str = Field(..., description="Sandbox ID")
    exposure_id: str = Field(..., description="Exposure ID to remove")
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class SandboxUploadConfig(BaseConfig):
    """Upload a file to a sandbox"""

    sandbox_id: str = Field(..., description="Sandbox ID to upload file to")
    local_file: str = Field(..., description="Path to local file to upload")
    remote_path: str = Field(..., description="Path where file should be stored in sandbox")


class SecretCreateConfig(BaseConfig):
    """Create a new global secret."""

    name: str | None = Field(
        None,
        validation_alias=AliasChoices("name", "n"),
        description="Secret name (used as environment variable name)",
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="Secret value"
    )
    description: str | None = Field(
        None, validation_alias=AliasChoices("description", "d"), description="Secret description"
    )
    file: bool = Field(
        False,
        validation_alias=AliasChoices("file", "f"),
        description="Treat value as file content (base64 encoded)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SecretDeleteConfig(BaseConfig):
    """Delete a global secret."""

    secret_id: str | None = Field(
        None, description="Secret ID to delete (interactive selection if not provided)"
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class SecretGetConfig(BaseConfig):
    """Get details of a specific secret."""

    secret_id: str = Field(..., description="Secret ID to get")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SecretListConfig(BaseConfig):
    """List your global secrets."""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SecretUpdateConfig(BaseConfig):
    """Update an existing global secret."""

    secret_id: str | None = Field(
        None, description="Secret ID to update (interactive selection if not provided)"
    )
    name: str | None = Field(
        None, validation_alias=AliasChoices("name", "n"), description="New secret name"
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="New secret value"
    )
    description: str | None = Field(
        None,
        validation_alias=AliasChoices("description", "d"),
        description="New secret description",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SwitchConfig(BaseConfig):
    """Switch between your personal account and team contexts"""

    target: str | None = Field(None, description="'personal', a team slug, or a team ID")


class TeamsListConfig(BaseConfig):
    """List teams for the current user."""

    limit: int = Field(100, description="Maximum number of teams to list")
    offset: int = Field(0, description="Number of teams to skip")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TeamsMembersConfig(BaseConfig):
    """List members of a team."""

    team_id: str | None = Field(None, description="Team ID (uses config team_id if omitted)")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TrainCheckpointsConfig(BaseConfig):
    """List checkpoints for a run."""

    run_id: str = Field(..., description="Run ID to list checkpoints for")
    status_filter: str | None = Field(
        None,
        validation_alias=AliasChoices("status_filter", "status", "s"),
        description="Filter by status (READY, PENDING, UPLOADING, FAILED)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TrainComponentsConfig(BaseConfig):
    """List pods (orchestrator + env-servers) for a run."""

    run_id: str = Field(..., description="Run ID to list components for")


class TrainConfigsConfig(BaseConfig):
    """List available configuration options for Hosted Training."""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TrainDeleteConfig(BaseConfig):
    """Delete a run."""

    run_id: str = Field(..., description="Run ID to delete")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Skip confirmation"
    )


class TrainDistributionsConfig(BaseConfig):
    """Get reward/advantage distribution histogram for a run."""

    run_id: str = Field(..., description="Run ID to get distributions for")
    distribution_type: str | None = Field(
        None,
        validation_alias=AliasChoices("distribution_type", "type", "t"),
        description="Distribution type (defaults to all)",
    )
    step: int | None = Field(
        None,
        validation_alias=AliasChoices("step", "s"),
        description="Step number (defaults to latest)",
    )


class TrainGetConfig(BaseConfig):
    """Get details of a specific run."""

    run_id: str = Field(..., description="Run ID to get details for")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TrainInitConfig(BaseConfig):
    """Generate a template config file for a Hosted Training run."""

    output_path: str = Field("rl.toml", description="Output path for the config file")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Overwrite existing file"
    )


class TrainListConfig(BaseConfig):
    """List your runs."""

    team: str | None = Field(
        None, validation_alias=AliasChoices("team", "t"), description="Filter by team ID"
    )
    mine: bool = Field(
        False, description="Filter to only your own runs (useful for admin accounts)"
    )
    num: int = Field(20, validation_alias=AliasChoices("num", "n"), description="Items per page")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TrainLogsConfig(BaseConfig):
    """Get logs for a run."""

    run_id: str = Field(..., description="Run ID to get logs for")
    component: str | None = Field(
        None,
        validation_alias=AliasChoices("component", "c"),
        description="Pod component: orchestrator, trainer, inference, or env-server.",
    )
    env: str | None = Field(
        None,
        description="Env-server name or name/N. Implies --component=env-server.",
    )
    tail: int = Field(
        1000, validation_alias=AliasChoices("tail", "n"), description="Number of lines to show"
    )
    follow: bool = Field(
        False, validation_alias=AliasChoices("follow", "f"), description="Follow log output"
    )
    raw: bool = Field(
        False,
        validation_alias=AliasChoices("raw", "r"),
        description="Show raw logs without formatting",
    )
    search: str | None = Field(
        None,
        description="Filter lines by substring, or by regex with --regex.",
    )
    regex: bool = Field(False, description="Treat --search as a regex (RE2 syntax).")
    level: str | None = Field(
        None, description="Filter to one log level: ERROR | WARNING | SUCCESS | INFO | DEBUG."
    )
    since: str | None = Field(
        None,
        description="Filtered query window, such as 15m, 1h, 24h, or seconds.",
    )


class TrainMetricsConfig(BaseConfig):
    """Get Hosted Training metrics for a run."""

    run_id: str = Field(..., description="Run ID to get metrics for")
    min_step: int | None = Field(None, description="Minimum step (inclusive)")
    max_step: int | None = Field(None, description="Maximum step (inclusive)")
    limit: int | None = Field(
        None, validation_alias=AliasChoices("limit", "n"), description="Maximum number of records"
    )


class TrainModelsConfig(BaseConfig):
    """List available models for Hosted Training."""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TrainProgressConfig(BaseConfig):
    """Get progress information, including which steps have samples and distributions."""

    run_id: str = Field(..., description="Run ID to get progress for")


class TrainRequestConfig(BaseConfig):
    """Request models for Hosted Training."""

    pass


class TrainRestartConfig(BaseConfig):
    """Restart a running run from its latest checkpoint."""

    run_id: str = Field(..., description="Run ID to restart")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Skip confirmation"
    )


class TrainRolloutsConfig(BaseConfig):
    """Get rollout samples for a run."""

    run_id: str = Field(..., description="Run ID to get rollouts for")
    step: int = Field(
        ...,
        validation_alias=AliasChoices("step", "s"),
        description="Step number to get rollouts for",
    )
    page: int = Field(
        1, validation_alias=AliasChoices("page", "p"), description="Page number (1-indexed)"
    )
    num: int = Field(100, validation_alias=AliasChoices("num", "n"), description="Items per page")


class TrainRunConfig(BaseConfig):
    """Launch a Hosted Training run from a config file."""

    config_path: str = Field(
        ..., description="Path to a TOML config file to launch as a Hosted Training run."
    )
    env: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("env", "e", "env_var"),
        description="Environment value: KEY=VALUE, KEY from the shell, or an env file.",
    )
    env_file: list[str] | None = Field(
        None,
        description=".env file containing secrets. Supports ${VAR} expansion.",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
    skip_action_check: bool = Field(
        False, description="Skip action status check and run even if environment action failed."
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )
    image_tag: str | None = Field(
        None,
        description="prime-rl image tag for full-FT runs.",
    )


class TrainStopConfig(BaseConfig):
    """Stop a run."""

    run_id: str = Field(..., description="Run ID to stop")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Skip confirmation"
    )


class TrainUsageConfig(BaseConfig):
    """Show token usage and price for a single training run."""

    run_id: str = Field(..., description="RFT run ID (e.g. rft_...")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
    watch: bool = Field(
        False,
        validation_alias=AliasChoices("watch", "w"),
        description="Poll continuously and update in place",
    )
    interval: int = Field(
        30,
        validation_alias=AliasChoices("interval", "n"),
        description="Seconds between polls when --watch is set",
    )


class TunnelListConfig(BaseConfig):
    """List active tunnels."""

    team_id: str | None = Field(
        None, description="Team ID to list team tunnels (uses config team_id if not specified)"
    )
    labels: list[str] | None = Field(
        None, validation_alias=AliasChoices("labels", "l"), description="Filter by labels."
    )
    status: str | None = Field(None, description="Filter by status")
    sort_by: str = Field(
        "createdAt", description="Sort field: createdAt, status, name, expiresAt, connectedAt"
    )
    sort_order: str = Field("desc", description="Sort order: asc or desc")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    num: int = Field(50, validation_alias=AliasChoices("num", "n"), description="Items per page")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TunnelStartConfig(BaseConfig):
    """Start a tunnel to expose a local port."""

    port: int = Field(
        8765, validation_alias=AliasChoices("port", "p"), description="Local port to tunnel"
    )
    name: str | None = Field(
        None, validation_alias=AliasChoices("name", "n"), description="Friendly name for the tunnel"
    )
    labels: list[str] | None = Field(
        None, validation_alias=AliasChoices("labels", "l"), description="Labels for the tunnel."
    )
    team_id: str | None = Field(
        None, description="Team ID for team tunnels (uses config team_id if not specified)"
    )
    auth: str | None = Field(
        None,
        description="Basic-auth username. A generated password is shown once.",
    )


class TunnelStatusConfig(BaseConfig):
    """Get status of a specific tunnel."""

    tunnel_id: str = Field(..., description="Tunnel ID to check")


class TunnelStopConfig(BaseConfig):
    """Stop and delete one or more tunnels."""

    tunnel_ids: list[str] | None = Field(
        None,
        description="Tunnel IDs to stop. Cannot be combined with filters.",
    )
    all: bool = Field(
        False,
        validation_alias=AliasChoices("all", "a"),
        description="Stop every tunnel in scope. May be narrowed with --status.",
    )
    labels: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("labels", "l"),
        description="Stop tunnels carrying all labels. Conflicts with IDs and --all.",
    )
    status: str | None = Field(
        None,
        description="Filter by pending, connected, or disconnected status.",
    )
    team_id: str | None = Field(
        None,
        description="Team for filtered operations. Required with --all-users.",
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )
    all_users: bool = Field(
        False, description="Target every team member's tunnels. Requires a team."
    )


class UpgradeConfig(BaseConfig):
    """Upgrade the Prime CLI to the latest version"""

    check: bool = Field(
        False,
        validation_alias=AliasChoices("check", "c"),
        description="Only check for updates, don't upgrade",
    )
    force: bool = Field(
        False,
        validation_alias=AliasChoices("force", "f"),
        description="Force upgrade even if already on latest version",
    )


class WalletConfig(BaseConfig):
    """Show wallet balance and most recent billing rows."""

    limit: int = Field(
        20,
        validation_alias=AliasChoices("limit", "n"),
        description="Number of recent billing rows to fetch (max 100)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class WhoamiConfig(BaseConfig):
    """Show current authenticated user and update config"""

    pass


# Env Hub commands are Prime-owned add-ons, so their config lives here directly.
class EnvActionListConfig(BaseConfig):
    """List actions for a published environment."""

    environment: str = Field(..., description="Published environment slug (owner/name).")
    version_id: str | None = Field(None, description="Filter by environment version id.")
    num: int = Field(20, validation_alias=AliasChoices("num", "n"), description="Items per page")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )


class EnvActionLogsConfig(BaseConfig):
    """Show logs for an environment action."""

    environment: str = Field(..., description="Published environment slug (owner/name).")
    action_id: str = Field(..., description="Action id to fetch logs for.")
    tail: int = Field(
        1000, validation_alias=AliasChoices("tail", "n"), description="Number of log lines to show"
    )
    follow: bool = Field(
        False, validation_alias=AliasChoices("follow", "f"), description="Follow log output"
    )


class EnvActionRetryConfig(BaseConfig):
    """Retry an environment action."""

    environment: str = Field(..., description="Published environment slug (owner/name).")
    action_id: str | None = Field(
        None, description="Action id to retry. Defaults to the latest action."
    )
    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )


class EnvBuildConfig(BaseConfig):
    """Build an OpenEnv-backed environment image."""

    env_id: str | None = Field(None, description="Environment id to build.")
    path: str = Field(
        "environments",
        validation_alias=AliasChoices("path", "p"),
        description="Environment root path",
    )


class EnvDeleteConfig(BaseConfig):
    """Delete a published environment."""

    env_id: str = Field(..., description="Published environment slug (owner/name).")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Skip confirmation prompt"
    )


class EnvInfoConfig(BaseConfig):
    """Show published environment details."""

    env_id: str = Field(..., description="Published environment slug (owner/name).")
    version: str = Field(
        "latest", validation_alias=AliasChoices("version", "v"), description="Environment version"
    )


class EnvInspectConfig(BaseConfig):
    """Inspect published environment source."""

    env_id: str = Field(..., description="Published environment slug (owner/name).")
    source_path: str | None = Field(None, description="Source path inside the environment archive")
    version: str = Field(
        "latest", validation_alias=AliasChoices("version", "v"), description="Environment version"
    )
    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )
    max_bytes: int = Field(65536, description="Maximum file bytes to return")


class EnvInstallConfig(BaseConfig):
    """Install one or more Verifiers environments."""

    env_ids: list[str] = Field(..., description="Environment names or Hub references to install")
    path: str = Field(
        "environments",
        validation_alias=AliasChoices("path", "p"),
        description="Local environments root path",
    )
    prerelease: bool = Field(False, description="Allow prerelease package versions")


class EnvListConfig(BaseConfig):
    """List published environments."""

    num: int = Field(20, validation_alias=AliasChoices("num", "n"), description="Items per page")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    owner: str | None = Field(None, description="Filter by owner slug")
    visibility: str | None = Field(None, description="Filter by visibility")
    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )
    search: str | None = Field(None, description="Search environments")
    tag: str | None = Field(None, description="Filter by tag")
    action_status: str | None = Field(None, description="Filter by latest action status")
    sort: str = Field("updated_at", description="Sort field")
    order: str = Field("desc", description="Sort order")
    show_actions: bool = Field(False, description="Show latest action status")
    starred: bool = Field(False, description="Show only starred environments")
    mine: bool = Field(False, description="Show only your environments")


class EnvPullConfig(BaseConfig):
    """Pull published environment source locally."""

    env_id: str = Field(..., description="Published environment slug (owner/name).")
    target: str | None = Field(
        None, validation_alias=AliasChoices("target", "t"), description="Target directory"
    )
    version: str = Field(
        "latest", validation_alias=AliasChoices("version", "v"), description="Environment version"
    )


class EnvPushConfig(BaseConfig):
    """Publish an environment to Prime Hub."""

    env_id: str | None = Field(None, description="Environment id to publish from ./environments")
    path: str | None = Field(
        None,
        validation_alias=AliasChoices("path", "p"),
        description="Environment path or root path",
    )
    name: str | None = Field(
        None, validation_alias=AliasChoices("name", "n"), description="Published environment name"
    )
    owner: str | None = Field(None, description="Owner slug")
    team: str | None = Field(
        None, validation_alias=AliasChoices("team", "t"), description="Team slug"
    )
    visibility: str | None = Field(None, description="Environment visibility")
    auto_bump: bool = Field(False, description="Bump patch version before publishing")
    rc: bool = Field(False, description="Bump or create an rc version suffix")
    post: bool = Field(False, description="Bump or create a post version suffix")


class EnvSecretCreateConfig(BaseConfig):
    """Create an environment-specific secret."""

    environment: str | None = Field(None, description="Published environment slug (owner/name).")
    name: str | None = Field(
        None, validation_alias=AliasChoices("name", "n"), description="Secret name"
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="Secret value"
    )
    description: str | None = Field(
        None, validation_alias=AliasChoices("description", "d"), description="Secret description"
    )
    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )


class EnvSecretDeleteConfig(BaseConfig):
    """Delete an environment-specific secret."""

    environment: str | None = Field(None, description="Published environment slug (owner/name).")
    secret_id: str | None = Field(
        None, validation_alias=AliasChoices("secret_id", "id"), description="Environment secret id"
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class EnvSecretLinkConfig(BaseConfig):
    """Link a global secret to an environment."""

    global_secret_id: str = Field(..., description="Global secret id to link")
    environment: str | None = Field(None, description="Published environment slug (owner/name).")
    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )


class EnvSecretListConfig(BaseConfig):
    """List environment secrets."""

    environment: str | None = Field(None, description="Published environment slug (owner/name).")
    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )


class EnvSecretUnlinkConfig(BaseConfig):
    """Unlink a global secret from an environment."""

    global_secret_id: str = Field(..., description="Global secret id to unlink")
    environment: str | None = Field(None, description="Published environment slug (owner/name).")
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class EnvSecretUpdateConfig(BaseConfig):
    """Update an environment-specific secret."""

    environment: str | None = Field(None, description="Published environment slug (owner/name).")
    secret_id: str | None = Field(
        None, validation_alias=AliasChoices("secret_id", "id"), description="Environment secret id"
    )
    name: str | None = Field(
        None, validation_alias=AliasChoices("name", "n"), description="New secret name"
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="New secret value"
    )
    description: str | None = Field(
        None,
        validation_alias=AliasChoices("description", "d"),
        description="New secret description",
    )
    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )


class EnvStatusConfig(BaseConfig):
    """Show environment action status."""

    env_id: str = Field(..., description="Published environment slug (owner/name).")
    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )


class EnvUninstallConfig(BaseConfig):
    """Uninstall an environment distribution."""

    env_name: str = Field(..., description="Environment name or Hub reference")


class EnvVarCreateConfig(BaseConfig):
    """Create an environment variable."""

    environment: str | None = Field(None, description="Published environment slug (owner/name).")
    name: str | None = Field(
        None, validation_alias=AliasChoices("name", "n"), description="Variable name"
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="Variable value"
    )
    description: str | None = Field(
        None, validation_alias=AliasChoices("description", "d"), description="Variable description"
    )
    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )


class EnvVarDeleteConfig(BaseConfig):
    """Delete an environment variable."""

    var_id: str = Field(
        ..., validation_alias=AliasChoices("var_id", "id"), description="Environment variable id"
    )
    environment: str | None = Field(None, description="Published environment slug (owner/name).")
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class EnvVarListConfig(BaseConfig):
    """List environment variables."""

    environment: str | None = Field(None, description="Published environment slug (owner/name).")
    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )


class EnvVarUpdateConfig(BaseConfig):
    """Update an environment variable."""

    var_id: str = Field(
        ..., validation_alias=AliasChoices("var_id", "id"), description="Environment variable id"
    )
    environment: str | None = Field(None, description="Published environment slug (owner/name).")
    name: str | None = Field(
        None, validation_alias=AliasChoices("name", "n"), description="New variable name"
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="New variable value"
    )
    description: str | None = Field(
        None,
        validation_alias=AliasChoices("description", "d"),
        description="New variable description",
    )
    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )


class EnvVersionDeleteConfig(BaseConfig):
    """Delete a published environment version."""

    env_id: str = Field(..., description="Published environment slug (owner/name).")
    content_hash: str = Field(..., description="Version content hash")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Skip confirmation prompt"
    )


class EnvVersionListConfig(BaseConfig):
    """List published environment versions."""

    env_id: str = Field(..., description="Published environment slug (owner/name).")
    full_hashes: bool = Field(False, description="Show full content hashes")
