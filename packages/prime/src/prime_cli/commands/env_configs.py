"""Pydantic Config schemas for the ``prime env`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig

_ENV_SLUG_HELP = "Environment slug (e.g., 'owner/environment-name'). Auto-detected if not provided."


class EnvActionListConfig(BaseConfig):
    """actions_list config."""

    environment: str = Field(..., description="Environment slug (e.g., 'owner/environment-name')")
    version_id: str | None = Field(
        None, validation_alias=AliasChoices("version_id", "v"), description="Filter by version ID"
    )
    num: int = Field(20, validation_alias=AliasChoices("num", "n"), description="Items per page")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class EnvActionLogsConfig(BaseConfig):
    """actions_logs config."""

    environment: str = Field(..., description="Environment slug (e.g., 'owner/environment-name')")
    action_id: str = Field(..., description="Action/job ID to get logs for")
    tail: int = Field(
        1000, validation_alias=AliasChoices("tail", "n"), description="Number of lines to show"
    )
    follow: bool = Field(
        False, validation_alias=AliasChoices("follow", "f"), description="Follow log output"
    )


class EnvActionRetryConfig(BaseConfig):
    """actions_retry config."""

    environment: str = Field(..., description="Environment slug (e.g., 'owner/environment-name')")
    action_id: str | None = Field(
        None, description="Action ID to retry (retries latest action if not provided)"
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class EnvBuildConfig(BaseConfig):
    """build config."""

    env_id: str | None = Field(None, description="Environment ID (hyphenated, e.g. openenv-echo)")
    path: str = Field(
        "./environments",
        validation_alias=AliasChoices("path", "p"),
        description="Base environments path, or the environment path when ENV_ID is omitted",
    )


class EnvDeleteConfig(BaseConfig):
    """delete config."""

    env_id: str = Field(..., description="Environment ID to delete")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Skip confirmation"
    )


class EnvInfoConfig(BaseConfig):
    """info config."""

    env_id: str = Field(..., description="Environment ID (owner/name)")
    version: str = Field(
        "latest", validation_alias=AliasChoices("version", "v"), description="Version to show"
    )


class EnvInspectConfig(BaseConfig):
    """inspect_cmd config."""

    env_id: str = Field(..., description="Environment ID (owner/name or owner/name@version)")
    source_path: str | None = Field(
        None, description="Optional file or directory path inside the environment source"
    )
    version: str = Field(
        "latest", validation_alias=AliasChoices("version", "v"), description="Version to inspect"
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
    max_bytes: int = Field(
        100000, description="Maximum file bytes to return when inspecting a file"
    )


class EnvInstallConfig(BaseConfig):
    """install config."""

    env_ids: list[str] = Field(
        ..., description="Environment ID(s) to install (owner/name or local name)"
    )
    path: str = Field(
        "./environments",
        validation_alias=AliasChoices("path", "p"),
        description="Path to local environments directory (for local installs)",
    )
    prerelease: bool = Field(
        False, description="Allow pre-release versions (e.g., verifiers>=0.1.12.dev3)."
    )


class EnvListConfig(BaseConfig):
    """list_cmd config."""

    num: int = Field(20, validation_alias=AliasChoices("num", "n"), description="Items per page")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    owner: str | None = Field(None, description="Filter by owner name")
    visibility: str | None = Field(None, description="Filter by visibility (PUBLIC/PRIVATE)")
    output: str = Field("table", description="Output format: table or json")
    search: str | None = Field(
        None,
        validation_alias=AliasChoices("search", "s"),
        description="Search by name or description",
    )
    tag: list[str] | None = Field(
        None, validation_alias=AliasChoices("tag", "t"), description="Filter by tag (repeatable)"
    )
    action_status: str | None = Field(
        None, description="Filter by action status (SUCCESS/FAILED/RUNNING/PENDING)"
    )
    sort: str = Field("created_at", description="Sort by: name, created_at, updated_at, stars")
    order: str = Field("desc", description="Sort order: asc, desc")
    show_actions: bool = Field(False, description="Show action status column")
    starred: bool = Field(False, description="Filter to only environments you have starred")
    mine: bool = Field(False, description="Filter to only your own environments (personal + team)")


class EnvPullConfig(BaseConfig):
    """pull config."""

    env_id: str = Field(..., description="Environment ID (owner/name or owner/name@version)")
    target: str | None = Field(
        None, validation_alias=AliasChoices("target", "t"), description="Target directory"
    )
    version: str = Field(
        "latest", validation_alias=AliasChoices("version", "v"), description="Version to pull"
    )


class EnvPushConfig(BaseConfig):
    """push config."""

    env_id: str | None = Field(
        None,
        description=(
            "Optional environment ID used as the local folder name (hyphens map to underscores)"
        ),
    )
    path: str | None = Field(
        None,
        validation_alias=AliasChoices("path", "p"),
        description=(
            "Path to environment directory. Defaults to '.' without env_id, "
            "or './environments' as the parent directory with env_id."
        ),
    )
    name: str | None = Field(
        None,
        validation_alias=AliasChoices("name", "n"),
        description="Override environment name (defaults to pyproject.toml name)",
    )
    owner: str | None = Field(
        None,
        validation_alias=AliasChoices("owner", "o"),
        description="Owner slug (user or team) to push to (for collaborators with write access)",
    )
    team: str | None = Field(
        None,
        validation_alias=AliasChoices("team", "t"),
        description="Team slug for team ownership (uses config team_id if not provided)",
    )
    visibility: str | None = Field(
        None,
        validation_alias=AliasChoices("visibility", "v"),
        description="Environment visibility (PUBLIC/PRIVATE)",
    )
    auto_bump: bool = Field(False, description="Automatically bump patch version before push")
    rc: bool = Field(False, description="Bump or create a .rc pre-release (rc0 -> rc1)")
    post: bool = Field(False, description="Bump or create a .post release (post0 -> post1)")


class EnvSecretCreateConfig(BaseConfig):
    """env_secret_create config."""

    environment: str | None = Field(
        None,
        description=_ENV_SLUG_HELP,
    )
    name: str | None = Field(
        None,
        validation_alias=AliasChoices("name", "n"),
        description="Secret name (must be uppercase with underscores, e.g., MY_SECRET)",
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="Secret value"
    )
    description: str | None = Field(
        None, validation_alias=AliasChoices("description", "d"), description="Secret description"
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class EnvSecretDeleteConfig(BaseConfig):
    """env_secret_delete config."""

    environment: str | None = Field(
        None,
        description=_ENV_SLUG_HELP,
    )
    secret_id: str | None = Field(
        None,
        validation_alias=AliasChoices("secret_id", "id"),
        description="Secret ID to delete (interactive selection if not provided)",
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class EnvSecretLinkConfig(BaseConfig):
    """env_secret_link config."""

    global_secret_id: str = Field(..., description="Global secret ID to link")
    environment: str | None = Field(
        None,
        description=_ENV_SLUG_HELP,
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class EnvSecretListConfig(BaseConfig):
    """env_secret_list config."""

    environment: str | None = Field(
        None,
        description=_ENV_SLUG_HELP,
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class EnvSecretUnlinkConfig(BaseConfig):
    """env_secret_unlink config."""

    global_secret_id: str = Field(..., description="Global secret ID to unlink")
    environment: str | None = Field(
        None,
        description=_ENV_SLUG_HELP,
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class EnvSecretUpdateConfig(BaseConfig):
    """env_secret_update config."""

    environment: str | None = Field(
        None,
        description=_ENV_SLUG_HELP,
    )
    secret_id: str | None = Field(
        None,
        validation_alias=AliasChoices("secret_id", "id"),
        description="Secret ID to update (interactive selection if not provided)",
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


class EnvStatusConfig(BaseConfig):
    """status_cmd config."""

    env_id: str = Field(..., description="Environment ID (owner/name)")
    output: str = Field("table", description="Output format: table or json")


class EnvUninstallConfig(BaseConfig):
    """uninstall config."""

    env_name: str = Field(..., description="Environment name to uninstall")


class EnvVarCreateConfig(BaseConfig):
    """var_create config."""

    environment: str | None = Field(
        None,
        description=_ENV_SLUG_HELP,
    )
    name: str | None = Field(
        None,
        validation_alias=AliasChoices("name", "n"),
        description="Variable name (must be uppercase with underscores, e.g., MY_VAR)",
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="Variable value"
    )
    description: str | None = Field(
        None, validation_alias=AliasChoices("description", "d"), description="Variable description"
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class EnvVarDeleteConfig(BaseConfig):
    """var_delete config."""

    var_id: str = Field(..., description="Variable ID to delete")
    environment: str | None = Field(
        None,
        description=_ENV_SLUG_HELP,
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class EnvVarListConfig(BaseConfig):
    """var_list config."""

    environment: str | None = Field(
        None,
        description=_ENV_SLUG_HELP,
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class EnvVarUpdateConfig(BaseConfig):
    """var_update config."""

    var_id: str = Field(..., description="Variable ID to update")
    environment: str | None = Field(
        None,
        description=_ENV_SLUG_HELP,
    )
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
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class EnvVersionDeleteConfig(BaseConfig):
    """delete_version config."""

    env_id: str = Field(..., description="Environment ID (owner/name)")
    content_hash: str = Field(..., description="Content hash of the version to delete")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Skip confirmation"
    )


class EnvVersionListConfig(BaseConfig):
    """list_versions config."""

    env_id: str = Field(..., description="Environment ID (owner/name)")
    full_hashes: bool = Field(
        False, description="Show full content hashes instead of shortened ones"
    )
