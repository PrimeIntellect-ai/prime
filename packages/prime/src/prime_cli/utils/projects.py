"""Local active-project context for Lab workflows."""

import json
import os
from pathlib import Path
from typing import Optional

from prime_cli.api.projects import Project, ProjectsClient
from prime_cli.core import APIClient, APIError, Config

PROJECT_CONTEXT_ENV = "PRIME_PROJECT_ID"
PROJECT_CONTEXT_CLEARED_KEY = "project_cleared"
PROJECT_CONTEXT_ENV_OVERRIDE_KEY = "project_overrides_env"


def project_context_path(workspace: Optional[Path] = None) -> Path:
    root = (workspace or Path.cwd()).resolve()
    return root / ".prime" / "lab" / "context.json"


def _find_lab_workspace_root(workspace: Optional[Path] = None) -> Optional[Path]:
    root = (workspace or Path.cwd()).resolve()
    for candidate in (root, *root.parents):
        if (candidate / ".prime" / "lab.json").exists():
            return candidate
    return None


def _find_project_context_path(workspace: Optional[Path] = None) -> Optional[Path]:
    root = (workspace or Path.cwd()).resolve()
    lab_root = _find_lab_workspace_root(root)
    if lab_root:
        path = project_context_path(lab_root)
        return path if path.exists() else None

    path = project_context_path(root)
    return path if path.exists() else None


def _write_project_context_path(workspace: Optional[Path] = None) -> Path:
    existing_path = _find_project_context_path(workspace)
    if existing_path:
        return existing_path

    lab_root = _find_lab_workspace_root(workspace)
    if lab_root:
        return project_context_path(lab_root)

    root = (workspace or Path.cwd()).resolve()
    return project_context_path(root)


def read_project_context(workspace: Optional[Path] = None) -> dict:
    path = _find_project_context_path(workspace)
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _env_project_id() -> Optional[str]:
    env_project_id = os.getenv(PROJECT_CONTEXT_ENV)
    if env_project_id and env_project_id.strip():
        return env_project_id.strip()
    return None


def _context_matches_scope(context: dict, config: Config) -> bool:
    if context.get("base_url") and context.get("base_url") != config.base_url:
        return False
    if context.get("team_id") != config.team_id:
        return False
    return True


def scope_label(team_id: Optional[str]) -> str:
    return f"team {team_id}" if team_id else "personal account"


def ensure_active_project_scope(
    project_team_id: Optional[str],
    config: Config,
    *,
    action: str,
    guidance: Optional[str] = None,
) -> None:
    if project_team_id == config.team_id:
        return

    message = (
        f"Cannot {action} for {scope_label(project_team_id)} while the CLI is "
        f"using {scope_label(config.team_id)}. Switch account context first with "
        "'prime switch <team-slug-or-id>'."
    )
    if guidance:
        message = f"{message} {guidance}"

    raise APIError(message)


def write_project_context(
    project: Project,
    config: Optional[Config] = None,
    workspace: Optional[Path] = None,
) -> None:
    config = config or Config()
    path = _write_project_context_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    context = {
        "project_id": project.id,
        "project_slug": project.slug,
        "project_name": project.name,
        "team_id": project.team_id,
        "base_url": config.base_url,
    }
    if _env_project_id():
        context[PROJECT_CONTEXT_ENV_OVERRIDE_KEY] = True

    path.write_text(
        json.dumps(context, indent=2) + "\n",
        encoding="utf-8",
    )


def clear_project_context(
    workspace: Optional[Path] = None,
    config: Optional[Config] = None,
) -> bool:
    path = _find_project_context_path(workspace)
    env_project_id = _env_project_id()
    if env_project_id:
        config = config or Config()
        path = path or _write_project_context_path(workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "project_id": None,
                    PROJECT_CONTEXT_CLEARED_KEY: True,
                    "team_id": config.team_id,
                    "base_url": config.base_url,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return True

    if path is None:
        return False
    path.unlink()
    return True


def get_active_project_id(
    config: Optional[Config] = None,
    workspace: Optional[Path] = None,
    client: Optional[APIClient] = None,
) -> Optional[str]:
    config = config or Config()
    context = read_project_context(workspace)
    if context.get(PROJECT_CONTEXT_CLEARED_KEY) is True and _context_matches_scope(
        context,
        config,
    ):
        return None
    if context.get(PROJECT_CONTEXT_ENV_OVERRIDE_KEY) is True and _context_matches_scope(
        context,
        config,
    ):
        project_id = context.get("project_id")
        return str(project_id) if project_id else None

    env_project_id = _env_project_id()
    if env_project_id:
        if client is not None:
            project = ProjectsClient(client).get(env_project_id, team_id=config.team_id)
            ensure_active_project_scope(
                project.team_id,
                config,
                action=f"use {PROJECT_CONTEXT_ENV}",
            )
            return project.id
        return env_project_id

    if not context:
        return None

    if not _context_matches_scope(context, config):
        return None

    project_id = context.get("project_id")
    return str(project_id) if project_id else None


def resolve_project_id(
    project_ref: Optional[str],
    *,
    no_project: bool = False,
    use_active_project: bool = False,
    config: Optional[Config] = None,
    client: Optional[APIClient] = None,
) -> Optional[str]:
    if project_ref and no_project:
        raise APIError("Cannot use --project and --no-project together.")

    if no_project:
        return None

    config = config or Config()
    if project_ref:
        projects_client = ProjectsClient(client or APIClient())
        return projects_client.get(project_ref, team_id=config.team_id).id

    if not use_active_project:
        return None

    return get_active_project_id(config, client=client or APIClient())
