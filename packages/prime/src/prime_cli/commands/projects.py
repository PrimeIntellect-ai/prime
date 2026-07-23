"""Lab Project commands."""

import os
from typing import Optional

import typer
from prime_evals import EvalsAPIError, EvalsClient
from rich.table import Table

from prime_cli.api.deployments import DeploymentsClient
from prime_cli.api.projects import Project, ProjectsClient
from prime_cli.api.rl import RLClient
from prime_cli.core import Config

from ..client import APIClient, APIError
from ..utils import (
    PlainTyper,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from ..utils.projects import (
    PROJECT_CONTEXT_ENV,
    clear_project_context,
    ensure_active_project_scope,
    get_active_project_id,
    read_project_context,
    write_project_context,
)

console = get_console()


def _usage_help(*examples: tuple[str, str], json_help_text: Optional[str] = None) -> str:
    lines = ["\b", "Examples:"]
    for command, annotation in examples:
        lines.append(f"  {command}")
        lines.append(f"    {annotation}")
    if json_help_text:
        lines.append("")
        lines.append("\b")
        lines.extend(json_help_text.splitlines())
    return "\n".join(lines)


PROJECT_USAGE_HELP = _usage_help(
    (
        'prime project create "Alphabet Sort Baselines"',
        "Create a project and make it active for the current workspace.",
    ),
    (
        "prime project current",
        "Show the active project that new Lab runs and evals will use.",
    ),
    (
        "prime project use <project-id>",
        "Switch this workspace to an existing project.",
    ),
    (
        "prime train rl.toml --no-project",
        "Launch a run without attaching it to the active workspace project.",
    ),
)

app = PlainTyper(
    help="Create and switch between Lab projects",
    no_args_is_help=True,
    epilog=PROJECT_USAGE_HELP,
)

PROJECT_JSON_HELP = json_output_help(
    ".project = {id, name, slug, description?, status, userId, teamId?, createdAt, updatedAt}",
)

PROJECT_LIST_JSON_HELP = json_output_help(
    ".projects[] = {id, name, slug, description?, status, userId, teamId?, createdAt, updatedAt}",
    ".total_count = number",
)

PROJECT_CREATE_HELP = _usage_help(
    (
        'prime project create "Alphabet Sort Baselines"',
        "Create a project and set it as active for this workspace.",
    ),
    (
        'prime project create "Alphabet Sort Baselines" --description "Baseline runs"',
        "Create with a description shown in project details.",
    ),
    (
        'prime project create "Team Project" --team-id <team-id> --no-use',
        "Create under a team without changing the active workspace project.",
    ),
    json_help_text=PROJECT_JSON_HELP,
)

PROJECT_LIST_HELP = _usage_help(
    (
        "prime project list",
        "List active projects for the current personal or team context.",
    ),
    (
        "prime project list --limit 50 --offset 50",
        "Page through active projects.",
    ),
    (
        "prime project list --output json",
        "Print machine-readable project rows.",
    ),
    json_help_text=PROJECT_LIST_JSON_HELP,
)

PROJECT_SHOW_HELP = _usage_help(
    (
        "prime project show",
        "Show the active workspace project.",
    ),
    (
        "prime project show <project-id>",
        "Show a specific project by id.",
    ),
    (
        "prime project show <project-id> --output json",
        "Print project details as JSON.",
    ),
    json_help_text=PROJECT_JSON_HELP,
)

PROJECT_USE_HELP = _usage_help(
    (
        "prime project use <project-id>",
        "Set the active project for this workspace.",
    ),
    (
        "prime switch <team-slug-or-id>",
        "Switch to a team before setting one of its projects as active.",
    ),
    (
        "prime project current",
        "Confirm which project this workspace will use by default.",
    ),
    json_help_text=PROJECT_JSON_HELP,
)

PROJECT_CURRENT_HELP = _usage_help(
    (
        "prime project current",
        "Show the active workspace project.",
    ),
    (
        "prime project current --output json",
        "Print the active project, or null when none is set.",
    ),
    json_help_text=PROJECT_JSON_HELP,
)

PROJECT_UPDATE_HELP = _usage_help(
    (
        'prime project update --description "Baseline alphabet sort runs"',
        "Update the active project's description.",
    ),
    (
        'prime project update <project-id> --name "New Project Name"',
        "Rename a specific project.",
    ),
    (
        "prime project update <project-id> --clear-description",
        "Clear the description field.",
    ),
    json_help_text=PROJECT_JSON_HELP,
)

PROJECT_CLEAR_HELP = _usage_help(
    (
        "prime project clear",
        "Stop attaching new Lab runs and evals to a workspace project by default.",
    ),
)

PROJECT_ASSIGN_HELP = _usage_help(
    (
        "prime project assign run <run-id>",
        "Add a training run to the active project and move its adapters too.",
    ),
    (
        "prime project assign run <run-id> <project-id> --no-move-adapters",
        "Add a run to a specific project without changing adapter project membership.",
    ),
    (
        "prime project assign eval <eval-id> <project-id>",
        "Set an evaluation's project.",
    ),
    (
        "prime project assign adapter <adapter-id> <project-id>",
        "Add an adapter to a project.",
    ),
)

PROJECT_REMOVE_HELP = _usage_help(
    (
        "prime project remove run <run-id>",
        "Clear all project memberships from a training run and its adapters.",
    ),
    (
        "prime project remove run <run-id> <project-id> --no-move-adapters",
        "Remove one project from a run without changing adapter project membership.",
    ),
    (
        "prime project remove eval <eval-id>",
        "Clear an evaluation's project.",
    ),
    (
        "prime project remove adapter <adapter-id> <project-id>",
        "Remove one project from an adapter.",
    ),
)


def _project_payload(project: Project) -> dict:
    return project.model_dump(mode="json", by_alias=True)


def _print_project(project: Project, *, active: bool = False) -> None:
    table = Table(title="Active Project" if active else "Project")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Name", project.name)
    table.add_row("Slug", project.slug)
    table.add_row("ID", project.id)
    table.add_row("Status", project.status)
    table.add_row("Team", project.team_id or "Personal")
    table.add_row("Description", project.description or "Not set")
    table.add_row("Created", project.created_at.isoformat())
    table.add_row("Updated", project.updated_at.isoformat())
    if project.archived_at:
        table.add_row("Archived", project.archived_at.isoformat())
    console.print(table)


def _active_project_ref_or_exit(
    config: Config,
    api_client: Optional[APIClient] = None,
) -> str:
    try:
        active_project_id = get_active_project_id(
            config,
            client=api_client or APIClient(),
        )
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    if not active_project_id:
        console.print("[yellow]No active project for this workspace.[/yellow]")
        raise typer.Exit(1)
    return active_project_id


def _active_project_id_for_display(config: Config, api_client: APIClient) -> Optional[str]:
    try:
        return get_active_project_id(config, client=api_client)
    except APIError:
        return None


def _normalize_artifact_kind(kind: str) -> str:
    normalized = kind.strip().lower().replace("_", "-")
    if normalized in {"run", "training", "training-run", "rft-run"}:
        return "training_run"
    if normalized in {"eval", "evaluation"}:
        return "evaluation"
    if normalized in {"adapter", "deployment", "inference", "lora"}:
        return "adapter"

    console.print("[red]Error:[/red] Kind must be one of run, eval, or adapter.")
    raise typer.Exit(1)


def _assignment_payload(
    *,
    kind: str,
    artifact_id: str,
    project_id: Optional[str],
    project_slug: Optional[str] = None,
    adapters_updated: Optional[int] = None,
) -> dict:
    payload = {
        "artifact_type": kind,
        "artifact_id": artifact_id,
        "project_id": project_id,
        "project_slug": project_slug,
    }
    if adapters_updated is not None:
        payload["adapters_updated"] = adapters_updated
    return payload


def _print_assignment_result(payload: dict, *, removed: bool = False) -> None:
    table = Table(title="Project Assignment")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Artifact Type", str(payload["artifact_type"]))
    table.add_row("Artifact ID", str(payload["artifact_id"]))
    table.add_row("Project", str(payload["project_slug"] or payload["project_id"] or "None"))
    if payload.get("adapters_updated") is not None:
        table.add_row("Adapters Updated", str(payload["adapters_updated"]))

    console.print(
        "[green]✓ Project removed[/green]" if removed else "[green]✓ Project assigned[/green]"
    )
    console.print(table)


@app.command("create", epilog=PROJECT_CREATE_HELP)
def create_project(
    name: str = typer.Argument(..., help="Project display name"),
    slug: Optional[str] = typer.Option(None, "--slug", help="Stable project slug"),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Project description",
    ),
    team_id: Optional[str] = typer.Option(
        None,
        "--team-id",
        help="Team ID. Defaults to the active CLI team, if any.",
    ),
    use_project: bool = typer.Option(
        True,
        "--use/--no-use",
        help="Set the created project as the active project for this workspace.",
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Create a Lab project."""
    validate_output_format(output, console)
    config = Config()
    resolved_team_id = team_id if team_id is not None else config.team_id

    try:
        if use_project:
            ensure_active_project_scope(
                resolved_team_id,
                config,
                action="create and set an active project",
                guidance="Use --no-use for one-off team project creation.",
            )

        project = ProjectsClient(APIClient()).create(
            name=name,
            slug=slug,
            description=description,
            team_id=resolved_team_id,
        )
        if use_project:
            write_project_context(project, config)

        if output == "json":
            output_data_as_json({"project": _project_payload(project)}, console)
            return

        console.print("[green]✓ Project created[/green]")
        _print_project(project, active=use_project)
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("list", epilog=PROJECT_LIST_HELP)
def list_projects(
    team_id: Optional[str] = typer.Option(
        None,
        "--team-id",
        help="Team ID. Defaults to the active CLI team, if any.",
    ),
    limit: int = typer.Option(100, "--limit", help="Maximum number of projects to list"),
    offset: int = typer.Option(0, "--offset", help="Number of projects to skip"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List Lab projects for the current workspace."""
    validate_output_format(output, console)
    config = Config()
    resolved_team_id = team_id if team_id is not None else config.team_id
    api_client = APIClient()

    try:
        projects, total = ProjectsClient(api_client).list(
            team_id=resolved_team_id,
            limit=limit,
            offset=offset,
        )
        if output == "json":
            output_data_as_json(
                {
                    "projects": [_project_payload(project) for project in projects],
                    "total_count": total,
                    "offset": offset,
                    "limit": limit,
                },
                console,
            )
            return

        active_project_id = _active_project_id_for_display(config, api_client)
        table = Table(title=f"Projects (Total: {total})")
        table.add_column("", style="green", no_wrap=True)
        table.add_column("Name", style="blue")
        table.add_column("Slug", style="green")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="yellow")
        table.add_column("Updated", style="magenta")
        for project in projects:
            table.add_row(
                "*" if project.id == active_project_id else "",
                project.name,
                project.slug,
                project.id,
                project.status,
                project.updated_at.isoformat(),
            )
        console.print(table)
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("show", epilog=PROJECT_SHOW_HELP)
def show_project(
    project_ref: Optional[str] = typer.Argument(
        None,
        help="Project ID or slug. Defaults to the active project for this workspace.",
    ),
    team_id: Optional[str] = typer.Option(
        None,
        "--team-id",
        help="Team ID. Defaults to the active CLI team, if any.",
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show a Lab project and its configured fields."""
    validate_output_format(output, console)
    config = Config()
    resolved_team_id = team_id if team_id is not None else config.team_id
    api_client = APIClient()
    resolved_project_ref = project_ref or _active_project_ref_or_exit(config, api_client)

    try:
        project = ProjectsClient(api_client).get(
            resolved_project_ref,
            team_id=resolved_team_id,
        )

        if output == "json":
            output_data_as_json({"project": _project_payload(project)}, console)
            return

        _print_project(
            project,
            active=project.id == _active_project_id_for_display(config, api_client),
        )
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("use", epilog=PROJECT_USE_HELP)
def use_project(
    project_ref: str = typer.Argument(..., help="Project ID or slug"),
    team_id: Optional[str] = typer.Option(
        None,
        "--team-id",
        help="Team ID. Defaults to the active CLI team, if any.",
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Set the active Lab project for this workspace."""
    validate_output_format(output, console)
    config = Config()
    resolved_team_id = team_id if team_id is not None else config.team_id

    try:
        project = ProjectsClient(APIClient()).get(project_ref, team_id=resolved_team_id)
        ensure_active_project_scope(
            project.team_id,
            config,
            action="set an active project",
        )
        write_project_context(project, config)

        if output == "json":
            output_data_as_json({"project": _project_payload(project)}, console)
            return

        console.print("[green]✓ Active project updated[/green]")
        _print_project(project, active=True)
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("current", epilog=PROJECT_CURRENT_HELP)
def current_project(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show the active Lab project for this workspace."""
    validate_output_format(output, console)
    config = Config()
    api_client = APIClient()
    try:
        active_project_id = get_active_project_id(config, client=api_client)
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
    context = read_project_context()

    if not active_project_id:
        if output == "json":
            output_data_as_json({"project": None}, console)
            return
        console.print("[yellow]No active project for this workspace.[/yellow]")
        return

    try:
        project = ProjectsClient(api_client).get(active_project_id, team_id=config.team_id)
        if output == "json":
            output_data_as_json({"project": _project_payload(project)}, console)
            return
        _print_project(project, active=True)
    except APIError:
        if output == "json":
            output_data_as_json(
                {
                    "project": None,
                    "context": context or {"project_id": active_project_id},
                },
                console,
            )
            return
        console.print(f"[yellow]Active project:[/yellow] {active_project_id}")
        if context.get("project_slug"):
            console.print(f"[dim]Slug:[/dim] {context['project_slug']}")
        console.print("[dim]Could not fetch current project details from the API.[/dim]")


@app.command("update", epilog=PROJECT_UPDATE_HELP)
def update_project(
    project_ref: Optional[str] = typer.Argument(
        None,
        help="Project ID or slug. Defaults to the active project for this workspace.",
    ),
    name: Optional[str] = typer.Option(None, "--name", help="Project display name"),
    slug: Optional[str] = typer.Option(None, "--slug", help="Stable project slug"),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Project description",
    ),
    clear_description: bool = typer.Option(
        False,
        "--clear-description",
        help="Clear the project description.",
    ),
    team_id: Optional[str] = typer.Option(
        None,
        "--team-id",
        help="Team ID. Defaults to the active CLI team, if any.",
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Update a Lab project's user-editable fields."""
    validate_output_format(output, console)
    config = Config()
    resolved_team_id = team_id if team_id is not None else config.team_id
    api_client = APIClient()
    resolved_project_ref = project_ref or _active_project_ref_or_exit(config, api_client)

    if description is not None and clear_description:
        console.print(
            "[red]Error:[/red] Use either --description or --clear-description, not both."
        )
        raise typer.Exit(1)

    resolved_description = "" if clear_description else description

    if name is None and slug is None and resolved_description is None:
        console.print(
            "[red]Error:[/red] Provide --name, --slug, --description, or --clear-description."
        )
        raise typer.Exit(1)

    try:
        project = ProjectsClient(api_client).update(
            resolved_project_ref,
            name=name,
            slug=slug,
            description=resolved_description,
            team_id=resolved_team_id,
        )

        if _active_project_id_for_display(config, api_client) == project.id:
            write_project_context(project, config)

        if output == "json":
            output_data_as_json({"project": _project_payload(project)}, console)
            return

        console.print("[green]✓ Project updated[/green]")
        _print_project(
            project,
            active=project.id == _active_project_id_for_display(config, api_client),
        )
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("clear", epilog=PROJECT_CLEAR_HELP)
def clear_project() -> None:
    """Clear the active Lab project for this workspace."""
    env_project_id = os.getenv(PROJECT_CONTEXT_ENV)
    if clear_project_context():
        console.print("[green]✓ Active project cleared[/green]")
        if env_project_id and env_project_id.strip():
            console.print(
                f"[yellow]{PROJECT_CONTEXT_ENV} is set; this workspace will ignore it "
                "until you choose another active project.[/yellow]"
            )
    else:
        console.print("[yellow]No active project was set.[/yellow]")


@app.command("assign", epilog=PROJECT_ASSIGN_HELP)
def assign_artifact_to_project(
    kind: str = typer.Argument(..., help="Artifact kind: run, eval, or adapter"),
    artifact_id: str = typer.Argument(..., help="Training run, evaluation, or adapter ID"),
    project_ref: Optional[str] = typer.Argument(
        None,
        help="Project ID or slug. Defaults to the active project for this workspace.",
    ),
    team_id: Optional[str] = typer.Option(
        None,
        "--team-id",
        help="Team ID. Defaults to the active CLI team, if any.",
    ),
    move_adapters: bool = typer.Option(
        True,
        "--move-adapters/--no-move-adapters",
        help="For training runs, also add adapters created by the run to the project.",
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Add a training run, evaluation, or adapter to a Lab project."""
    validate_output_format(output, console)
    config = Config()
    api_client = APIClient()
    resolved_team_id = team_id if team_id is not None else config.team_id
    resolved_project_ref = project_ref or _active_project_ref_or_exit(config, api_client)
    normalized_kind = _normalize_artifact_kind(kind)

    try:
        project = ProjectsClient(api_client).get(
            resolved_project_ref,
            team_id=resolved_team_id,
        )

        adapters_updated: Optional[int] = None
        if normalized_kind == "training_run":
            _, adapters_updated = RLClient(api_client).update_run_project(
                artifact_id,
                project.id,
                operation="add",
                move_adapters=move_adapters,
            )
        elif normalized_kind == "evaluation":
            EvalsClient(api_client).update_evaluation(
                artifact_id,
                project_id=project.id,
            )
        else:
            DeploymentsClient(api_client).update_adapter_project(
                artifact_id,
                project.id,
                operation="add",
            )

        payload = _assignment_payload(
            kind=normalized_kind,
            artifact_id=artifact_id,
            project_id=project.id,
            project_slug=project.slug,
            adapters_updated=adapters_updated,
        )
        if output == "json":
            output_data_as_json(payload, console)
            return

        _print_assignment_result(payload)
    except (APIError, EvalsAPIError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("remove", epilog=PROJECT_REMOVE_HELP)
def remove_artifact_from_project(
    kind: str = typer.Argument(..., help="Artifact kind: run, eval, or adapter"),
    artifact_id: str = typer.Argument(..., help="Training run, evaluation, or adapter ID"),
    project_ref: Optional[str] = typer.Argument(
        None,
        help="Project ID or slug to remove. Omit to remove all project memberships.",
    ),
    team_id: Optional[str] = typer.Option(
        None,
        "--team-id",
        help="Team ID. Defaults to the active CLI team, if any.",
    ),
    move_adapters: bool = typer.Option(
        True,
        "--move-adapters/--no-move-adapters",
        help="For training runs, also remove adapters created by the run from the project.",
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Remove a training run, evaluation, or adapter from a Lab project."""
    validate_output_format(output, console)
    config = Config()
    api_client = APIClient()
    resolved_team_id = team_id if team_id is not None else config.team_id
    normalized_kind = _normalize_artifact_kind(kind)

    if normalized_kind == "evaluation" and project_ref:
        console.print(
            "[red]Error:[/red] Evaluation project removal clears the evaluation's project. "
            "Targeted removal from one project is not supported for evaluations."
        )
        console.print("[dim]Omit the project argument to clear the evaluation project.[/dim]")
        raise typer.Exit(1)

    try:
        project: Optional[Project] = None
        if project_ref:
            project = ProjectsClient(api_client).get(
                project_ref,
                team_id=resolved_team_id,
            )

        adapters_updated: Optional[int] = None
        if normalized_kind == "training_run":
            _, adapters_updated = RLClient(api_client).update_run_project(
                artifact_id,
                project.id if project else None,
                operation="remove" if project else "clear",
                move_adapters=move_adapters,
            )
        elif normalized_kind == "evaluation":
            EvalsClient(api_client).update_evaluation(
                artifact_id,
                clear_project=True,
            )
        else:
            DeploymentsClient(api_client).update_adapter_project(
                artifact_id,
                project.id if project else None,
                operation="remove" if project else "clear",
            )

        payload = _assignment_payload(
            kind=normalized_kind,
            artifact_id=artifact_id,
            project_id=project.id if project else None,
            project_slug=project.slug if project else None,
            adapters_updated=adapters_updated,
        )
        if output == "json":
            output_data_as_json(payload, console)
            return

        _print_assignment_result(payload, removed=True)
    except (APIError, EvalsAPIError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
