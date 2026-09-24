"""`prime volumes`: named volumes FFT runs write their outputs to.

A volume is a PVC in your team's (or personal) namespace on the cluster it
was created on. `prime train config.toml --volume <name>` makes the run
write under `runs/<runId>/` on it, and it outlives every run.
"""

import typer
from rich.table import Table

from prime_cli.api.training import HostedTrainingClient
from prime_cli.core import APIClient, APIError, Config

from ..utils import (
    PlainTyper,
    confirm_or_skip,
    get_console,
    output_data_as_json,
    validate_output_format,
)

app = PlainTyper(help="Manage volumes for full-FT run outputs", no_args_is_help=True)
console = get_console()


def _client() -> tuple[HostedTrainingClient, str | None]:
    return HostedTrainingClient(APIClient()), Config().team_id


@app.command()
def create(
    name: str = typer.Argument(..., help="Volume name (lowercase letters, digits, '-')"),
    size: str = typer.Option("1Ti", "--size", help="Size, e.g. 500Gi or 2Ti. Can grow later."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Create a volume on your team's (or personal) cluster."""
    validate_output_format(output, console)
    client, team_id = _client()
    try:
        volume = client.create_volume(name, size, team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    if output == "json":
        output_data_as_json(volume.model_dump(by_alias=True), console)
        return
    console.print(f"[green]Creating volume {volume.name} ({volume.size}).[/green]")
    console.print(f"Use it with: prime train config.toml --volume {volume.name}")


@app.command("list")
def list_volumes(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your volumes."""
    validate_output_format(output, console)
    client, team_id = _client()
    try:
        volumes = client.list_volumes(team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    if output == "json":
        output_data_as_json([v.model_dump(by_alias=True) for v in volumes], console)
        return
    table = Table("Name", "Size", "Status", "Namespace", "Created")
    for v in volumes:
        table.add_row(v.name, v.size or "-", v.status, v.namespace, v.created_at or "-")
    console.print(table)


@app.command()
def resize(
    name: str = typer.Argument(..., help="Volume name"),
    size: str = typer.Option(..., "--size", help="New size, larger than the current one"),
) -> None:
    """Grow a volume in place. Running pods see the new size; volumes can't shrink."""
    client, team_id = _client()
    try:
        volume = client.resize_volume(name, size, team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(f"[green]Volume {volume.name} is now {volume.size}.[/green]")


@app.command()
def delete(
    name: str = typer.Argument(..., help="Volume name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a volume and everything on it."""
    if not confirm_or_skip(f"Delete volume {name} and all run data on it?", yes):
        raise typer.Exit(0)
    client, team_id = _client()
    try:
        client.delete_volume(name, team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(f"[green]Deleting volume {name}.[/green]")
