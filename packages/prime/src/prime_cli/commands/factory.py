"""`prime factory`: Model Factory resources.

`prime factory list` shows the dedicated GPU clusters assigned to you or
your team — the same fleet as the dashboard's Model Factory tab — with the
machine name and id that `prime volumes create --cluster` accepts.
"""

import typer
from rich.table import Table

from prime_cli.api.training import HostedTrainingClient
from prime_cli.core import APIClient, APIError, Config

from ..utils import (
    PlainTyper,
    get_console,
    output_data_as_json,
    validate_output_format,
)

app = PlainTyper(
    help="Manage Model Factory resources",
    no_args_is_help=True,
)
console = get_console()


def _client() -> tuple[HostedTrainingClient, str | None]:
    return HostedTrainingClient(APIClient()), Config().team_id


@app.command("list")
def list_clusters(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List the clusters assigned to you or your team."""
    validate_output_format(output, console)
    client, team_id = _client()
    try:
        clusters = client.list_clusters(team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    if output == "json":
        output_data_as_json([c.model_dump(by_alias=True) for c in clusters], console)
        return
    if not clusters:
        console.print("[dim]No clusters assigned yet.[/dim]")
        return
    table = Table("Name", "ID", "GPU type", "GPUs", "Status")
    for c in clusters:
        gpus = f"{c.free_gpus}/{c.total_gpus}" if c.free_gpus is not None and c.total_gpus else "-"
        # A cordoned cluster is still "online" (the controller runs) but
        # cannot be picked for `volumes create --cluster`. Both facts are
        # shown, since this table is the discovery surface for that flag.
        status = "cordoned" if c.cordoned else c.status
        table.add_row(c.name, c.cluster_id, c.gpu_type or "-", gpus, status)
    console.print(table)
