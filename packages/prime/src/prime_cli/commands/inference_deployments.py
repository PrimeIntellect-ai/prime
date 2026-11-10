"""`prime inference deployments`: list, inspect and scale dedicated deployments."""

from typing import Any, Dict

import typer
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ..api.inference_deployments import InferenceDeploymentsClient
from ..client import APIClient, APIError
from ..core import Config
from ..utils import (
    PlainTyper,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)

DEPLOYMENT_JSON = (
    "{id, deploymentName, status, statusPhase?, errorMessage?, servedModelAlias,",
    " endpointUrl?, checkpoint?, isOnDemand, replicas?, gpusPerReplica?,",
    " maxReplicas?, scaleState?, createdAt, dashboardUrl}",
)
GET_JSON_HELP = json_output_help(". = " + "".join(DEPLOYMENT_JSON))
LIST_JSON_HELP = json_output_help(".data[] = " + "".join(DEPLOYMENT_JSON))

app = PlainTyper(
    help="Manage dedicated inference deployments of the selected team, or your own",
    no_args_is_help=True,
)
console = get_console()

JSON_OPTION = typer.Option(False, "--json", help="Print the API response as JSON")


@app.command("get", epilog=GET_JSON_HELP)
def get_deployment(deployment_id: str, json_output: bool = JSON_OPTION) -> None:
    """Show one deployment."""
    deployment = _call(lambda client: client.get_deployment(deployment_id))
    if json_output:
        output_data_as_json(deployment, console)
        return
    _print_detail(deployment)


@app.command("list", epilog=LIST_JSON_HELP)
def list_deployments(
    json_output: bool = JSON_OPTION,
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List the selected team's deployments, or your personal ones without a team."""
    if json_output:
        output = "json"
    validate_output_format(output, console)

    envelope = _call(lambda client: client.list_deployments(Config().team_id))
    if output == "json":
        output_data_as_json(envelope, console)
        return
    if not envelope["data"]:
        console.print("No inference deployments found.")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="cyan")
    table.add_column("NAME")
    table.add_column("STATUS")
    table.add_column("REPLICAS", justify="right")
    table.add_column("SOURCE")
    for deployment in envelope["data"]:
        table.add_row(
            Text(deployment["id"]),
            Text(deployment["deploymentName"]),
            Text(deployment["status"]),
            Text(_replicas(deployment)),
            Text("on-demand" if deployment["isOnDemand"] else "team cluster"),
        )
    console.print(table)


@app.command("scale", epilog=GET_JSON_HELP)
def scale_deployment(
    deployment_id: str,
    replicas: int = typer.Argument(..., min=1, help="Desired replica count"),
    json_output: bool = JSON_OPTION,
) -> None:
    """Set a running deployment's replica count. On-demand scale-ups reserve and bill the
    added GPUs."""
    deployment = _call(lambda client: client.scale_deployment(deployment_id, replicas))
    if json_output:
        output_data_as_json(deployment, console)
        return
    console.print(
        f"Scaling [cyan]{escape(deployment['deploymentName'])}[/cyan] to "
        f"{deployment['replicas']} replicas."
    )
    console.print(f"[dim]Track it with `prime inference deployments get {deployment_id}`.[/dim]")


def _call(request) -> Dict[str, Any]:
    try:
        return request(InferenceDeploymentsClient(APIClient()))
    except APIError as error:
        console.print(f"[red]Error:[/red] {escape(str(error))}")
        raise typer.Exit(1) from error


def _print_detail(deployment: Dict[str, Any]) -> None:
    rows = [
        ("ID", deployment["id"]),
        ("Name", deployment["deploymentName"]),
        ("Status", deployment["status"]),
        ("Replicas", _replicas(deployment)),
        ("Source", "on-demand" if deployment["isOnDemand"] else "team cluster"),
        ("Model alias", deployment["servedModelAlias"]),
        ("Endpoint", deployment.get("endpointUrl") or "—"),
        ("Dashboard", deployment["dashboardUrl"]),
    ]
    for label, value in rows:
        console.print(f"[bold]{label}:[/bold] {escape(str(value))}")


def _replicas(deployment: Dict[str, Any]) -> str:
    replicas = deployment.get("replicas")
    if replicas is None:
        return "—"
    per = deployment.get("gpusPerReplica")
    shape = f"{replicas} x {per} GPU" if per else str(replicas)
    scale_state = deployment.get("scaleState")
    return f"{shape} ({scale_state})" if scale_state else shape
