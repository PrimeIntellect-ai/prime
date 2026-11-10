"""`prime inference` deployment commands: catalog, deploy, deploy-checkpoint, deployments."""

from __future__ import annotations

import re
import shlex
import time
from typing import Any, Dict, List, NoReturn, Optional, Tuple

import typer
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.table import Table
from rich.text import Text

from ..api.inference_deployments import InferenceDeployment, InferenceDeploymentsClient
from ..core import APIClient, APIError, Config
from ..utils import (
    PlainTyper,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)

console = get_console()

_DEPLOYMENT_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_FAILED_STATUSES = {"FAILED", "STOPPED", "TERMINATED", "CANCELLED"}
_POLL_INTERVAL_SECONDS = 5.0
_STATUS_COLORS = {"RUNNING": "green", "FAILED": "red", "PENDING": "yellow"}

DEPLOYMENT_JSON_HELP = json_output_help(
    "InferenceDeployment: {id, deploymentName, status, statusPhase?, errorMessage?, "
    "servedModelAlias, endpointUrl?, checkpoint?, isOnDemand, createdAt, dashboardUrl}",
)
CATALOG_JSON_HELP = json_output_help(
    "InferenceCatalogResponse passthrough: the deployable catalog models plus the "
    "team's clusters that can serve them",
)


def _err_console(output: str) -> Console:
    # JSON mode keeps stdout strictly data: diagnostics go to stderr.
    return get_console(stderr=True) if output == "json" else console


def _fail(err: Console, message: str) -> NoReturn:
    err.print(f"[red]Error:[/red] {rich_escape(message)}")
    raise typer.Exit(1)


def _client() -> InferenceDeploymentsClient:
    return InferenceDeploymentsClient(APIClient())


def _first(item: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def _list_field(data: Dict[str, Any], *keys: str) -> List[Dict[str, Any]]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
    return []


def _cell(value: Any) -> Text:
    # Catalog values are upstream data; Text renders them literally so
    # Rich markup inside a name cannot restyle the table.
    return Text("—" if value in (None, "") else str(value))


def _gpu_cell(item: Dict[str, Any]) -> Text:
    gpu_type = _first(item, "gpuType", "gpu_type")
    gpu_count = _first(item, "gpuCount", "gpu_count", "gpusPerReplica")
    if gpu_type and gpu_count:
        return _cell(f"{gpu_count}x {gpu_type}")
    return _cell(gpu_type or gpu_count)


def _print_deployment(deployment: InferenceDeployment) -> None:
    color = _STATUS_COLORS.get(deployment.status, "yellow")
    status = deployment.status
    if deployment.status_phase:
        status += f" ({deployment.status_phase})"
    console.print(f"[bold]Deployment {rich_escape(deployment.id)}[/bold]\n")
    if deployment.deployment_name:
        console.print(f"  Name: {rich_escape(deployment.deployment_name)}")
    console.print(f"  Status: [{color}]{rich_escape(status)}[/{color}]")
    if deployment.served_model_alias:
        console.print(f"  Model: [magenta]{rich_escape(deployment.served_model_alias)}[/magenta]")
    if deployment.checkpoint:
        console.print(f"  Checkpoint: {rich_escape(str(deployment.checkpoint))}")
    if deployment.is_on_demand is not None:
        console.print(f"  Capacity: {'on-demand' if deployment.is_on_demand else 'team cluster'}")
    if deployment.endpoint_url:
        console.print(f"  Endpoint: [cyan]{rich_escape(deployment.endpoint_url)}[/cyan]")
    if deployment.created_at:
        console.print(f"  Created: [dim]{rich_escape(deployment.created_at)}[/dim]")
    if deployment.dashboard_url:
        console.print(f"  Dashboard: {rich_escape(deployment.dashboard_url)}")
    if deployment.error_message:
        console.print(f"  Error: [red]{rich_escape(deployment.error_message)}[/red]")
    if deployment.status == "RUNNING" and deployment.endpoint_url:
        _print_usage_snippet(deployment)


def _print_usage_snippet(deployment: InferenceDeployment) -> None:
    body = (
        '{"model": "'
        + (deployment.served_model_alias or "<model>")
        + '", "messages": [{"role": "user", "content": "Hello"}]}'
    )
    url = f"{(deployment.endpoint_url or '').rstrip('/')}/chat/completions"
    snippet = (
        f"curl {shlex.quote(url)} \\\n"
        '  -H "Authorization: Bearer $PRIME_API_KEY" \\\n'
        '  -H "Content-Type: application/json" \\\n'
        f"  -d {shlex.quote(body)}"
    )
    console.print("\n[bold]Try it:[/bold]")
    console.print(snippet, markup=False, highlight=False, soft_wrap=True)


def _wait_for_deployment(
    client: InferenceDeploymentsClient,
    raw: Dict[str, Any],
    timeout: int,
    err: Console,
) -> Tuple[InferenceDeployment, Dict[str, Any]]:
    deployment = InferenceDeployment.model_validate(raw)
    deadline = time.monotonic() + timeout
    last_status = None
    while True:
        phase = f" ({deployment.status_phase})" if deployment.status_phase else ""
        current = deployment.status + phase
        if current != last_status:
            err.print(f"[dim]{rich_escape(deployment.id)}: {rich_escape(current)}[/dim]")
            last_status = current
        if deployment.status == "RUNNING" or deployment.status in _FAILED_STATUSES:
            return deployment, raw
        if time.monotonic() >= deadline:
            _fail(
                err,
                f"Timed out after {timeout}s waiting for deployment {deployment.id} "
                f"(status: {deployment.status}). Check it later with "
                f"`prime inference deployments get {deployment.id}`.",
            )
        time.sleep(_POLL_INTERVAL_SECONDS)
        raw = client.get_deployment(deployment.id)
        deployment = InferenceDeployment.model_validate(raw)


def _finish_deploy(
    client: InferenceDeploymentsClient,
    raw: Dict[str, Any],
    wait: bool,
    timeout: int,
    output: str,
) -> None:
    err = _err_console(output)
    if wait:
        deployment, raw = _wait_for_deployment(client, raw, timeout, err)
    else:
        deployment = InferenceDeployment.model_validate(raw)

    if output == "json":
        output_data_as_json(raw, console)
    else:
        _print_deployment(deployment)
        if not wait and deployment.status != "RUNNING":
            console.print(
                f"\n[dim]Track it with `prime inference deployments get {deployment.id}` "
                "or rerun with --wait.[/dim]"
            )

    if deployment.status in _FAILED_STATUSES:
        raise typer.Exit(1)


def catalog_command(
    team: Optional[str] = typer.Option(
        None, "--team", "-t", help="Team ID (defaults to the selected account context)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List models you can deploy, plus your team's clusters that can serve them."""
    validate_output_format(output, console)
    err = _err_console(output)
    try:
        data = _client().get_catalog(team or Config().team_id)
    except APIError as e:
        _fail(err, str(e))

    if output == "json":
        output_data_as_json(data, console)
        return

    models = _list_field(data, "models", "catalog", "items", "data")
    clusters = _list_field(data, "clusters", "primeClusters")
    if not models:
        console.print("[yellow]No deployable models in the catalog.[/yellow]")
        return

    table = Table(title="Inference Catalog")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Model")
    table.add_column("GPUs")
    for m in models:
        table.add_row(
            _cell(_first(m, "id", "modelCatalogId")),
            _cell(_first(m, "displayName", "name", "modelName", "huggingFaceId", "model")),
            _gpu_cell(m),
        )
    console.print(table)

    if clusters:
        cluster_table = Table(title="Team Clusters")
        cluster_table.add_column("ID", style="cyan", no_wrap=True)
        cluster_table.add_column("Name")
        cluster_table.add_column("GPUs")
        for c in clusters:
            cluster_table.add_row(
                _cell(_first(c, "id", "primeClusterId")),
                _cell(_first(c, "displayName", "name")),
                _gpu_cell(c),
            )
        console.print(cluster_table)

    console.print(
        "\n[dim]Deploy with `prime inference deploy <ID>`; add --cluster <cluster-id> to "
        "use a team cluster instead of Prime's on-demand pool.[/dim]"
    )


def deploy_command(
    model_catalog_id: str = typer.Argument(
        ..., help="Catalog model ID (see `prime inference catalog`)"
    ),
    team: Optional[str] = typer.Option(
        None, "--team", "-t", help="Team ID (defaults to the selected account context)"
    ),
    cluster: Optional[str] = typer.Option(
        None, "--cluster", help="Team cluster ID to serve on (default: Prime's on-demand pool)"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Deployment name (lowercase letters, digits and dashes)"
    ),
    replicas: int = typer.Option(1, "--replicas", "-r", min=1, max=4, help="Replicas (1-4)"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait until RUNNING or FAILED"),
    timeout: int = typer.Option(
        1800, "--timeout", min=1, help="Seconds to --wait before giving up"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Deploy a catalog model as a dedicated OpenAI-compatible endpoint.

    Example:

        prime inference deploy <model-catalog-id> --name my-endpoint --wait
    """
    validate_output_format(output, console)
    err = _err_console(output)
    if name is not None and not _DEPLOYMENT_NAME_RE.match(name):
        _fail(
            err,
            "--name must be lowercase letters, digits and dashes, "
            "starting and ending with a letter or digit.",
        )
    try:
        client = _client()
        raw = client.deploy(
            model_catalog_id,
            team_id=team or Config().team_id,
            deployment_name=name,
            replicas=replicas,
            prime_cluster_id=cluster,
        )
        _finish_deploy(client, raw, wait, timeout, output)
    except APIError as e:
        _fail(err, str(e))


def deploy_checkpoint_command(
    run_id: str = typer.Argument(..., help="Training run ID"),
    step: int = typer.Option(..., "--step", "-s", min=0, help="Checkpoint step to deploy"),
    team: Optional[str] = typer.Option(
        None, "--team", "-t", help="Team ID (defaults to the run's team)"
    ),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait until RUNNING or FAILED"),
    timeout: int = typer.Option(
        1800, "--timeout", min=1, help="Seconds to --wait before giving up"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Deploy a training run's checkpoint as an OpenAI-compatible endpoint.

    Returns the existing deployment if that checkpoint is already served.

    Example:

        prime inference deploy-checkpoint <run-id> --step 100 --wait
    """
    validate_output_format(output, console)
    err = _err_console(output)
    try:
        client = _client()
        raw = client.deploy_checkpoint(run_id, step, team_id=team)
        _finish_deploy(client, raw, wait, timeout, output)
    except APIError as e:
        _fail(err, str(e))


deployments_app = PlainTyper(help="Inspect inference deployments", no_args_is_help=True)


@deployments_app.command("get", epilog=DEPLOYMENT_JSON_HELP)
def get_deployment(
    deployment_id: str = typer.Argument(..., help="Deployment ID"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show a deployment's status and endpoint."""
    validate_output_format(output, console)
    try:
        raw = _client().get_deployment(deployment_id)
    except APIError as e:
        _fail(_err_console(output), str(e))
    if output == "json":
        output_data_as_json(raw, console)
        return
    _print_deployment(InferenceDeployment.model_validate(raw))
