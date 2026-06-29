"""Deployments command for managing model deployments for inference."""

from typing import List, Optional

from rich.table import Table

from prime_cli.command_configs import (
    DeploymentsCreateConfig,
    DeploymentsDeleteConfig,
    DeploymentsListConfig,
)

from ..api.deployments import DeploymentsClient
from ..client import APIClient, APIError
from ..core import Config
from ..utils import (
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from ..utils.prompt import confirm

console = get_console()

LIST_DEPLOYMENTS_JSON_HELP = json_output_help(
    ".models[] = {id, display_name, base_model, step, status, deployment_status, deployable?, ...}",
    ".total = number",
    ".page = number",
    ".per_page = number",
)

API_KEYS_DOCS_URL = "https://docs.primeintellect.ai/api-reference/api-keys"


def _print_inference_usage(base_model: str, adapter_id: str) -> None:
    model_id = f"{base_model}:{adapter_id}"
    console.print("\n[bold]Once deployed, you can run inference with:[/bold]")
    console.print(f'[dim]prime inference chat "{model_id}" "Hello" --max-tokens 100[/dim]')
    console.print(
        "[dim]For scripts or API clients, create a Platform API key at "
        f"{API_KEYS_DOCS_URL}, export it, then run cURL:[/dim]"
    )
    console.print(
        f"""
[dim]export PRIME_API_KEY=<insert_key_here>

curl -X POST https://api.pinference.ai/api/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $PRIME_API_KEY" \\
  -d '{{
    "model": "{model_id}",
    "messages": [{{"role": "user", "content": "Hello"}}],
    "max_tokens": 100
  }}'[/dim]"""
    )


def list_deployments(config: DeploymentsListConfig) -> None:
    """List adapters and their deployment status.

    Example:

        prime deployments list

        prime deployments list -n 50

        prime deployments list --page 2

        prime deployments list -o json

        prime deployments list --team <team_id>
    """
    team = config.team
    num = config.num
    page = config.page
    output = config.output

    validate_output_format(output, console)

    if num < 1 or page < 1:
        console.print("[red]Error:[/red] --num and --page must be at least 1")
        raise SystemExit(1)

    try:
        api_client = APIClient()
        deployments_client = DeploymentsClient(api_client)
        prime_config = Config()

        team_id = team or prime_config.team_id
        offset = (page - 1) * num

        models, total = deployments_client.list_adapters(team_id=team_id, limit=num, offset=offset)

        deployable_models: Optional[List[str]] = None
        try:
            deployable_models = deployments_client.get_deployable_models()
        except APIError:
            console.print("[dim]Warning: Could not fetch deployable models list.[/dim]")

        if output == "json":
            models_data = []
            for m in models:
                data = m.model_dump()
                if deployable_models is not None:
                    data["deployable"] = m.base_model in deployable_models
                models_data.append(data)
            output_data_as_json(
                {"models": models_data, "total": total, "page": page, "per_page": num},
                console,
            )
            return

        if not models:
            if page > 1:
                console.print("[yellow]No more results.[/yellow]")
            else:
                console.print("[yellow]No models found.[/yellow]")
                console.print("[dim]Train a model first, then deploy it here.[/dim]")
            return

        table = Table(title="Adapter Deployments")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="white")
        table.add_column("Base Model", style="magenta")
        table.add_column("Step", style="yellow", justify="right")
        table.add_column("Status", style="white")
        table.add_column("Deployable", style="white")
        table.add_column("Deployed At", style="dim")

        from ..utils.display import DEPLOYMENT_STATUS_COLORS

        for model in models:
            deployed_at = model.deployed_at.strftime("%Y-%m-%d %H:%M") if model.deployed_at else "-"
            base_model = model.base_model[:30] if len(model.base_model) > 30 else model.base_model
            status_color = DEPLOYMENT_STATUS_COLORS.get(model.deployment_status, "white")
            status = f"[{status_color}]{model.deployment_status}[/{status_color}]"

            if deployable_models is not None:
                is_deployable = model.base_model in deployable_models
                deployable = "[green]Yes[/green]" if is_deployable else "[red]No[/red]"
            else:
                deployable = "[dim]-[/dim]"

            table.add_row(
                model.id,
                model.display_name or "-",
                base_model,
                str(model.step) if model.step is not None else "-",
                status,
                deployable,
                deployed_at,
            )

        console.print(table)

        if total > page * num:
            console.print(
                f"\n[yellow]Showing page {page} of results. "
                f"Use --page {page + 1} to see more.[/yellow]"
            )
        else:
            console.print(f"\n[dim]Total: {total} adapter(s)[/dim]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def create_deployment(config: DeploymentsCreateConfig) -> None:
    """Deploy a model for inference.

    Makes the trained model available for inference requests.
    Model must be in READY status.

    Example:

        prime deployments create <model_id>

        prime deployments create <model_id> --yes
    """
    model_id = config.model_id
    yes = config.yes

    try:
        api_client = APIClient()
        deployments_client = DeploymentsClient(api_client)

        # Get model to validate status
        model = deployments_client.get_adapter(model_id)

        if model.status != "READY":
            console.print("[red]Error:[/red] Model is not ready for deployment.")
            console.print(f"Current status: [yellow]{model.status}[/yellow]")
            console.print("[dim]Only models with READY status can be deployed.[/dim]")
            raise SystemExit(1)

        if model.deployment_status == "DEPLOYED":
            console.print("[yellow]Model is already deployed.[/yellow]")
            raise SystemExit(0)

        if model.deployment_status in ("DEPLOYING", "UNLOADING"):
            console.print("[yellow]Model deployment is in progress.[/yellow]")
            console.print(f"Current status: {model.deployment_status}")
            raise SystemExit(1)

        # Check if base model supports LoRA deployment
        try:
            deployable_models = deployments_client.get_deployable_models()
            if model.base_model not in deployable_models:
                console.print(
                    "[red]Error:[/red] Base model is not currently available for LoRA deployment."
                )
                console.print(f"  Base model: [yellow]{model.base_model}[/yellow]")
                raise SystemExit(1)
        except APIError:
            console.print(
                "[dim]Warning: Could not verify base model deployability. Proceeding anyway.[/dim]"
            )

        # Show model details and confirm
        console.print("[bold]Deploying model:[/bold]")
        console.print(f"  ID: {model.id}")
        if model.display_name:
            console.print(f"  Name: {model.display_name}")
        console.print(f"  Base Model: {model.base_model}")
        console.print()

        if not yes:
            confirmed = confirm("Are you sure you want to deploy this model?")
            if not confirmed:
                console.print("Cancelled.")
                raise SystemExit(0)

        # Deploy the model
        updated_model = deployments_client.deploy_adapter(model_id)

        console.print("[green]Deployment initiated successfully![/green]")
        console.print(f"Status: [yellow]{updated_model.deployment_status}[/yellow]")
        console.print("\n[dim]The model is being deployed. This may take a few minutes.[/dim]")
        console.print("[dim]Use 'prime deployments list' to check deployment status.[/dim]")

        _print_inference_usage(model.base_model, model.id)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def delete_deployment(config: DeploymentsDeleteConfig) -> None:
    """Unload a model from inference.

    Removes the model from serving. Model files remain stored
    and can be deployed again.

    Example:

        prime deployments delete <model_id>
    """
    model_id = config.model_id

    try:
        api_client = APIClient()
        deployments_client = DeploymentsClient(api_client)

        # Get model to validate status
        model = deployments_client.get_adapter(model_id)

        if model.deployment_status == "NOT_DEPLOYED":
            console.print("[yellow]Model is not deployed.[/yellow]")
            raise SystemExit(0)

        if model.deployment_status in ("DEPLOYING", "UNLOADING"):
            console.print("[yellow]Model deployment is in progress.[/yellow]")
            console.print(f"Current status: {model.deployment_status}")
            raise SystemExit(1)

        if model.deployment_status not in ("DEPLOYED", "DEPLOY_FAILED", "UNLOAD_FAILED"):
            console.print("[red]Error:[/red] Cannot unload model in current state.")
            console.print(f"Current status: {model.deployment_status}")
            raise SystemExit(1)

        # Unload the model
        updated_model = deployments_client.unload_adapter(model_id)

        console.print("[green]Unload initiated successfully![/green]")
        console.print(f"Status: [yellow]{updated_model.deployment_status}[/yellow]")
        console.print("\n[dim]The model is being unloaded.[/dim]")
        console.print("[dim]Use 'prime deployments list' to check status.[/dim]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
