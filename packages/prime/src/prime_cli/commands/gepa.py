"""GEPA commands."""

import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypedDict, cast

import typer
from prime_evals import EvalsAPIError, EvalsClient, InvalidEvaluationError
from rich.console import Console
from typer.core import TyperGroup

from ..client import APIClient
from ..utils import output_data_as_json
from ..utils.display import get_eval_viewer_url
from ..utils.eval_push import load_results_jsonl
from ..verifiers_bridge import is_help_request, print_gepa_run_help, run_gepa_passthrough

console = Console()

GEPA_FRAMEWORK = "verifiers"
GEPA_EVAL_KIND = "gepa"


class GepaRunData(TypedDict):
    eval_name: str
    model_name: str
    env_id: str
    framework: str
    eval_kind: str
    metadata: dict[str, Any]
    results: list[dict[str, Any]]


class DefaultGroup(TyperGroup):
    def __init__(self, *args, default_cmd_name: str = "run", **kwargs):
        super().__init__(*args, **kwargs)
        self.default_cmd_name = default_cmd_name

    def parse_args(self, ctx, args):
        if not args:
            return super().parse_args(ctx, args)
        if args[0] in ("--help", "-h"):
            return super().parse_args(ctx, args)
        if args[0] in self.commands:
            return super().parse_args(ctx, args)
        args = [self.default_cmd_name] + list(args)
        return super().parse_args(ctx, args)

    def format_usage(self, ctx, formatter):
        formatter.write_usage(
            ctx.command_path,
            "run ENV_OR_CONFIG [ARGS]... | COMMAND [ARGS]...",
        )


app = typer.Typer(
    cls=DefaultGroup,
    help="Run GEPA prompt optimization.",
    no_args_is_help=True,
)


@app.command(
    "run",
    no_args_is_help=True,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def run_gepa_cmd(
    ctx: typer.Context,
    environment_or_config: str | None = typer.Argument(
        None,
        help="Environment name/slug or TOML config path",
    ),
) -> None:
    """Run optimization with local-first environment resolution."""
    passthrough_args = list(ctx.args)

    if is_help_request(environment_or_config or "", passthrough_args):
        print_gepa_run_help()
        raise typer.Exit(0)

    if environment_or_config is None:
        console.print("[red]Error:[/red] Missing argument 'ENV_OR_CONFIG'.")
        console.print("[dim]Example: prime gepa run wordle --max-calls 100[/dim]")
        raise typer.Exit(2)

    if environment_or_config.startswith("-"):
        console.print("[red]Error:[/red] Environment/config must be the first argument.")
        console.print("[dim]Example: prime gepa run wordle --max-calls 100[/dim]")
        raise typer.Exit(2)

    run_gepa_passthrough(environment_or_config, passthrough_args)


def _validate_gepa_run_dir(path_str: str) -> Path:
    run_dir = Path(path_str)

    if not run_dir.exists():
        raise FileNotFoundError(f"Path not found: {run_dir}")
    if not run_dir.is_dir():
        raise ValueError(f"Expected a GEPA run directory, but got file: {run_dir}")

    missing = [
        artifact
        for artifact in ("metadata.json", "results.jsonl")
        if not (run_dir / artifact).exists()
    ]
    if missing:
        raise ValueError(f"GEPA run directory '{run_dir}' is missing {', '.join(missing)}")

    return run_dir


def _require_metadata_string(metadata: dict[str, Any], field: str, metadata_path: Path) -> str:
    value = metadata.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"Missing required '{field}' field in {metadata_path}")
    return value


def _build_gepa_eval_name(metadata: dict[str, Any], env_id: str, model: str) -> str:
    for field in ("name", "eval_name", "run_name"):
        value = metadata.get(field)
        if type(value) is str and value:
            return value

    timestamp = None
    for field in ("timestamp", "created_at", "started_at", "start_time", "date", "run_id"):
        value = metadata.get(field)
        if value is not None:
            timestamp = str(value)
            break

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{env_id}--{model}--gepa--{timestamp}"


def _load_gepa_artifacts(run_dir: Path) -> dict[str, Any]:
    artifact_paths: dict[str, Any] = {
        "metadata_path": str((run_dir / "metadata.json").resolve()),
        "results_path": str((run_dir / "results.jsonl").resolve()),
    }

    system_prompt_path = run_dir / "system_prompt.txt"
    if system_prompt_path.exists():
        artifact_paths["system_prompt_path"] = str(system_prompt_path.resolve())
        artifact_paths["system_prompt"] = system_prompt_path.read_text(encoding="utf-8")

    pareto_frontier_path = run_dir / "pareto_frontier.jsonl"
    if pareto_frontier_path.exists():
        artifact_paths["pareto_frontier_path"] = str(pareto_frontier_path.resolve())
        artifact_paths["pareto_frontier_count"] = len(load_results_jsonl(pareto_frontier_path))

    return artifact_paths


def _build_gepa_environments(env_id: str) -> list[dict[str, str]]:
    if "/" in env_id:
        return [{"slug": env_id}]
    return [{"name": env_id}]


def _load_gepa_run(run_dir: Path) -> GepaRunData:
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    if type(metadata) is not dict:
        raise ValueError(f"Expected object metadata in {metadata_path}")

    env_id = _require_metadata_string(metadata, "env_id", metadata_path)
    model = _require_metadata_string(metadata, "model", metadata_path)

    results = load_results_jsonl(run_dir / "results.jsonl")
    eval_metadata = dict(metadata)
    eval_metadata.setdefault("eval_kind", GEPA_EVAL_KIND)
    eval_metadata.setdefault("framework", GEPA_FRAMEWORK)
    eval_metadata["artifacts"] = _load_gepa_artifacts(run_dir)

    return {
        "eval_name": _build_gepa_eval_name(metadata, env_id, model),
        "model_name": model,
        "env_id": env_id,
        "framework": metadata.get("framework") or GEPA_FRAMEWORK,
        "eval_kind": metadata.get("eval_kind") or GEPA_EVAL_KIND,
        "metadata": eval_metadata,
        "results": results,
    }


def _create_evaluation_supports_eval_kind(client: EvalsClient) -> bool:
    try:
        signature = inspect.signature(client.create_evaluation)
    except (TypeError, ValueError):
        return False
    return "eval_kind" in signature.parameters


def _create_gepa_evaluation(
    client: EvalsClient,
    eval_data: GepaRunData,
    environments: list[dict[str, str]],
    is_public: bool,
) -> dict[str, Any]:
    if _create_evaluation_supports_eval_kind(client):
        create_evaluation = cast(Callable[..., dict[str, Any]], client.create_evaluation)
        return create_evaluation(
            name=eval_data["eval_name"],
            environments=environments,
            model_name=eval_data["model_name"],
            dataset=eval_data["env_id"],
            framework=eval_data["framework"],
            metadata=eval_data["metadata"],
            is_public=is_public,
            eval_kind=eval_data["eval_kind"],
        )

    return client.create_evaluation(
        name=eval_data["eval_name"],
        environments=environments,
        model_name=eval_data["model_name"],
        dataset=eval_data["env_id"],
        framework=eval_data["framework"],
        metadata=eval_data["metadata"],
        is_public=is_public,
    )


def _push_gepa_run(config_path: str, is_public: bool = False) -> str:
    run_dir = _validate_gepa_run_dir(config_path)
    eval_data = _load_gepa_run(run_dir)
    console.print(f"[blue]✓ Loaded GEPA run:[/blue] {run_dir}")
    console.print()

    api_client = APIClient()
    client = EvalsClient(api_client)
    environments = _build_gepa_environments(eval_data["env_id"])

    console.print("[blue]Creating GEPA evaluation...[/blue]")
    create_response = _create_gepa_evaluation(client, eval_data, environments, is_public)
    eval_id = create_response.get("evaluation_id")
    if not eval_id:
        raise EvalsAPIError("Failed to get evaluation ID from create_evaluation response")

    console.print(f"[green]✓ Created evaluation:[/green] {eval_id}")
    console.print()

    results = eval_data["results"]
    if results:
        console.print(f"[blue]Pushing {len(results)} GEPA samples...[/blue]")
        client.push_samples(eval_id, results)
        console.print("[green]✓ Samples pushed successfully[/green]")
        console.print()

    console.print("[blue]Finalizing evaluation...[/blue]")
    client.finalize_evaluation(eval_id)
    console.print("[green]✓ Evaluation finalized[/green]")
    console.print()

    console.print("[green]✓ Success[/green]")
    console.print(f"[blue]Evaluation ID:[/blue] {eval_id}")
    eval_url = get_eval_viewer_url(eval_id)
    console.print(f"[dim]View results:[/dim] {eval_url}")

    return eval_id


@app.command("push", no_args_is_help=True)
def push_gepa_cmd(
    run_dir: str = typer.Argument(
        ...,
        help="GEPA run directory containing metadata.json and results.jsonl",
    ),
    output: str = typer.Option("pretty", "--output", "-o", help="json|pretty"),
    is_public: bool = typer.Option(
        False,
        "--public",
        help="Make the pushed evaluation public. Evaluations are private by default.",
    ),
) -> None:
    """Push GEPA run outputs to Prime Evals."""
    if output not in ("json", "pretty"):
        console.print("[red]Error:[/red] output must be one of: json, pretty")
        raise typer.Exit(1)

    try:
        eval_id = _push_gepa_run(run_dir, is_public=is_public)
        if output == "json":
            console.print()
            output_data_as_json({"evaluation_id": eval_id}, console)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON in metadata.json: {e}")
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e
    except InvalidEvaluationError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e
    except EvalsAPIError as e:
        console.print(f"[red]API Error:[/red] {e}")
        raise typer.Exit(1) from e
