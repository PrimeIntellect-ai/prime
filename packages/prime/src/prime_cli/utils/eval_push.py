import json
from datetime import datetime
from pathlib import Path

from prime_core import APIClient
from prime_evals import EvalsAPIError, EvalsClient
from rich.console import Console

from .env_metadata import get_environment_metadata

console = Console()


def push_eval_results_to_hub(
    env_name: str,
    model: str,
    job_id: str,
) -> None:
    """
    Push evaluation results to Prime Evals Hub after vf-eval completes.

    This function:
    1. Locates the most recent vf-eval output directory
    2. Reads and parses metadata.json and results.jsonl
    3. Resolves environment ID (from metadata or by name)
    4. Converts results to Prime Evals API format
    5. Creates evaluation, pushes samples, and finalizes

    Args:
        environment: Environment name (e.g., "simpleqa")
        model: Model identifier (e.g., "meta-llama/llama-3.1-70b-instruct")
        job_id: Unique job ID for tracking
    """
    console.print("\n[blue]Pushing evaluation results to hub...[/blue]")

    # Step 1: Find the output directory
    module_name = env_name.replace("-", "_")
    model_name = model.replace("/", "--")
    env_model_str = f"{env_name}--{model_name}"

    local_env_dir = Path("./environments") / module_name
    if local_env_dir.exists():
        base_evals_dir = local_env_dir / "outputs" / "evals" / env_model_str
    else:
        base_evals_dir = Path("./outputs") / "evals" / env_model_str

    if not base_evals_dir.exists():
        raise FileNotFoundError(f"Evaluation output directory not found: {base_evals_dir}")

    subdirs = [d for d in base_evals_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No evaluation results found in {base_evals_dir}")

    output_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    console.print(f"[dim]Found results in: {output_dir}[/dim]")

    metadata_path = output_dir / "metadata.json"
    results_path = output_dir / "results.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {output_dir}")
    if not results_path.exists():
        raise FileNotFoundError(f"results.jsonl not found in {output_dir}")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    results_samples = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results_samples.append(json.loads(line))

    console.print(f"[dim]Loaded {len(results_samples)} samples[/dim]")

    # Resolve environment slug from metadata if available
    env_dir = Path("./environments") / module_name
    hub_metadata = get_environment_metadata(env_dir)
    resolved_env_slug = None

    if hub_metadata and hub_metadata.get("owner") and hub_metadata.get("name"):
        resolved_env_slug = f"{hub_metadata.get('owner')}/{hub_metadata.get('name')}"
        console.print(f"[blue]✓ Found environment:[/blue] {env_name}")
    else:
        console.print(f"[blue]Using environment name:[/blue] {env_name} (will be resolved)")
        resolved_env_slug = None
        resolved_env_id = None

    if resolved_env_id:
        environments = [{"id": resolved_env_id}]
    elif resolved_env_slug:
        environments = [{"slug": resolved_env_slug}]
    else:
        environments = [{"name": env_name}]
    metrics = {k: v for k, v in metadata.items() if k.startswith("avg_")}

    eval_metadata = {"framework": "verifiers", "job_id": job_id, **metadata}

    converted_results = [
        {
            "example_id": sample.get("id", 0),
            "reward": sample.get("reward", 0.0),
            **{k: v for k, v in sample.items() if k not in {"id", "reward"}},
        }
        for sample in results_samples
    ]

    eval_name = f"{env_name}--{model}--{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    api_client = APIClient()
    evals_client = EvalsClient(api_client)

    console.print("[blue]Creating evaluation...[/blue]")
    create_response = evals_client.create_evaluation(
        name=eval_name,
        environments=environments,
        model_name=model,
        dataset=env_name,
        framework="verifiers",
        task_type=metadata.get("task_type"),
        metadata=eval_metadata,
        metrics=metrics,
        is_public=False,  # Private by default - only visible to the user who created it
    )

    eval_id = create_response.get("evaluation_id")
    if not eval_id:
        raise EvalsAPIError("Failed to get evaluation ID from create_evaluation response")

    console.print(f"[green]✓ Created evaluation:[/green] {eval_id}")

    if converted_results:
        console.print(f"[blue]Pushing {len(converted_results)} samples...[/blue]")
        evals_client.push_samples(eval_id, converted_results)
        console.print("[green]✓ Samples pushed successfully[/green]")

    console.print("[blue]Finalizing evaluation...[/blue]")
    evals_client.finalize_evaluation(eval_id, metrics=metrics)
    console.print("[green]✓ Evaluation finalized[/green]\n")

    console.print("[green]✓ Successfully pushed to hub[/green]")

    frontend_url = api_client.config.frontend_url
    eval_url = f"{frontend_url}/dashboard/evaluations/{eval_id}"
    console.print("\n[green]View at:[/green]")
    console.print(eval_url)
