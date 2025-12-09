import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from prime_evals import EvalsAPIError, EvalsClient
from rich.console import Console

from prime_cli.core import APIClient

from .env_metadata import find_environment_metadata

console = Console()


def push_eval_results_to_hub(
    env_name: str,
    model: str,
    job_id: str,
    env_path: Optional[Path] = None,
    upstream_slug: Optional[str] = None,
) -> None:
    """
    Push evaluation results to Prime Evals Hub after vf-eval completes.

    This function:
    1. Locates the most recent vf-eval output directory
    2. Reads and parses metadata.json and results.jsonl
    3. Resolves environment ID (from metadata, upstream_slug, or by name)
    4. Converts results to Prime Evals API format
    5. Creates evaluation, pushes samples, and finalizes

    Args:
        env_name: Environment name (e.g., "simpleqa")
        model: Model identifier (e.g., "meta-llama/llama-3.1-70b-instruct")
        job_id: Unique job ID for tracking
        env_path: Optional path to the environment directory (defaults to current directory)
        upstream_slug: Optional upstream environment slug (e.g., "primeintellect/gpqa")
                      If provided, bypasses metadata file lookup
    """
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

    resolved_env_slug = None
    resolved_env_id = None

    if upstream_slug:
        resolved_env_slug = upstream_slug
        resolved_env_id = None
    else:
        # Search for environment metadata in multiple possible locations
        hub_metadata = find_environment_metadata(
            env_name=env_name,
            env_path=env_path,
            module_name=module_name,
        )

        if hub_metadata:
            try:
                # Prefer database ID if available (no resolution needed)
                if hub_metadata.get("environment_id"):
                    resolved_env_id = hub_metadata.get("environment_id")
                    owner = hub_metadata.get("owner")
                    name = hub_metadata.get("name")
                    resolved_env_slug = f"{owner}/{name}" if owner and name else None
                elif hub_metadata.get("owner") and hub_metadata.get("name"):
                    resolved_env_slug = f"{hub_metadata.get('owner')}/{hub_metadata.get('name')}"
                    resolved_env_id = None
                else:
                    resolved_env_slug = None
                    resolved_env_id = None
            except (KeyError, AttributeError) as e:
                console.print(
                    f"[yellow]Warning: Could not parse environment metadata: {e}[/yellow]"
                )
                resolved_env_slug = None
                resolved_env_id = None

    # Require accurate upstream for evaluation tracking
    if not resolved_env_slug and not resolved_env_id:
        console.print(
            "[yellow]No upstream environment found. Evaluation results will not be "
            "uploaded or viewable on the platform. Use `prime env push` to set an "
            "upstream, or use `--env-path` to specify the correct path to the "
            "environment.[/yellow]"
        )
        return None

    env_identifier = resolved_env_slug or resolved_env_id
    console.print(f"\n[blue]Uploading evaluation results, using upstream: {env_identifier}[/blue]")

    api_client = APIClient()

    if resolved_env_id:
        environments = [{"id": resolved_env_id}]
    elif resolved_env_slug:
        try:
            owner, name = resolved_env_slug.split("/", 1)
            response = api_client.get(f"/environmentshub/{owner}/{name}/@latest")
            details = response.get("data", response)
            env_id = details.get("id")
            if env_id:
                environments = [{"id": env_id}]
            else:
                environments = [{"slug": resolved_env_slug}]
        except Exception:
            environments = [{"slug": resolved_env_slug}]
    else:
        raise ValueError("No valid environment identifier found")
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

    evals_client = EvalsClient(api_client)

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

    if converted_results:
        evals_client.push_samples(eval_id, converted_results)

    evals_client.finalize_evaluation(eval_id, metrics=metrics)

    console.print("[green]✓ Successfully uploaded evaluation results[/green]")

    frontend_url = api_client.config.frontend_url
    eval_url = f"{frontend_url}/dashboard/evaluations/{eval_id}"
    console.print("\n[green]View results at:[/green]")
    console.print(eval_url)
