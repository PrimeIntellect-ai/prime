"""Upload local eval runs to the Prime Evals Hub.

Reading and converting the on-disk artifacts is verifiers' job
(`verifiers.v1.cli.output.read_upload_data`); this module owns the online part:
resolving the upstream environment and calling the Evals API.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from prime_evals import EvalsAPIError, EvalsClient

from prime_cli.core import APIClient

from .display import get_eval_viewer_url
from .env_metadata import find_environment_metadata
from .plain import get_console

if TYPE_CHECKING:
    from verifiers.v1.cli.output import EvalUploadData

console = get_console()


def find_latest_eval_output_dir(env_name: str, model: str) -> Path:
    """The newest run directory in the legacy (v0) eval output layout."""
    module_name = env_name.replace("-", "_")
    env_model_str = f"{env_name}--{model.replace('/', '--')}"
    local_env_dir = Path("./environments") / module_name
    base_evals_dir = (
        local_env_dir / "outputs" / "evals" / env_model_str
        if local_env_dir.exists()
        else Path("./outputs") / "evals" / env_model_str
    )
    if not base_evals_dir.exists():
        raise FileNotFoundError(f"Evaluation output directory not found: {base_evals_dir}")
    subdirs = [d for d in base_evals_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No evaluation results found in {base_evals_dir}")
    return max(subdirs, key=lambda d: d.stat().st_mtime)


def push_eval_results_to_hub(
    env_name: str,
    job_id: str,
    upload: "EvalUploadData",
    upstream_slug: Optional[str] = None,
    env_path: Optional[Path] = None,
) -> None:
    """Create, populate, and finalize a platform evaluation from an `EvalUploadData` payload.

    The upstream environment is `upstream_slug` when given, else looked up from the local
    environment metadata; without either the upload is skipped with a hint.
    """
    resolved_env_slug = upstream_slug
    resolved_env_id = None
    if not resolved_env_slug:
        hub_metadata = find_environment_metadata(
            env_name=env_name,
            env_path=env_path,
            module_name=env_name.replace("-", "_"),
        )
        if hub_metadata:
            resolved_env_id = hub_metadata.get("environment_id")
            owner, name = hub_metadata.get("owner"), hub_metadata.get("name")
            if owner and name:
                resolved_env_slug = f"{owner}/{name}"

    if not resolved_env_slug and not resolved_env_id:
        console.print(
            "[yellow]No upstream environment found. Evaluation results will not be "
            "uploaded or viewable on the platform. Use `prime env push` to set an "
            "upstream, or use `--env-path` to specify the correct path to the "
            "environment.[/yellow]"
        )
        return

    env_identifier = resolved_env_slug or resolved_env_id
    console.print(f"\n[blue]Uploading evaluation results, using upstream: {env_identifier}[/blue]")

    api_client = APIClient()
    if resolved_env_id:
        environments = [{"id": resolved_env_id}]
    else:
        try:
            owner, name = resolved_env_slug.split("/", 1)
            response = api_client.get(f"/environmentshub/{owner}/{name}/@latest")
            details = response.get("data", response)
            env_id = details.get("id")
            environments = [{"id": env_id}] if env_id else [{"slug": resolved_env_slug}]
        except Exception:
            environments = [{"slug": resolved_env_slug}]

    evals_client = EvalsClient(api_client)
    eval_name = f"{env_name}--{upload.model_name}--{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    create_response = evals_client.create_evaluation(
        name=eval_name,
        environments=environments,
        model_name=upload.model_name,
        dataset=env_name,
        framework="verifiers",
        task_type=upload.metadata.get("task_type"),
        metadata={"framework": "verifiers", "job_id": job_id, **upload.metadata},
        metrics=upload.metrics,
        is_public=False,  # Private by default - only visible to the user who created it
    )

    eval_id = create_response.get("evaluation_id")
    if not eval_id:
        raise EvalsAPIError("Failed to get evaluation ID from create_evaluation response")

    if upload.results:
        evals_client.push_samples(eval_id, upload.results)

    evals_client.finalize_evaluation(eval_id, metrics=upload.metrics)

    console.print("[green]✓ Successfully uploaded evaluation results[/green]")

    eval_url = get_eval_viewer_url(eval_id)
    console.print("\n[green]View results at:[/green]")
    console.print(f"  [link={eval_url}]{eval_url}[/link]")
