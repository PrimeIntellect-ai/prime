import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from prime_evals import EvalsAPIError, EvalsClient

from prime_cli.core import APIClient

from .display import get_eval_viewer_url
from .env_metadata import find_environment_metadata
from .plain import get_console

console = get_console()


def load_results_jsonl(path: Path) -> list[dict]:
    """
    Load and parse a results.jsonl file, skipping invalid lines with warnings.

    Args:
        path: Path to the results.jsonl file

    Returns:
        List of valid dict samples from the file
    """
    results = []
    skipped = []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if not isinstance(sample, dict):
                    skipped.append((line_num, f"expected dict, got {type(sample).__name__}"))
                    continue
                results.append(sample)
            except json.JSONDecodeError:
                skipped.append((line_num, "invalid JSON"))

    if skipped:
        preview = [f"line {num}: {reason}" for num, reason in skipped[:5]]
        suffix = ", ..." if len(skipped) > 5 else ""
        console.print(
            f"[yellow]Warning: Skipped {len(skipped)} invalid lines in results.jsonl "
            f"({', '.join(preview)}{suffix})[/yellow]"
        )

    return results


def extract_verifiers_metrics(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if k.startswith("avg_")}


_SAMPLE_FIELD_ALIASES = {
    "exampleId": "example_id",
    "rolloutNumber": "rollout_number",
    "numSteps": "num_steps",
    "totalTime": "total_time",
    "latencyMs": "latency_ms",
}

_STANDARD_SAMPLE_FIELDS = {
    "evaluation_id",
    "sample_id",
    "example_id",
    "reward",
    "task",
    "prompt",
    "completion",
    "answer",
    "score",
    "correct",
    "num_steps",
    "total_time",
    "latency_ms",
    "rollout_number",
}

_RESERVED_RESULT_KEYS = {
    "id",
    "info",
    "metadata",
    *_STANDARD_SAMPLE_FIELDS,
    *_SAMPLE_FIELD_ALIASES,
}


def _numeric_or_none(value: object) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _timing_total_time(timing: object) -> float | None:
    if not isinstance(timing, dict):
        return None

    timing_dict = {str(key): value for key, value in timing.items()}

    total_ms = _numeric_or_none(timing_dict.get("total_ms"))
    if total_ms is None:
        total_ms = _numeric_or_none(timing_dict.get("totalMs"))
    if total_ms is not None:
        return float(total_ms) / 1000.0

    total = _numeric_or_none(timing_dict.get("total"))
    return float(total) if total is not None else None


def normalize_verifiers_result_sample(sample: dict) -> dict:
    normalized = {
        "example_id": sample.get("example_id", sample.get("exampleId", sample.get("id", 0))),
        "reward": sample.get("reward", 0.0),
    }

    for key in _STANDARD_SAMPLE_FIELDS - {"example_id", "reward"}:
        if key in sample:
            normalized[key] = sample[key]

    for source_key, target_key in _SAMPLE_FIELD_ALIASES.items():
        if source_key in sample and target_key not in normalized:
            normalized[target_key] = sample[source_key]

    timing_total_time = _timing_total_time(sample.get("timing"))
    if timing_total_time is not None:
        normalized.setdefault("total_time", timing_total_time)
        normalized.setdefault("latency_ms", int(round(timing_total_time * 1000)))

    info = {}
    for key in ("metadata", "info"):
        value = sample.get(key)
        if isinstance(value, dict):
            info.update(value)

    for key, value in sample.items():
        if key not in _RESERVED_RESULT_KEYS:
            info.setdefault(key, value)

    if info:
        normalized["info"] = info

    return normalized


def normalize_verifiers_result_samples(samples: list[dict]) -> list[dict]:
    return [normalize_verifiers_result_sample(sample) for sample in samples]


def push_eval_results_to_hub(
    env_name: str,
    model: str,
    job_id: str,
    env_path: Optional[Path] = None,
    upstream_slug: Optional[str] = None,
) -> None:
    """
    Push evaluation results to Prime Evals Hub after `prime eval run` completes.

    This function:
    1. Locates the most recent evaluation output directory
    2. Reads and parses metadata.json and results.jsonl
    3. Resolves environment ID (from metadata, upstream_slug, or by name)
    4. Converts results to Prime Evals API format
    5. Creates evaluation, pushes samples, and finalizes

    Args:
        env_name: Environment name (e.g., "simpleqa")
        model: Model identifier (e.g., "openai/gpt-4.1-mini")
        job_id: Unique job ID for tracking
        env_path: Optional path to the environment directory (defaults to current directory)
        upstream_slug: Optional upstream environment slug (e.g., "primeintellect/wordle")
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

    results_samples = load_results_jsonl(results_path)

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
    metrics = extract_verifiers_metrics(metadata)

    eval_metadata = {"framework": "verifiers", "job_id": job_id, **metadata}

    converted_results = normalize_verifiers_result_samples(results_samples)

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

    eval_url = get_eval_viewer_url(eval_id)
    console.print("\n[green]View results at:[/green]")
    console.print(f"  [link={eval_url}]{eval_url}[/link]")
