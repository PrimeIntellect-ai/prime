from datetime import datetime, timezone
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from prime_evals import APIClient, EvalsClient, InvalidEvaluationError, push_verifiers_eval_to_hub
from rich.console import Console
from verifiers.utils.eval_runner import (
    eval_environments_parallel,
    serialize_messages_for_hub,
)

console = Console()


async def run_and_push_eval(
    environments: list[str],
    model: str,
    client: AsyncOpenAI,
    num_examples: int | list[int],
    rollouts_per_example: int | list[int],
    max_concurrent: int | list[int],
    env_args_dict: dict[str, dict] | None = None,
    sampling_args: dict | None = None,
    sampling_args_dict: dict[str, dict] | None = None,
    save_to_hub: bool = True,
    eval_name: str | None = None,
    framework: str = "verifiers",
    env_dir_path: str = "./environments",
) -> dict[str, Any]:
    """Run multi-env evaluation and save to Hub"""
    # Normalize inputs to lists
    if isinstance(num_examples, int):
        num_examples = [num_examples] * len(environments)
    if isinstance(rollouts_per_example, int):
        rollouts_per_example = [rollouts_per_example] * len(environments)
    if isinstance(max_concurrent, int):
        max_concurrent = [max_concurrent] * len(environments)

    env_args_dict = env_args_dict or {}

    console.print(f"[dim]Running evaluation on {len(environments)} environments[/dim]")
    results_dict = await eval_environments_parallel(
        envs=environments,
        env_args_dict=env_args_dict,
        client=client,
        model=model,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
        sampling_args=sampling_args,
        sampling_args_dict=sampling_args_dict,
    )

    # Aggregate metrics
    metrics_dict = {}
    for env_id, results in results_dict.items():
        idx = environments.index(env_id)
        rollouts = rollouts_per_example[idx]
        rewards = results.reward

        env_metrics = {
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "num_samples": len(rewards),
            "num_examples": num_examples[idx],
            "rollouts_per_example": rollouts,
        }

        for metric_name, metric_values in results.metrics.items():
            env_metrics[f"avg_{metric_name}"] = float(np.mean(metric_values))
            env_metrics[f"std_{metric_name}"] = float(np.std(metric_values))

        metrics_dict[env_id] = env_metrics

    # Save to Hub if requested
    eval_ids_dict = {}
    if save_to_hub:
        console.print(f"[dim]Pushing {len(environments)} evaluations to Hub[/dim]")
        api_client = APIClient()
        evals_client = EvalsClient(api_client)

        for env_id, results in results_dict.items():
            idx = environments.index(env_id)

            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            eval_name_for_env = eval_name or f"{model.replace('/', '-')}-{env_id}-{timestamp_str}"

            env_sampling = sampling_args
            if sampling_args_dict:
                env_sampling = sampling_args_dict.get(env_id, sampling_args)

            evaluation_id = push_verifiers_eval_to_hub(
                env_id=env_id,
                results=results,
                model=model,
                eval_name=eval_name_for_env,
                framework=framework,
                metrics=metrics_dict[env_id],
                num_examples=num_examples[idx],
                rollouts_per_example=rollouts_per_example[idx],
                max_concurrent=max_concurrent[idx],
                sampling_args=env_sampling,
                env_dir_path=env_dir_path,
                evals_client=evals_client,
                serialize_messages_fn=serialize_messages_for_hub,
                on_success=lambda eid, eval_id, url: console.print(
                    f"[green]✓ Pushed eval for {eid} to Hub[/green]" + (f": {url}" if url else "")
                ),
                on_error=lambda eid, exc: console.print(
                    f"[red]✗ Cannot push eval for {eid}: {exc}[/red]"
                    if isinstance(exc, InvalidEvaluationError)
                    else f"[red]Failed to push eval for {eid}: {exc}[/red]"
                ),
                on_warning=lambda msg: console.print(f"[yellow]Warning: {msg}[/yellow]"),
            )

            if evaluation_id:
                eval_ids_dict[env_id] = evaluation_id

    return {
        "results": results_dict,
        "metrics": metrics_dict,
        "eval_ids": eval_ids_dict if save_to_hub else {},
    }
