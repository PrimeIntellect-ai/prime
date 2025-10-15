import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from prime_evals import APIClient, EvalsClient, InvalidEvaluationError
from rich.console import Console
from verifiers.utils.eval_runner import (
    eval_environments_parallel,
    serialize_messages_for_hub,
)

console = Console()


def _push_env_eval_to_hub(
    env_id: str,
    results: Any,
    model: str,
    eval_name: str,
    framework: str,
    metrics: dict[str, float],
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    sampling_args: dict | None,
    env_dir_path: str,
    evals_client: EvalsClient,
) -> str | None:
    """Push a single environment's evaluation to Hub"""
    # Look up environment Hub ID from metadata
    env_hub_id = None
    version_id = None
    env_dir = Path(env_dir_path) / env_id.replace("-", "_")
    hub_metadata_file = env_dir / ".env-metadata.json"

    if hub_metadata_file.exists():
        try:
            with open(hub_metadata_file) as f:
                hub_metadata = json.load(f)
                env_hub_id = hub_metadata.get("environment_id")
                version_id = hub_metadata.get("version_id")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load metadata for {env_id}: {e}[/yellow]")

    # Prepare environment reference
    environments_list = []
    if env_hub_id:
        env_dict = {"id": env_hub_id}
        if version_id:
            env_dict["version_id"] = version_id
        environments_list.append(env_dict)
    else:
        environments_list.append({"id": env_id})

    metadata = {
        "environment": env_id,
        "model": model,
        "num_examples": num_examples,
        "rollouts_per_example": rollouts_per_example,
        "max_concurrent": max_concurrent,
        "sampling_args": sampling_args,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Prepare samples
    sample_results = []
    n_samples = len(results.reward)
    example_ids = [i // rollouts_per_example for i in range(n_samples)]

    for i in range(n_samples):
        task_val = results.task[i] if i < len(results.task) else ""
        answer_val = results.answer[i] if i < len(results.answer) else ""

        # Serialize task and answer
        if hasattr(task_val, "model_dump"):
            task_str = json.dumps(task_val.model_dump())
        else:
            task_str = str(task_val) if task_val else ""

        if hasattr(answer_val, "model_dump"):
            answer_str = json.dumps(answer_val.model_dump())
        else:
            answer_str = str(answer_val) if answer_val else ""

        # Serialize prompt and completion
        prompt_data = None
        if i < len(results.prompt):
            prompt_data = serialize_messages_for_hub(results.prompt[i])
        completion_data = None
        if i < len(results.completion):
            completion_data = serialize_messages_for_hub(results.completion[i])

        result_entry = {
            "example_id": int(example_ids[i]),
            "rollout_number": int(i % rollouts_per_example),
            "reward": float(results.reward[i]),
            "task": task_str,
            "answer": answer_str,
            "prompt": prompt_data,
            "completion": completion_data,
        }

        # Add info fields
        if i < len(results.info):
            info = results.info[i]
            if isinstance(info, dict):
                if "score" in info:
                    result_entry["score"] = float(info["score"])
                if "correct" in info:
                    result_entry["correct"] = bool(info["correct"])

        # Add metrics
        for metric_name, metric_values in results.metrics.items():
            result_entry[metric_name] = float(metric_values[i])

        sample_results.append(result_entry)

    # Push to Hub
    try:
        create_response = evals_client.create_evaluation(
            name=eval_name,
            environments=environments_list,
            model_name=model,
            framework=framework,
            metadata=metadata,
            metrics=metrics,
        )

        evaluation_id = create_response["evaluation_id"]
        console.print(f"[dim]Created evaluation for {env_id}: {evaluation_id}[/dim]")

        evals_client.push_samples(evaluation_id, sample_results)

        finalize_response = evals_client.finalize_evaluation(evaluation_id, metrics=metrics)

        viewer_url = finalize_response.get("viewer_url")
        if viewer_url:
            console.print(f"[green]✓ Pushed eval for {env_id} to Hub[/green]: {viewer_url}")
        else:
            console.print(f"[green]✓ Pushed eval for {env_id} to Hub[/green]")

        return evaluation_id

    except InvalidEvaluationError as e:
        console.print(
            f"[red]✗ Cannot push eval for {env_id}: Environment not found on Hub.[/red]\n"
            f"  Please push the environment first:\n"
            f"  1. Using verifiers: env.push_to_env_hub(hub_name='<owner>/{env_id}')\n"
            f"  2. Using prime CLI: prime env push {env_id}\n"
            f"  3. Visit: https://app.primeintellect.ai/environments\n"
            f"  Error: {e}"
        )
        return None
    except Exception as e:
        console.print(f"[red]Failed to push eval for {env_id}: {e}[/red]")
        return None


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

            evaluation_id = _push_env_eval_to_hub(
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
            )

            if evaluation_id:
                eval_ids_dict[env_id] = evaluation_id

    return {
        "results": results_dict,
        "metrics": metrics_dict,
        "eval_ids": eval_ids_dict if save_to_hub else {},
    }
