import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from prime_evals import APIClient
from prime_evals.evals import EvalsClient
from prime_evals.exceptions import InvalidEvaluationError


def push_verifiers_eval_to_hub(
    env_id: str,
    results: Any,
    model: str,
    eval_name: str,
    framework: str,
    metrics: dict[str, float],
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    sampling_args: dict | None = None,
    custom_metadata: dict | None = None,
    env_dir_path: str = "./environments",
    evals_client: EvalsClient | None = None,
    serialize_messages_fn: Callable[[Any], Any] | None = None,
    on_success: Callable[[str, str, str | None], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> str | None:
    """Push verifiers evaluation results to Environment Hub"""
    if evals_client is None:
        api_client = APIClient()
        evals_client = EvalsClient(api_client)

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
            if on_warning:
                on_warning(f"Could not load metadata for {env_id}: {e}")

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

    # Merge custom metadata if provided
    if custom_metadata:
        metadata.update(custom_metadata)

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
            if serialize_messages_fn:
                prompt_data = serialize_messages_fn(results.prompt[i])
            else:
                prompt_data = results.prompt[i]

        completion_data = None
        if i < len(results.completion):
            if serialize_messages_fn:
                completion_data = serialize_messages_fn(results.completion[i])
            else:
                completion_data = results.completion[i]

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

        # Push samples
        if sample_results:
            evals_client.push_samples(evaluation_id, sample_results)

        # Finalize
        finalize_response = evals_client.finalize_evaluation(evaluation_id, metrics=metrics)

        viewer_url = finalize_response.get("viewer_url")
        if on_success:
            on_success(env_id, evaluation_id, viewer_url)

        return evaluation_id

    except InvalidEvaluationError as e:
        if on_error:
            on_error(env_id, e)
        return None
    except Exception as e:
        if on_error:
            on_error(env_id, e)
        return None
