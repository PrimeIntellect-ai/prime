from pathlib import Path

from .plain import get_console

console = get_console()


def load_eval_config(run_dir: Path) -> dict:
    """Load a native V1 run's resolved config."""
    from verifiers.v1.cli.output import read_config

    return read_config(run_dir)


def load_results_jsonl(path: Path) -> list[dict]:
    """
    Load and parse a results.jsonl file, skipping invalid lines with warnings.

    Args:
        path: Path to the results.jsonl file

    Returns:
        List of valid dict samples from the file
    """
    from verifiers.v1.cli.output import read_results

    results, skipped = read_results(path)

    if skipped:
        preview = [f"line {error.line}: {error.reason}" for error in skipped[:5]]
        suffix = ", ..." if len(skipped) > 5 else ""
        console.print(
            f"[yellow]Warning: Skipped {len(skipped)} invalid lines in results.jsonl "
            f"({', '.join(preview)}{suffix})[/yellow]"
        )

    return results


def convert_eval_results(samples: list[dict]) -> list[dict]:
    """Convert v1 traces to the sample schema while preserving legacy results."""
    trace_type = None
    trace_fields = {}
    node_fields = {}
    rollout_counts: dict[int, int] = {}
    converted = []

    for sample in samples:
        if not (
            isinstance(sample.get("nodes"), list)
            and isinstance(sample.get("task"), dict)
            and isinstance(sample.get("rewards"), dict)
        ):
            legacy_sample = dict(sample)
            if "id" in legacy_sample and "example_id" not in legacy_sample:
                legacy_sample["example_id"] = legacy_sample["id"]
            converted.append(legacy_sample)
            continue

        if trace_type is None:
            from verifiers.v1 import WireTrace
            from verifiers.v1.graph import MessageNode

            trace_type = WireTrace
            trace_fields = trace_type.model_fields
            node_fields = MessageNode.model_fields
        trace_data = {key: value for key, value in sample.items() if key in trace_fields}
        trace_data["nodes"] = [
            {key: value for key, value in node.items() if key in node_fields}
            for node in sample["nodes"]
        ]
        trace = trace_type.model_validate(trace_data)
        task = trace.task.model_dump(mode="json", exclude_none=True)
        branches = trace.branches
        main_messages = (
            [
                message.model_dump(mode="json", exclude_none=True)
                for message in branches[-1].messages
            ]
            if branches
            else []
        )
        trajectory = [
            {
                "messages": [
                    message.model_dump(mode="json", exclude_none=True)
                    for message in branch.messages
                ],
                "reward": trace.reward,
                "num_input_tokens": branch.prompt_len or branch.num_prompt_tokens,
                "num_output_tokens": branch.completion_len or branch.num_completion_tokens,
            }
            for branch in branches
        ]
        example_id = trace.task.idx
        rollout_counts[example_id] = rollout_counts.get(example_id, 0) + 1
        info = dict(trace.info)
        info.update({key: value for key, value in sample.items() if key not in trace_fields})

        converted.append(
            {
                "sample_id": trace.id,
                "example_id": example_id,
                "rollout_number": rollout_counts[example_id],
                "task": task,
                "prompt": [],
                "completion": main_messages,
                "answer": task.get("answer"),
                "reward": trace.reward,
                "timing": trace.timing.model_dump(mode="json", exclude_none=True),
                "is_completed": trace.is_completed,
                "is_truncated": trace.is_truncated,
                "metrics": trace.metrics,
                "error": (
                    trace.error.model_dump(mode="json", exclude_none=True) if trace.error else None
                ),
                "stop_condition": trace.stop_condition,
                "trajectory": trajectory,
                "token_usage": (
                    trace.usage.model_dump(mode="json", exclude_none=True) if trace.usage else None
                ),
                "num_steps": trace.num_turns,
                "info": info or None,
            }
        )

    return converted
