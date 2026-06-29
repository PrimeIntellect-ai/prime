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
    """Convert v1 traces to the sample schema while preserving legacy results.

    Delegates to Verifiers' ``convert_results_for_upload`` so the artifact
    format stays owned by Verifiers; Prime only consumes the normalized output.
    """
    from verifiers.v1.cli.output import convert_results_for_upload

    return convert_results_for_upload(samples)
