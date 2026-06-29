"""Lazy access to Verifiers-owned eval artifact helpers."""

from pathlib import Path
from typing import Any


def _output_module() -> Any:
    from verifiers.v1.cli import output

    return output


def convert_eval_results(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _output_module().convert_results_for_upload(samples)


def discover_eval_outputs(outputs_dir: Path = Path("outputs")) -> list[Path]:
    return _output_module().discover_eval_artifact_dirs(outputs_dir)


def has_eval_files(directory: Path) -> bool:
    return _output_module().has_eval_artifacts(directory)


def load_eval_config(run_dir: Path) -> dict[str, Any]:
    return _output_module().read_config(run_dir)


def load_eval_upload(path: Path) -> Any:
    return _output_module().read_upload_data(path)


def resolve_eval_path(path: str | Path) -> Path:
    return _output_module().resolve_eval_artifact_dir(path)


def skipped_results_warning(upload: Any) -> str | None:
    skipped = upload.invalid_results
    if not skipped:
        return None
    preview = [f"line {error.line}: {error.reason}" for error in skipped[:5]]
    suffix = ", ..." if len(skipped) > 5 else ""
    return (
        f"Warning: Skipped {len(skipped)} invalid lines in results.jsonl "
        f"({', '.join(preview)}{suffix})"
    )
