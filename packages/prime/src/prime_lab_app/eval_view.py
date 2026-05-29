"""Standalone evaluation viewer for `prime eval view`."""

from __future__ import annotations

from pathlib import Path

from .app import PrimeLabView
from .data import LabDataSource, LabLoadOptions

DEFAULT_EVAL_VIEW_LIMIT = 50


def build_eval_view_app(
    *,
    limit: int = DEFAULT_EVAL_VIEW_LIMIT,
    env_dir: str = "./environments",
    outputs_dir: str = "./outputs",
    workspace: Path | None = None,
    data_source: LabDataSource | None = None,
) -> PrimeLabView:
    if limit < 1:
        raise ValueError("limit must be at least 1")

    current_workspace = (workspace or Path.cwd()).resolve()
    source = data_source or LabDataSource()

    def options_for(current_limit: int) -> LabLoadOptions:
        return LabLoadOptions(
            limit=current_limit,
            workspace=current_workspace,
            env_dir=env_dir,
            outputs_dir=outputs_dir,
        )

    return PrimeLabView(
        lambda: source.load_evaluations(options_for(limit)),
        lambda item, include_logs, log_tail_lines, metrics_limit, metrics_min_step: (
            source.load_item_detail(
                item,
                include_logs=include_logs,
                log_tail_lines=log_tail_lines,
                metrics_limit=metrics_limit,
                metrics_min_step=metrics_min_step,
            )
        ),
        lambda: source.load_evaluations_initial(options_for(limit)),
        initial_section_key="evaluations",
        show_launch_screen=False,
        sync_agent_runtime=False,
    )


def run_eval_view(
    *,
    limit: int = DEFAULT_EVAL_VIEW_LIMIT,
    env_dir: str = "./environments",
    outputs_dir: str = "./outputs",
    workspace: Path | None = None,
) -> None:
    build_eval_view_app(
        limit=limit,
        env_dir=env_dir,
        outputs_dir=outputs_dir,
        workspace=workspace,
    ).run()
