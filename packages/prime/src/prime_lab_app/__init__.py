"""Interactive Lab terminal viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import PrimeLabView, run_lab_view
    from .eval_screen import LocalEvalRunScreen

__all__ = ["LocalEvalRunScreen", "PrimeLabView", "run_lab_view"]


def __getattr__(name: str) -> Any:
    if name in {"PrimeLabView", "run_lab_view"}:
        from .app import PrimeLabView, run_lab_view

        return {"PrimeLabView": PrimeLabView, "run_lab_view": run_lab_view}[name]
    if name == "LocalEvalRunScreen":
        from .eval_screen import LocalEvalRunScreen

        return LocalEvalRunScreen
    raise AttributeError(name)
