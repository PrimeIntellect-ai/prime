"""Shared title normalization for Lab agent widgets."""

from __future__ import annotations


def clean_widget_title(value: str) -> str:
    title = value.strip() or "Action"
    lowered = title.lower()
    for prefix in ("eval:", "evaluation:", "train:", "training:", "run:"):
        if lowered.startswith(prefix):
            return title[len(prefix) :].strip() or title
    return title


def config_picker_summary(config_kind: str) -> str:
    if config_kind == "eval":
        return "Environment, model, examples, rollouts, tokens, concurrency"
    if config_kind == "rl":
        return "Environment, model, steps, rollouts, batch, tokens"
    if config_kind == "gepa":
        return "Environment, model"
    return "Environment, model"
