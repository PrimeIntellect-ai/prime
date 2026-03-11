import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from rich.console import Console

from prime_cli.client import APIError, AsyncAPIClient

from .formatters import strip_ansi

console = Console()

PROGRESS_BAR = re.compile(r".*\|[█▏▎▍▌▋▊▉ ]{10,}\|.*")


class EvalStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"

    @classmethod
    def terminal_statuses(cls) -> set["EvalStatus"]:
        return {cls.COMPLETED, cls.FAILED, cls.TIMEOUT, cls.CANCELLED}

    @classmethod
    def has_logs_statuses(cls) -> set["EvalStatus"]:
        return {cls.RUNNING, cls.COMPLETED, cls.FAILED, cls.TIMEOUT, cls.CANCELLED}

    @property
    def color(self) -> str:
        return {
            EvalStatus.PENDING: "yellow",
            EvalStatus.RUNNING: "cyan",
            EvalStatus.COMPLETED: "green",
            EvalStatus.FAILED: "red",
            EvalStatus.TIMEOUT: "red",
            EvalStatus.CANCELLED: "yellow",
        }.get(self, "white")


@dataclass
class HostedEvalConfig:
    environment_id: str
    inference_model: str
    num_examples: int
    rollouts_per_example: int
    env_args: Optional[dict[str, str]] = None
    name: Optional[str] = None
    timeout_minutes: Optional[int] = None
    allow_sandbox_access: bool = False
    allow_instances_access: bool = False
    custom_secrets: Optional[dict[str, str]] = None


@dataclass
class HostedEvalResult:
    evaluation_id: str
    status: EvalStatus
    total_samples: int
    avg_score: Optional[float]
    min_score: Optional[float]
    max_score: Optional[float]
    error_message: Optional[str] = None
    logs: Optional[str] = None


def filter_progress_bars(text: str) -> str:
    lines = text.splitlines()
    filtered: list[str] = []
    for line in lines:
        if PROGRESS_BAR.search(line) or re.search(r"\d+%\|", line):
            if "100%" in line:
                match = re.search(r"([^|]*100%\|[█▏▎▍▌▋▊▉ ]+\|[^\n]*?)(?=\d+%\||$)", line)
                filtered.append((match.group(1) if match else line).strip())
            continue
        if line.strip():
            filtered.append(line)
    return "\n".join(filtered)


STATUS_MESSAGES = {
    "Waiting for container to start...",
    "No logs available",
    "Unable to retrieve logs",
    "Failed to fetch logs from sandbox",
    "The hosted eval is still initializing",
}


def is_status_message(text: str) -> bool:
    stripped = text.strip()
    return any(stripped.startswith(msg) for msg in STATUS_MESSAGES)


def clean_logs(text: str) -> str:
    cleaned = filter_progress_bars(strip_ansi(text))
    if is_status_message(cleaned):
        return ""
    return cleaned


def get_new_log_lines(old_logs: str, new_logs: str) -> list[str]:
    old_lines = old_logs.splitlines() if old_logs else []
    new_lines = new_logs.splitlines()

    if not old_logs:
        return new_lines

    overlap = 0
    max_overlap = min(len(old_lines), len(new_lines))
    for i in range(1, max_overlap + 1):
        if old_lines[-i:] == new_lines[:i]:
            overlap = i
    return new_lines[overlap:]


async def create_hosted_evaluation(
    client: AsyncAPIClient,
    config: HostedEvalConfig,
) -> dict[str, Any]:
    eval_config: dict[str, Any] = {
        "num_examples": config.num_examples,
        "rollouts_per_example": config.rollouts_per_example,
        "allow_sandbox_access": config.allow_sandbox_access,
        "allow_instances_access": config.allow_instances_access,
    }

    if config.env_args:
        eval_config["env_args"] = config.env_args
    if config.timeout_minutes is not None:
        eval_config["timeout_minutes"] = config.timeout_minutes
    if config.custom_secrets:
        eval_config["custom_secrets"] = config.custom_secrets

    payload: dict[str, Any] = {
        "environment_ids": [config.environment_id],
        "inference_model": config.inference_model,
        "eval_config": eval_config,
    }
    if config.name:
        payload["name"] = config.name

    return await client.post("/hosted-evaluations", json=payload)


async def get_evaluation(client: AsyncAPIClient, evaluation_id: str) -> dict[str, Any]:
    return await client.get(f"/evaluations/{evaluation_id}")


async def get_evaluation_logs(client: AsyncAPIClient, evaluation_id: str) -> str:
    response = await client.get(f"/hosted-evaluations/{evaluation_id}/logs")
    return response.get("logs") or ""


async def stop_hosted_evaluation(evaluation_id: str) -> dict[str, Any]:
    async with AsyncAPIClient() as client:
        return await client.post(f"/hosted-evaluations/{evaluation_id}/cancel")


async def run_hosted_evaluation(
    config: HostedEvalConfig,
    poll_interval: float = 10.0,
    follow: bool = True,
) -> HostedEvalResult:
    async with AsyncAPIClient() as client:
        created = await create_hosted_evaluation(client, config)
        evaluation_id = created.get("evaluation_id")
        if not evaluation_id:
            raise APIError(f"Failed to get evaluation ID from response: {created}")

        if not follow:
            return HostedEvalResult(
                evaluation_id=evaluation_id,
                status=EvalStatus.PENDING,
                total_samples=0,
                avg_score=None,
                min_score=None,
                max_score=None,
            )

        while True:
            eval_data = await get_evaluation(client, evaluation_id)
            status = EvalStatus(str(eval_data.get("status", EvalStatus.FAILED.value)))
            if status in EvalStatus.terminal_statuses():
                logs = await get_evaluation_logs(client, evaluation_id)
                return HostedEvalResult(
                    evaluation_id=evaluation_id,
                    status=status,
                    total_samples=eval_data.get("total_samples", 0),
                    avg_score=eval_data.get("avg_score"),
                    min_score=eval_data.get("min_score"),
                    max_score=eval_data.get("max_score"),
                    error_message=eval_data.get("error_message"),
                    logs=logs,
                )
            await asyncio.sleep(poll_interval)
