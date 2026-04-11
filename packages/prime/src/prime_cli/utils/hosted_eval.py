import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .formatters import strip_ansi

PROGRESS_BAR_MIN_WIDTH = 10
PROGRESS_BAR = re.compile(rf".*\|[█▏▎▍▌▋▊▉ ]{{{PROGRESS_BAR_MIN_WIDTH},}}\|.*")


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
    sampling_args: Optional[dict[str, Any]] = None
    extra_env_kwargs: Optional[dict[str, Any]] = None
    api_base_url: Optional[str] = None
    api_key_var: Optional[str] = None


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
