import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class HostedEvalConfig(BaseModel):
    """Prime-owned input contract for one hosted V0 evaluation."""

    model_config = ConfigDict(extra="forbid")

    env_id: str = Field(min_length=1)
    model: str = Field(default="openai/gpt-4.1-mini", min_length=1)
    num_examples: int = Field(default=5, ge=-1)
    rollouts_per_example: int = Field(default=3, ge=1)
    env_args: Optional[dict[str, Any]] = None
    name: Optional[str] = None
    timeout_minutes: Optional[int] = Field(default=None, ge=1)
    allow_sandbox_access: bool = True
    allow_instances_access: bool = False
    allow_tunnel_access: bool = True
    custom_secrets: Optional[dict[str, str]] = None
    sampling_args: Optional[dict[str, Any]] = None
    max_concurrent: Optional[int] = Field(default=None, ge=1)
    max_retries: Optional[int] = Field(default=None, ge=0)
    state_columns: Optional[list[str]] = None
    independent_scoring: bool = False
    verbose: bool = False
    headers: Optional[list[str]] = None
    extra_env_kwargs: Optional[dict[str, Any]] = None
    api_client_type: Optional[str] = None
    api_base_url: Optional[str] = None
    api_key_var: Optional[str] = None

    @field_validator("headers")
    @classmethod
    def validate_headers(cls, headers: Optional[list[str]]) -> Optional[list[str]]:
        for header in headers or []:
            name, separator, value = header.partition(":")
            if not separator or not name.strip() or not value.strip():
                raise ValueError("headers must use 'Name: Value'")
        return headers


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
