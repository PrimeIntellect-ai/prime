from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
        return {cls.RUNNING, cls.COMPLETED, cls.FAILED}

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
    viewer_url: Optional[str]
    total_samples: int
    avg_score: Optional[float]
    min_score: Optional[float]
    max_score: Optional[float]
    error_message: Optional[str] = None
    logs: Optional[str] = None


@dataclass
class HostedEvalCreationResult:
    evaluation_id: str
    viewer_url: Optional[str] = None
