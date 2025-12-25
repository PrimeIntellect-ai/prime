from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvaluationStatus(str, Enum):
    """Evaluation status enum"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Evaluation(BaseModel):
    """Evaluation model"""

    id: str = Field(..., alias="evaluation_id")
    name: str
    model_name: Optional[str] = Field(None, alias="modelName")
    dataset: Optional[str] = None
    framework: Optional[str] = None
    task_type: Optional[str] = Field(None, alias="taskType")
    eval_type: Optional[str] = Field(None, alias="evalType")
    description: Optional[str] = None
    status: Optional[str] = None
    environment_ids: Optional[List[str]] = Field(None, alias="environmentIds")
    suite_id: Optional[str] = Field(None, alias="suiteId")
    run_id: Optional[str] = Field(None, alias="runId")
    version_id: Optional[str] = Field(None, alias="versionId")
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    total_samples: Optional[int] = Field(None, alias="totalSamples")
    created_at: Optional[datetime] = Field(None, alias="createdAt")
    updated_at: Optional[datetime] = Field(None, alias="updatedAt")
    finalized_at: Optional[datetime] = Field(None, alias="finalizedAt")
    user_id: Optional[str] = Field(None, alias="userId")
    team_id: Optional[str] = Field(None, alias="teamId")

    model_config = ConfigDict(populate_by_name=True)


class EnvironmentReference(BaseModel):
    """Environment reference with optional version"""

    id: str
    version_id: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class CreateEvaluationRequest(BaseModel):
    """Create evaluation request model"""

    name: str
    environments: Optional[List[Dict[str, str]]] = None
    suite_id: Optional[str] = None
    run_id: Optional[str] = None
    model_name: Optional[str] = None
    dataset: Optional[str] = None
    framework: Optional[str] = None
    task_type: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


class Sample(BaseModel):
    """Evaluation sample model"""

    example_id: Optional[int] = Field(None, alias="exampleId")
    task: Optional[str] = None
    prompt: Optional[List[Dict[str, str]]] = None
    completion: Optional[List[Dict[str, str]]] = None
    answer: Optional[str] = None
    reward: Optional[float] = None
    score: Optional[float] = None
    correct: Optional[bool] = None
    format_reward: Optional[float] = Field(None, alias="formatReward")
    correctness: Optional[float] = None
    info: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class SamplesResponse(BaseModel):
    """Samples response model"""

    samples: List[Sample]
    total: int
    page: int
    limit: int
    has_more: bool = Field(..., alias="hasMore")

    model_config = ConfigDict(populate_by_name=True)


class EvaluationListResponse(BaseModel):
    """Evaluation list response model"""

    evaluations: List[Evaluation]
    total: int
    skip: int
    limit: int

    model_config = ConfigDict(populate_by_name=True)


class PushSamplesRequest(BaseModel):
    """Push samples request model"""

    samples: List[Dict[str, Any]]


class FinalizeEvaluationRequest(BaseModel):
    """Finalize evaluation request model"""

    metrics: Optional[Dict[str, Any]] = None


class Environment(BaseModel):
    """Environment model"""

    id: str
    name: str
    owner: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")
