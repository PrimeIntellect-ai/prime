"""Pydantic models for the Prime Traces API.

Response models allow extra fields (``extra="allow"``): the summary response is
documented to grow additively as more columns are extracted server-side, and an
older SDK must not break when that happens.

The list-page envelope (``traces`` / ``next_cursor`` keys) is provisional until
the service lands; it is asserted in one place so realignment is one edit.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class LineFormat(str, Enum):
    """Declared shape of each JSONL line in an upload request."""

    TRACE = "trace"  # default; the header is omitted on the wire
    EPISODE = "episode"  # X-Prime-Line-Format: episode


class ErrorCode(str, Enum):
    """Error codes returned by the service.

    Kept in lockstep with the service's ``ErrorCode``
    (``prime-traces/src/errors.py`` in the platform repo).
    Producers branch on the rejection codes — correct the file and
    resubmit, retry unchanged, or stop; 429/503 codes are retryable.
    """

    # Batch rejections (400): nothing stored. Validation is deterministic, so
    # resubmitting the same bytes yields the same verdict; corrected content
    # hashes to a new batch ID.
    BATCH_TOO_LARGE = "batch_too_large"
    TRACE_TOO_LARGE = "trace_too_large"
    TOO_MANY_TRACES_IN_EPISODE = "too_many_traces_in_episode"
    DUPLICATE_TRACE_ID = "duplicate_trace_id"
    DUPLICATE_EPISODE_ID = "duplicate_episode_id"
    DIGEST_MISMATCH = "digest_mismatch"
    LINE_FORMAT_MISMATCH = "line_format_mismatch"
    MALFORMED_ENCODING = "malformed_encoding"
    INVALID_TRACE = "invalid_trace"
    # The `metadata` part itself: absent, duplicated, oversized, or the wrong
    # shape. Separate from `invalid_trace` because the producer fixes its
    # uploader, not its trace file.
    INVALID_METADATA = "invalid_metadata"
    CREATED_AT_OUT_OF_WINDOW = "created_at_out_of_window"
    UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"
    UNKNOWN_EPISODE_REFERENCE = "unknown_episode_reference"
    # 409 — the one header not covered by the content digest
    LINE_FORMAT_CONFLICT = "line_format_conflict"

    # Malformed requests (400)
    INVALID_IDEMPOTENCY_KEY = "invalid_idempotency_key"
    INVALID_CURSOR = "invalid_cursor"
    INVALID_FILTER = "invalid_filter"

    # Auth (401/403/503)
    UNAUTHENTICATED = "unauthenticated"
    FORBIDDEN = "forbidden"
    AUTH_UNAVAILABLE = "auth_unavailable"

    # Not found (404)
    TRACE_NOT_FOUND = "trace_not_found"
    EPISODE_NOT_FOUND = "episode_not_found"
    EXPORT_JOB_NOT_FOUND = "export_job_not_found"

    # Retryable (429/503) and service state
    RATE_LIMITED = "rate_limited"
    WRITER_POOL_SATURATED = "writer_pool_saturated"
    INGEST_CAPACITY_EXCEEDED = "ingest_capacity_exceeded"
    INGEST_UNAVAILABLE = "ingest_unavailable"
    STORAGE_UNAVAILABLE = "storage_unavailable"
    NOT_IMPLEMENTED = "not_implemented"


class BatchReceipt(BaseModel):
    """Acknowledgment for one committed upload request.

    ``status == "committed"`` means every line in the request is durably
    stored — there is no partial success to interpret.
    """

    model_config = ConfigDict(extra="allow")

    # The batch ID *is* the content digest (64 lowercase hex, no prefix), so
    # the service does not restate it in a separate field.
    batch_id: str
    status: str


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: Optional[str] = None
    id: Optional[str] = None


class Score(BaseModel):
    model_config = ConfigDict(extra="allow")

    # None means unscored, which is distinct from a scored 0.0.
    reward: Optional[float] = None
    outcome: Optional[str] = None


class Execution(BaseModel):
    model_config = ConfigDict(extra="allow")

    has_error: Optional[bool] = None
    is_truncated: Optional[bool] = None


class TraceSummary(BaseModel):
    """The extracted column set for one trace — deliberately nothing more.

    Token usage, node/call counts, and per-phase timing are absent because the
    v0 extractor does not write them; fetch the raw document for those.
    """

    model_config = ConfigDict(extra="allow")

    trace_id: str
    batch_id: Optional[str] = None
    episode_id: Optional[str] = None
    created_at: Optional[datetime] = None
    ingested_at: Optional[datetime] = None
    run_id: Optional[str] = None
    environment_id: Optional[str] = None
    model: Optional[ModelInfo] = None
    task_id: Optional[str] = None
    score: Optional[Score] = None
    execution: Optional[Execution] = None
    duration_ms: Optional[int] = None
    size_bytes: Optional[int] = None
    context: Dict[str, str] = Field(default_factory=dict)


class EpisodeSummary(BaseModel):
    """One episode's summary plus episode-owned fields.

    Aggregates (total tokens, participating agents, any-trace-error) are
    computed from member traces at read time by the service.
    """

    model_config = ConfigDict(extra="allow")

    episode_id: str
    batch_id: Optional[str] = None
    created_at: Optional[datetime] = None
    ingested_at: Optional[datetime] = None
    environment_id: Optional[str] = None
    run_id: Optional[str] = None
    outcome: Optional[str] = None
    has_error: Optional[bool] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    total_tokens: Optional[int] = None
    total_agent_time: Optional[int] = None
    any_trace_error: Optional[bool] = None
    participating_agents: Optional[List[str]] = None


class TraceListPage(BaseModel):
    """One page of trace summaries. ``next_cursor`` is opaque and only valid
    with the exact filters that produced it."""

    model_config = ConfigDict(extra="allow")

    traces: List[TraceSummary] = Field(default_factory=list)
    next_cursor: Optional[str] = None


class EpisodeListPage(BaseModel):
    model_config = ConfigDict(extra="allow")

    episodes: List[EpisodeSummary] = Field(default_factory=list)
    next_cursor: Optional[str] = None
