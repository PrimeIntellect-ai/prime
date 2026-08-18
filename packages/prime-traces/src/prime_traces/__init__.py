"""Prime Intellect Traces SDK.

Upload, query and export training/evaluation/inference traces through the
Prime Traces service. Uploads are content-addressed JSONL batches; reads are
cursor-paginated summaries over extracted columns plus raw-document retrieval.
"""

from .async_traces import AsyncTracesClient
from .batching import (
    DEFAULT_TARGET_BATCH_BYTES,
    MAX_BATCH_BYTES,
    MAX_BATCH_LINES,
    MAX_LINE_BYTES,
    Batch,
    aiter_batches,
    iter_batches,
    read_jsonl_lines,
)
from .core import AsyncTracesAPIClient, Config, TracesAPIClient
from .exceptions import (
    AmbiguousDeleteError,
    APIError,
    APITimeoutError,
    ForbiddenError,
    LineFormatConflictError,
    NotFoundError,
    PaymentRequiredError,
    PrimeTracesError,
    RetryableAPIError,
    TraceTooLargeError,
    TransportError,
    UnauthorizedError,
    ValidationRejectedError,
)
from .models import (
    EpisodeDetail,
    EpisodeError,
    EpisodeListPage,
    EpisodeSummary,
    EpisodeTraceAggregate,
    ErrorCode,
    Execution,
    LineFormat,
    ModelInfo,
    Score,
    TraceListPage,
    TraceSummary,
    UploadReceipt,
)
from .traces import SupportsToRecord, TraceRecord, TracesClient

__version__ = "0.0.2"

__all__ = [
    # Clients & config
    "TracesClient",
    "TracesAPIClient",
    "AsyncTracesClient",
    "AsyncTracesAPIClient",
    "Config",
    "SupportsToRecord",
    "TraceRecord",
    # Batching
    "Batch",
    "aiter_batches",
    "iter_batches",
    "read_jsonl_lines",
    "DEFAULT_TARGET_BATCH_BYTES",
    "MAX_BATCH_BYTES",
    "MAX_BATCH_LINES",
    "MAX_LINE_BYTES",
    # Models
    "EpisodeDetail",
    "EpisodeError",
    "EpisodeListPage",
    "EpisodeSummary",
    "EpisodeTraceAggregate",
    "ErrorCode",
    "Execution",
    "LineFormat",
    "ModelInfo",
    "Score",
    "TraceListPage",
    "TraceSummary",
    "UploadReceipt",
    # Exceptions
    "PrimeTracesError",
    "APIError",
    "APITimeoutError",
    "AmbiguousDeleteError",
    "ForbiddenError",
    "LineFormatConflictError",
    "NotFoundError",
    "PaymentRequiredError",
    "RetryableAPIError",
    "TraceTooLargeError",
    "TransportError",
    "UnauthorizedError",
    "ValidationRejectedError",
]
