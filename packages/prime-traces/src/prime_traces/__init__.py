"""Prime Intellect Traces SDK.

Upload, query and export training/evaluation/inference traces through the
Prime Traces service. Uploads are content-addressed JSONL batches; reads are
cursor-paginated summaries over extracted columns plus raw-document retrieval.
"""

from .batching import (
    DEFAULT_TARGET_BATCH_BYTES,
    MAX_BATCH_BYTES,
    MAX_BATCH_LINES,
    MAX_LINE_BYTES,
    Batch,
    iter_batches,
    read_jsonl_lines,
)
from .core import Config, TracesAPIClient
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
    ErrorCode,
    Execution,
    LineFormat,
    ModelInfo,
    Score,
    TraceListPage,
    TraceSummary,
    UploadReceipt,
)
from .traces import TracesClient

__version__ = "0.0.1"

__all__ = [
    # Clients & config
    "TracesClient",
    "TracesAPIClient",
    "Config",
    # Batching
    "Batch",
    "iter_batches",
    "read_jsonl_lines",
    "DEFAULT_TARGET_BATCH_BYTES",
    "MAX_BATCH_BYTES",
    "MAX_BATCH_LINES",
    "MAX_LINE_BYTES",
    # Models
    "UploadReceipt",
    "ErrorCode",
    "Execution",
    "LineFormat",
    "ModelInfo",
    "Score",
    "TraceListPage",
    "TraceSummary",
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
