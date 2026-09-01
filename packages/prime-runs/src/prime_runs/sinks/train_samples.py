"""Training samples sink: the step-keyed Parquet table the training viewer reads.

What prime-rl's ``TrainRun.upload_samples`` did, in the SDK: for each training
step, encode the step's episodes to one Parquet object and send it through the
platform's presign -> PUT to object storage -> confirm flow. Like the eval
samples sink this serves *today's* viewer; Prime Traces is the system of record
and this sink leaves the default list once the viewer reads traces natively.

Which episodes to upload is read off the records: a verifiers episode carries
``run.work`` (``TrainWorkInfo(step=...)``) once its producer has stamped it, and
prime-rl stamps every dispatched episode. Episodes from eval work, bare traces
and JSON episodes have no row here and go to Prime Traces only.

Objects are additive per step: every upload mints its own
``step_{step}_{uuid}.parquet`` key and the viewer unions every object under a
step, so a step's episodes may arrive across several ``log_episodes`` calls —
prime-rl hands over a step's batch at ship time, and an off-policy episode
dispatched at step N can land in a later batch. Each call becomes one more
object; ``sample_id`` is numbered from a per-upload offset so (step,
sample_id), which the viewer looks samples up by, stays unique across them.
Log a step in one call to keep it to one object.
"""

import logging
import secrets
import time
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import httpx
from prime_traces.core.client import retry_delay

from .. import _fork
from .._http import UPLOAD_TIMEOUT, PlatformClient
from ..exceptions import (
    APIError,
    RetryableAPIError,
    TransportError,
    is_record_rejection,
    is_transient,
)
from .base import Sink, SinkWriteError, is_episode

logger = logging.getLogger(__name__)

#: Upload every Nth training step (prime-rl's cadence, chosen so the sample
#: table does not receive every rollout of every step).
DEFAULT_STEP_INTERVAL = 10
#: Attempts for the object-storage PUT, which is idempotent.
UPLOAD_ATTEMPTS = 3

#: ``sample_id = range * SAMPLE_ID_STRIDE + row``. In the process that opened
#: the run the range is the step's upload index: the first object numbers its
#: rows from 0 (what a single-object step always did), a straggler object from
#: 2**32, and so on; no object holds 2**32 rows. A forked child inherits a copy
#: of those counters and would hand out the same ranges as its parent, so it
#: draws its ranges at random from the upper half of the range space instead,
#: which the parent's counter never reaches. Every id stays below 2**53: the
#: viewer parses ``sample_id`` as a JSON number.
SAMPLE_ID_STRIDE = 1 << 32
SAMPLE_ID_RANGES = 1 << 21


class Encoder(Protocol):
    """``(episodes, run_id, step, sample_id_offset=...) -> parquet bytes``, or
    ``None`` when there is nothing to upload for the step. Rows are numbered
    from ``sample_id_offset``."""

    def __call__(
        self, episodes: Sequence[Any], run_id: str, step: int, *, sample_id_offset: int = 0
    ) -> Optional[bytes]: ...


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def training_step(record: Any) -> Optional[int]:
    """The training step an episode was dispatched at, from its ``run.work``
    (verifiers ``TrainRunInfo`` / ``TrainWorkInfo``). ``None`` for anything
    else: eval work, a record without provenance, a bare trace."""
    run = _field(record, "run")
    work = _field(run, "work") if run is not None else None
    if _field(run, "type") != "train" or _field(work, "type") != "train":
        return None
    step = _field(work, "step")
    return step if isinstance(step, int) and not isinstance(step, bool) else None


class RftSamplesSink(Sink):
    """Encodes each training step's episodes to Parquet and uploads it."""

    name = "rft_samples"

    def __init__(
        self,
        client: PlatformClient,
        *,
        encoder: Optional[Encoder] = None,
        step_interval: int = DEFAULT_STEP_INTERVAL,
        upload_client: Optional[httpx.Client] = None,
    ) -> None:
        if step_interval < 1:
            raise ValueError("step_interval must be at least 1")
        self.enabled = True
        self._client = client
        self._encoder = encoder
        self._step_interval = step_interval
        self._upload_client = upload_client
        self._owns_upload_client = upload_client is None
        # An inherited socket belongs to the parent; see reset_after_fork.
        self._forked_with_injected_client = False
        self._run_id: Optional[str] = None
        self.steps_written = 0
        #: Records with no row in this table (eval work, bare traces, JSON).
        self.skipped = 0
        #: Training episodes left out by the step cadence. Not a loss.
        self.sampled_out = 0
        #: Uploads attempted per step; each numbers its rows in its own range.
        self._uploads_per_step: Dict[int, int] = {}
        #: A forked child cannot share the parent's counters; see SAMPLE_ID_RANGES.
        self._forked = False
        _fork.register(self)

    # ------------------------------------------------------------------ setup

    def reset_after_fork(self) -> None:
        """Drop (not close) the inherited storage client: its socket is the
        parent's. An owned client is rebuilt on the next upload; an injected
        one cannot be, so the child's uploads fail rather than share it. The
        inherited upload counters are the parent's too: from here on ranges
        are drawn at random, apart from anything the parent hands out."""
        if self._owns_upload_client:
            self._upload_client = None
        else:
            self._forked_with_injected_client = True
        self._forked = True

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        self._run_id = run_id
        if self._encoder is None:
            from ..projection import episodes_to_parquet_bytes, parquet_available

            if not parquet_available():
                # Not a failure: nothing was lost, the episodes still reach
                # Prime Traces. Say once how to turn the table on.
                logger.warning(
                    "Training samples sink off: pyarrow is not installed "
                    "(pip install 'prime-runs[train]'); episodes reach Prime Traces only"
                )
                self.enabled = False
                return
            self._encoder = episodes_to_parquet_bytes

    # ------------------------------------------------------------------ write

    def write(self, records: Sequence[Any]) -> None:
        if not self.enabled or not records:
            return
        if self._run_id is None:
            raise RuntimeError("RftSamplesSink.write called before start()")
        assert self._encoder is not None  # start() set it or disabled the sink

        steps = self._group(records)
        failed = 0
        reported: Optional[Exception] = None
        for position, (step, episodes) in enumerate(steps):
            try:
                uploaded = self._upload_step(step, episodes)
            except Exception as exc:  # noqa: BLE001 - classified below
                failed += len(episodes)
                if not (is_record_rejection(exc) or is_transient(exc)):
                    # Sink-wide: the remaining steps would fail the same way.
                    failed += sum(len(rest) for _, rest in steps[position + 1 :])
                    reported = exc
                    break
                if reported is None or is_transient(exc):
                    reported = exc
            else:
                if uploaded:
                    self.steps_written += 1

        if reported is not None:
            raise SinkWriteError(reported, failed_records=failed) from reported

    def _group(self, records: Sequence[Any]) -> List[Tuple[int, List[Any]]]:
        """Training episodes by step, on the upload cadence, in step order."""
        by_step: Dict[int, List[Any]] = {}
        skipped = 0
        for record in records:
            step = training_step(record)
            # The encoder projects attributes (``build_samples``), so a JSON
            # episode has no row here either.
            if step is None or isinstance(record, Mapping) or not is_episode(record):
                skipped += 1
                continue
            if step % self._step_interval:
                self.sampled_out += 1
                continue
            by_step.setdefault(step, []).append(record)
        if skipped:
            if not self.skipped:
                logger.info(
                    "The training sample table holds training-work episodes only; %d "
                    "record(s) in this batch have no row there and reach Prime Traces only.",
                    skipped,
                )
            self.skipped += skipped
        return sorted(by_step.items())

    def _upload_step(self, step: int, episodes: List[Any]) -> bool:
        """Encode and upload one step. ``False`` when the step had nothing to
        show (no trajectories) and no request was made."""
        assert self._encoder is not None and self._run_id is not None
        # Every upload of a step numbers its rows in its own range, counted by
        # attempt: an upload whose confirm was lost may still have landed.
        payload = self._encoder(
            episodes, self._run_id, step, sample_id_offset=self._next_range(step) * SAMPLE_ID_STRIDE
        )
        if payload is None:
            return False

        # Replayable: a presign only mints a URL.
        presign = self._client.post(
            "/rft/samples/presign",
            json_body={"run_id": self._run_id, "step": step},
            idempotent=True,
        )
        data = presign.get("data") or presign
        url = data.get("presignedUrl") or data.get("presigned_url")
        key = data.get("s3Key") or data.get("s3_key")
        if not url or not key:
            raise APIError(
                f"POST /rft/samples/presign returned no upload URL/key (keys: {sorted(data)})"
            )

        self._put_object(url, payload)

        # Records the object under the step. Replayable: confirm checks the
        # key and the object, then refreshes the run's progress; a second
        # confirm of the same key changes nothing.
        self._client.post(
            "/rft/samples/confirm",
            json_body={"run_id": self._run_id, "step": step, "s3_key": key},
            idempotent=True,
        )
        return True

    def _next_range(self, step: int) -> int:
        """The ``sample_id`` range for the next upload of ``step``."""
        if self._forked:
            half = SAMPLE_ID_RANGES // 2
            return half + secrets.randbelow(half)
        upload_index = self._uploads_per_step.get(step, 0)
        self._uploads_per_step[step] = upload_index + 1
        return upload_index

    def _put_object(self, url: str, payload: bytes) -> None:
        """PUT the object to storage. Bare client: the presigned URL carries its
        own credentials and rejects the platform's auth headers. Idempotent, so
        transport failures and 5xx are retried."""
        client = self._upload()
        for attempt in range(UPLOAD_ATTEMPTS):
            error: APIError
            try:
                response = client.put(
                    url,
                    content=payload,
                    headers={"Content-Type": "application/parquet"},
                    timeout=UPLOAD_TIMEOUT,
                )
            except httpx.RequestError as exc:
                error = TransportError(f"uploading samples failed: {type(exc).__name__}: {exc}")
            else:
                if response.is_success:
                    return
                message = f"uploading samples failed: HTTP {response.status_code}"
                if response.status_code >= 500 or response.status_code == 429:
                    error = RetryableAPIError(message, status_code=response.status_code)
                else:
                    raise APIError(message, status_code=response.status_code)
            if attempt == UPLOAD_ATTEMPTS - 1:
                raise error
            time.sleep(retry_delay(error, attempt))

    def _upload(self) -> httpx.Client:
        if self._forked_with_injected_client:
            raise RuntimeError("an injected upload client cannot be reused after a fork")
        if self._upload_client is None:
            self._upload_client = httpx.Client(follow_redirects=False)
        return self._upload_client

    def flush(self) -> None:
        """Uploads are synchronous; nothing is held back here."""

    def close(self) -> None:
        client = self._upload_client
        self._upload_client = None
        if client is not None and self._owns_upload_client:
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                logger.debug("Error closing the upload client: %s", exc)
