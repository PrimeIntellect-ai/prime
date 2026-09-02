"""Legacy sink: the flat eval-sample table today's viewer reads.

``POST /samples`` appends, so a request whose response was lost is not replayed:
a lost batch is recoverable, duplicated rows skew every average.
"""

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .._http import UPLOAD_TIMEOUT, PlatformClient, encode_json
from ..projection import batch_samples, build_samples
from .base import Sink, Unit, is_episode, send_each

logger = logging.getLogger(__name__)


class EvalSamplesSink(Sink):
    """Projects episodes to v0 samples and posts them to the evaluations API."""

    name = "eval_samples"

    def __init__(self, client: PlatformClient) -> None:
        self.enabled = True
        self._client = client
        self._run_id: Optional[str] = None
        # Per-example rollout counter, carried across calls so streamed batches
        # number rollouts the way a one-shot upload did.
        self._rollout_numbers: Dict[Any, int] = {}
        self.samples_written = 0
        #: Records with no v0 projection (JSON episodes, bare traces). They
        #: reach Prime Traces only and are not a loss.
        self.skipped = 0

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        self._run_id = run_id

    def write(self, records: Sequence[Any]) -> None:
        if not self.enabled or not records:
            return
        if self._run_id is None:
            raise RuntimeError("EvalSamplesSink.write called before start()")
        samples, owners = self._to_samples(records)
        units: List[Unit] = []
        offset = 0
        for batch in batch_samples(samples):
            batch_owners = set(owners[offset : offset + len(batch)])
            offset += len(batch)
            units.append((len(batch_owners), self._sender(batch)))
        send_each(units)

    def _sender(self, batch: List[Dict[str, Any]]) -> Callable[[], None]:
        def send() -> None:
            self._client.post(
                f"/evaluations/{self._run_id}/samples",
                content=encode_json({"samples": batch}),
                timeout=UPLOAD_TIMEOUT,
                idempotent=False,
            )
            self.samples_written += len(batch)

        return send

    def _to_samples(self, records: Sequence[Any]) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Project episode objects; pass v0 sample dicts through; skip the rest,
        warned once. ``owners`` maps each sample back to its input record so
        loss accounting survives HTTP batching."""
        samples: List[Dict[str, Any]] = []
        owners: List[int] = []
        skipped = 0
        first_skipped_type: Optional[str] = None
        for record_index, record in enumerate(records):
            if isinstance(record, Mapping) and "sample_id" in record:
                samples.append(dict(record))
                owners.append(record_index)
            elif not isinstance(record, Mapping) and is_episode(record):
                projected = build_samples([record], self._rollout_numbers)
                samples.extend(projected)
                owners.extend([record_index] * len(projected))
            else:
                skipped += 1
                if first_skipped_type is None:
                    first_skipped_type = type(record).__name__
        if skipped:
            if not self.skipped:
                logger.warning(
                    "The v0 sample table is projected from episode objects; %d record(s) "
                    "in this batch (%s) have no projection and reach Prime Traces only. "
                    "Further skips are counted, not logged.",
                    skipped,
                    first_skipped_type,
                )
            self.skipped += skipped
        return samples, owners

    def flush(self) -> None:
        """Writes are synchronous."""

    def close(self) -> None:
        """The client belongs to the backend."""
