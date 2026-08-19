"""Legacy sink: the flat eval-sample table behind today's viewer.

This exists so the migration is a refactor rather than a regression. The viewer
reads the v0 sample table; Prime Traces is in closed beta on an account
allowlist. Shipping traces-only would leave every non-allowlisted account
staring at an empty dashboard — so both sinks run, and this one retires when
the Viewer API reads traces natively. Retiring it is a one-line change to the
default sink list, with nothing to do in verifiers or prime-rl.

Its known weakness is why traces is the primary: ``POST /samples`` *appends*,
so a request whose response was lost cannot be safely replayed. The client
therefore does not retry it through an ambiguous failure — losing a batch is
recoverable, duplicated rows silently skew every average on the dashboard.
Content-addressed uploads have neither problem, which is exactly the property
the traces sink was built on.
"""

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .._http import UPLOAD_TIMEOUT, PlatformClient, encode_json
from ..projection import batch_samples, build_samples, trace_to_sample
from .base import Sink

logger = logging.getLogger(__name__)


class EvalSamplesSink(Sink):
    """Projects episodes to v0 samples and pushes them to the evaluations API."""

    name = "eval_samples"

    def __init__(self, client: PlatformClient) -> None:
        self.enabled = True
        self._client = client
        self._run_id: Optional[str] = None
        # Carried across calls so a streaming producer numbers rollouts the same
        # way a one-shot upload does: the Nth episode for an example is rollout N,
        # whether it arrived alone or in a batch of five hundred.
        self._rollout_numbers: Dict[Any, int] = {}
        self.samples_written = 0

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        self._run_id = run_id

    def write(
        self,
        records: Sequence[Any],
        *,
        line_format: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        if not self.enabled or not records:
            return
        if self._run_id is None:
            raise RuntimeError("EvalSamplesSink.write called before start()")

        samples = self._to_samples(records)
        if not samples:
            return
        for batch in batch_samples(samples):
            self._client.post(
                f"/evaluations/{self._run_id}/samples",
                content=encode_json({"samples": batch}),
                timeout=UPLOAD_TIMEOUT,
                # Appends. Left non-replayable (the POST default) so a lost
                # response cannot turn into duplicate rows.
                idempotent=False,
            )
            self.samples_written += len(batch)

    def _to_samples(self, records: Sequence[Any]) -> List[Dict[str, Any]]:
        """Project native episodes/traces and pass through existing samples.

        A producer that already speaks the v0 sample format (a dict with
        ``sample_id``) sends it unchanged; anything with ``traces`` is a native
        episode, and anything with ``branches`` is a native trace. Anything else
        is skipped loudly rather than posted as a malformed row the API would
        reject for the whole batch.
        """
        samples: List[Dict[str, Any]] = []
        for record in records:
            if isinstance(record, Mapping):
                if "sample_id" in record:
                    samples.append(dict(record))
                elif "traces" in record:
                    logger.debug(
                        "Skipping a pre-serialized episode: this sink projects native "
                        "episode objects, not their JSON records"
                    )
                else:
                    logger.debug("Skipping a record with no sample_id and no traces")
                continue
            if hasattr(record, "traces"):
                samples.extend(build_samples([record], self._rollout_numbers))
            elif hasattr(record, "branches"):
                idx = record.task.data.idx
                self._rollout_numbers[idx] = number = self._rollout_numbers.get(idx, 0) + 1
                samples.append(trace_to_sample(record, rollout_number=number))
            else:
                logger.debug("Skipping %s: not an episode or trace", type(record).__name__)
        return samples

    def flush(self) -> None:
        """Writes are synchronous; the uploader thread owns the asynchrony."""

    def close(self) -> None:
        """The platform client is shared with the backend, which closes it."""
