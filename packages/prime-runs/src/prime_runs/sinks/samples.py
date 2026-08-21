"""Legacy sink: the flat eval-sample table behind today's viewer.

Prime Traces is gated to an allowlist and the viewer reads the v0 sample table,
so both sinks run until the viewer reads traces natively; retiring this one is
a change to the default sink list.

``POST /samples`` appends, so a request whose response was lost is not
replayed: a lost batch is recoverable, duplicated rows skew every average.
"""

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .._http import UPLOAD_TIMEOUT, PlatformClient, encode_json
from ..projection import batch_samples, build_samples
from .base import Sink, is_episode

logger = logging.getLogger(__name__)


class EvalSamplesSink(Sink):
    """Projects episodes to v0 samples and pushes them to the evaluations API."""

    name = "eval_samples"

    def __init__(self, client: PlatformClient) -> None:
        self.enabled = True
        self._client = client
        self._run_id: Optional[str] = None
        # Carried across calls so a streaming producer numbers rollouts the
        # same way a one-shot upload did.
        self._rollout_numbers: Dict[Any, int] = {}
        self.samples_written = 0

    def start(self, run_id: str, context: Mapping[str, str]) -> None:
        self._run_id = run_id

    def write(self, records: Sequence[Any]) -> None:
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
                idempotent=False,  # appends; a lost response must not duplicate rows
            )
            self.samples_written += len(batch)

    def _to_samples(self, records: Sequence[Any]) -> List[Dict[str, Any]]:
        """Episode objects are projected; v0 sample dicts (``sample_id``) pass
        through. Anything else fails loudly rather than vanishing."""
        samples: List[Dict[str, Any]] = []
        for record in records:
            if isinstance(record, Mapping) and "sample_id" in record:
                samples.append(dict(record))
            elif not isinstance(record, Mapping) and is_episode(record):
                samples.extend(build_samples([record], self._rollout_numbers))
            else:
                raise TypeError(
                    f"EvalSamplesSink cannot project {type(record).__name__}; expected an "
                    "episode object or a v0 sample dict"
                )
        return samples

    def flush(self) -> None:
        """Writes are synchronous; the uploader thread owns the asynchrony."""

    def close(self) -> None:
        """The client is shared with the backend, which closes it."""
