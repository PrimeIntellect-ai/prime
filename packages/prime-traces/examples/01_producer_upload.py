#!/usr/bin/env python3
"""1 — Getting a finished run into Prime Traces.

The producer's view: a training or evaluation run just finished and its traces
need to land somewhere durable. This is the path most callers will only ever
touch once, from inside a harness, so it is the one most worth making boring.

Run it:

    PRIME_API_KEY=... PRIME_TRACES_URL=https://dev-prime-traces.pintel.dev \
        uv run python examples/01_producer_upload.py
"""

import json
import tempfile
from pathlib import Path

from _sample import new_run_id, sample_trace

from prime_traces import Batch, TracesClient, UploadReceipt

RUN_ID = new_run_id("producer")


class Rollout:
    """Stands in for a verifiers `Trace` or a prime-rl `Rollout`.

    The SDK depends on neither package. It accepts anything exposing
    `to_record()`, which both already do -- so a producer hands over its own
    objects rather than converting them first.
    """

    def __init__(self, run_id: str, index: int) -> None:
        self._run_id, self._index = run_id, index

    def to_record(self) -> dict:
        return sample_trace(self._run_id, index=self._index, reward=self._index / 10)


def main() -> None:
    client = TracesClient()  # PRIME_API_KEY / PRIME_TRACES_URL / ~/.prime/config.json

    # ------------------------------------------------------------------
    # From memory, straight off the producer's objects.
    # ------------------------------------------------------------------
    # `records` is consumed lazily, so a run that produced more traces than fit
    # in memory still uploads in bounded batches without a temporary file.
    def finished_rollouts():
        for i in range(8):
            yield Rollout(RUN_ID, i)

    def show(batch: Batch, receipt: UploadReceipt) -> None:
        mib = batch.size / (1024 * 1024)
        print(
            f"  {receipt.upload_id[:12]}…  {batch.num_lines:>4} lines  "
            f"{mib:5.2f} MiB  {receipt.status}"
        )

    print(f"uploading run {RUN_ID}")
    receipts = client.upload_records(
        finished_rollouts(),
        # Context is free-form and travels with the batch, not the trace. It is
        # how you answer "where did these come from" months later.
        context={"source": "hosted_eval", "harness": "verifiers"},
        on_batch=show,
    )
    print(f"  -> {len(receipts)} batch(es)\n")

    # ------------------------------------------------------------------
    # From a completed JSONL file.
    # ------------------------------------------------------------------
    # The same batching path. Use this when the harness already writes JSONL and
    # you would rather not hold the run in memory at all.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "traces.jsonl"
        path.write_text("".join(json.dumps(sample_trace(RUN_ID, index=i)) + "\n" for i in range(4)))

        print("uploading the same run from a file")
        from_file = client.upload_file(path, context={"source": "hosted_eval"}, on_batch=show)

        # ------------------------------------------------------------------
        # Interrupted? Just run it again.
        # ------------------------------------------------------------------
        # Every batch is identified by the SHA-256 of its exact uncompressed
        # bytes. Re-uploading the same file replays the committed receipts
        # instead of storing a second copy -- so the recovery story for a
        # half-finished upload is "run the same command", with no bookkeeping
        # on the producer's side.
        print("\nre-uploading the identical file")
        again = client.upload_file(path, context={"source": "hosted_eval"}, on_batch=show)
        replayed = [r.upload_id for r in again] == [r.upload_id for r in from_file]
        print(f"  identical upload_ids, nothing stored twice: {replayed}")

    total = sum(1 for _ in client.iter(run_id=RUN_ID))
    print(f"\n{total} traces stored for {RUN_ID}")

    # Examples clean up after themselves so they can be run repeatedly.
    client.delete_run(RUN_ID)
    print(f"deleted {RUN_ID}")


if __name__ == "__main__":
    main()
