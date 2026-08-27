#!/usr/bin/env python3
"""
Basic usage example for the prime-traces SDK.

This demonstrates the standalone SDK without any CLI dependencies.

Credentials come from PRIME_API_KEY / ~/.prime/config.json. The client talks
to the production Prime Traces service by default; set PRIME_TRACES_URL to
point it elsewhere — e.g. the service's local compose stack:
PRIME_TRACES_URL=http://localhost:8083
"""

import tempfile
import time
from pathlib import Path

from prime_traces import APIError, ForbiddenError, TracesClient, ValidationRejectedError


def sample_trace(trace_id: str, reward: float, started_at: float) -> dict:
    """One in-memory Verifiers-compatible trace record.

    Trimmed to the fields the service's extractor reads, which is a small
    subset of a real v1 record — the serialized record is stored verbatim,
    and every summary column is a projection of it. Two of these are not
    optional: a non-empty string ``id``, and a numeric ``timing.start`` inside
    the accepted window (generous lookback, tight lookahead — producers upload
    completed files, so an old run is ordinary and a future one never is).
    A line missing either is rejected with ``invalid_trace`` /
    ``created_at_out_of_window`` and the whole request stores nothing.
    """
    return {
        "version": 4,
        "id": trace_id,
        "run": {"id": "run_example"},
        "task": {"type": "ExampleTask", "data": {"name": "example-0001"}},
        "agent": {
            "name": "solver",
            "config": {
                "model": "deepseek-v4-flash",
                "client": {"base_url": "https://api.pinference.ai/api/v1"},
            },
        },
        "calls": [{"model": "deepseek-v4-flash", "usage": {"total_tokens": 1834}}],
        "rewards": {"correctness": {"score": reward, "weight": 1.0}},
        "metrics": {},
        "stop_condition": "done",
        "ok": True,
        "errors": [],
        # `timing.start` is the producer's wall clock and becomes `created_at`;
        # `timing.scoring.end` is the last phase, so duration_ms comes from the
        # two together.
        "timing": {"start": started_at, "scoring": {"end": started_at + 12.5}},
        "info": {},
    }


def main():
    # Verifiers Trace objects and prime-rl Rollouts can be passed directly;
    # both expose the same to_record() protocol accepted by upload_records.
    started_at = time.time()
    traces = [
        sample_trace("3f2a9c1e", reward=0.85, started_at=started_at),
        sample_trace("b81d4e77", reward=0.40, started_at=started_at + 1.0),
    ]
    output_dir = Path(tempfile.mkdtemp())

    with TracesClient() as client:
        try:
            print("Uploading...")
            # Content-addressed: rerunning this replays committed receipts
            # without storing anything twice.
            receipts = client.upload_records(
                traces,
                context={"source": "example", "suite_commit": "a1f39c2"},
            )
            for receipt in receipts:
                print(f"✓ upload {receipt.upload_id[:12]}… {receipt.status}")

            print("\nListing this run's traces...")
            page = client.list(run_id="run_example", limit=10)
            for summary in page.items:
                # `score` is a nested object, and a null `reward` inside it
                # means unscored — distinct from a scored 0.0.
                score = summary.score
                reward = score.reward if score else None
                print(f"  {summary.trace_id}  reward={reward}")

            if page.items:
                trace_id = page.items[0].trace_id
                print(f"\nFetching raw document for {trace_id}...")
                dest = output_dir / "trace.json"
                written = client.download_raw(trace_id, dest)
                print(f"✓ wrote {written} bytes to {dest}")

        except ValidationRejectedError as error:
            # A 400 rejects the whole request and stores nothing; branch on
            # error.code (see prime_traces.ErrorCode), fix the file, rerun.
            print(f"✗ rejected: {error.code}: {error}")
        except ForbiddenError as error:
            # `service_not_enabled` means the account is not in the private
            # beta; `forbidden` means the token lacks traces:read/traces:write.
            print(f"✗ not permitted: {error.code}: {error}")
        except APIError as error:
            print(f"✗ API error: {error}")


if __name__ == "__main__":
    main()
