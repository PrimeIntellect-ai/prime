#!/usr/bin/env python3
"""
Basic usage example for the prime-traces SDK.

This demonstrates the standalone SDK without any CLI dependencies.

Credentials come from PRIME_API_KEY / ~/.prime/config.json. Point
PRIME_TRACES_URL at the Prime Traces service — e.g. the service's local
compose stack: PRIME_TRACES_URL=http://localhost:8083
"""

import tempfile
from pathlib import Path

from prime_traces import APIError, TracesClient, ValidationRejectedError


def main():
    # A completed Verifiers JSONL file: one complete trace per line. For
    # episode-grouped files, pass line_format=LineFormat.EPISODE instead.
    traces_file = Path(tempfile.mkdtemp()) / "traces.jsonl"
    traces_file.write_bytes(
        b'{"id":"3f2a9c1e","run":{"id":"run_example"}}\n'
        b'{"id":"b81d4e77","run":{"id":"run_example"}}\n'
    )

    with TracesClient() as client:
        try:
            print("Uploading...")
            # Content-addressed: rerunning this replays committed receipts
            # without storing anything twice.
            receipts = client.upload_file(
                traces_file,
                context={"source": "example", "suite_commit": "a1f39c2"},
            )
            for receipt in receipts:
                print(f"✓ upload {receipt.upload_id[:12]}… {receipt.status}")

            print("\nListing this run's traces...")
            page = client.list(run_id="run_example", limit=10)
            for summary in page.traces:
                print(f"  {summary.trace_id}  reward={summary.score and summary.score.reward}")

            if page.traces:
                trace_id = page.traces[0].trace_id
                print(f"\nFetching raw document for {trace_id}...")
                dest = traces_file.with_name("trace.json")
                written = client.download_raw(trace_id, dest)
                print(f"✓ wrote {written} bytes to {dest}")

        except ValidationRejectedError as error:
            # A 400 rejects the whole request and stores nothing; branch on
            # error.code (see prime_traces.ErrorCode), fix the file, rerun.
            print(f"✗ rejected: {error.code}: {error}")
        except APIError as error:
            print(f"✗ API error: {error}")


if __name__ == "__main__":
    main()
