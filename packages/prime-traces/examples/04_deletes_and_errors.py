#!/usr/bin/env python3
"""4 — Deleting things, and what failure looks like.

The least glamorous surface and the one most worth getting right: these are the
calls people reach for when something has already gone wrong. Every exception
below is a distinct thing the caller can *do*, which is the test for whether a
typed error is earning its place.

Run it:

    PRIME_API_KEY=... PRIME_TRACES_URL=https://dev-prime-traces.pintel.dev \
        uv run python examples/04_deletes_and_errors.py
"""

from _sample import new_run_id, sample_trace

from prime_traces import (
    AmbiguousDeleteError,
    ErrorCode,
    ForbiddenError,
    NotFoundError,
    PaymentRequiredError,
    RetryableAPIError,
    TracesClient,
    TransportError,
    UnauthorizedError,
    ValidationRejectedError,
)

RUN_ID = new_run_id("lifecycle")


def main() -> None:
    client = TracesClient()

    records = [sample_trace(RUN_ID, index=i) for i in range(5)]
    client.upload_records(iter(records), context={"source": "example"})
    stored = client.list(run_id=RUN_ID, limit=100).items
    print(f"seeded {RUN_ID} with {len(stored)} traces\n")

    # ------------------------------------------------------------------
    # One trace.
    # ------------------------------------------------------------------
    def remaining() -> int:
        return len(client.list(run_id=RUN_ID, limit=100).items)

    target = stored[0]
    client.delete(target.trace_id)
    print(f"deleted {target.trace_id[:20]}… -> {remaining()} left")

    # `created_at` is a pure performance hint: it lets the service prune on its
    # ordering-key prefix instead of searching. Correctness never depends on
    # it, but a hint that matches no stored copy is a 404 even though the trace
    # exists -- so pass the value from the summary you already hold, or nothing.
    second = stored[1]
    client.delete(second.trace_id, created_at=second.created_at.isoformat())
    print(f"deleted with a created_at hint -> {remaining()} left")

    # ------------------------------------------------------------------
    # Deleting something that is already gone is an error, not a no-op.
    # ------------------------------------------------------------------
    # Worth knowing, because "make sure this is gone" is the more common intent
    # and it needs a try/except to express. The design docs call deletion
    # idempotent; the service checks existence first and answers 404.
    try:
        client.delete(target.trace_id)
    except NotFoundError as exc:
        print(f"\nrepeat delete -> NotFoundError: {exc}")

    def ensure_absent(trace_id: str) -> None:
        """What most callers actually mean by "delete"."""
        try:
            client.delete(trace_id)
        except NotFoundError:
            pass

    ensure_absent(target.trace_id)
    print("ensure_absent() swallows it")

    # ------------------------------------------------------------------
    # A whole run, in one call.
    # ------------------------------------------------------------------
    # One mutation over the run predicate rather than N per-trace deletes, and
    # synchronous -- 202 with an empty body, so there is no job to poll.
    client.delete_run(RUN_ID)
    print(f"\ndelete_run -> {remaining()} traces left")

    # ------------------------------------------------------------------
    # The failure vocabulary.
    # ------------------------------------------------------------------
    # Each of these implies a different next action, which is the only reason
    # to have separate types:
    #
    #   UnauthorizedError       the token is bad          -> re-authenticate
    #   ForbiddenError          not enabled, or no scope  -> ask for access
    #   ValidationRejectedError the payload is wrong      -> fix and re-upload
    #   NotFoundError           it is not there           -> often ignorable
    #   PaymentRequiredError    the account cannot pay    -> top up
    #   RetryableAPIError       transient, safe to replay -> back off, retry
    #   TransportError          never reached the service -> retry
    #   AmbiguousDeleteError    delivery unknown          -> do NOT replay
    #
    # `AmbiguousDeleteError` is the one that pays for the whole hierarchy. A
    # response-path failure on a delete means the deletion may already have
    # happened; replaying it could destroy a *new* trace uploaded in between.
    # So the SDK refuses to retry and hands the ambiguity to the caller, who is
    # the only one who knows whether that risk is acceptable.
    print("\nfailure handling, in the shape callers should copy:")
    try:
        client.get("tr_definitely_not_a_real_trace")
    except NotFoundError as exc:
        # `code` comes off the wire as a plain string, but `ErrorCode` is a
        # str-enum, so comparing against it works and beats matching on
        # message text. Note `==`, not `is`: the attribute is a `str`, not an
        # `ErrorCode` member, so identity comparison silently returns False.
        print(f"  NotFoundError  code={exc.code!r}")
        print(f"  is a missing trace: {exc.code == ErrorCode.TRACE_NOT_FOUND}")
        print(f"  (a missing run would be {ErrorCode.RUN_NOT_FOUND.value!r})")
    except (UnauthorizedError, ForbiddenError) as exc:
        print(f"  access problem: {exc}")
    except (RetryableAPIError, TransportError) as exc:
        print(f"  transient, retry later: {exc}")

    # Everything above derives from PrimeTracesError, so a caller that does not
    # care about the distinction can catch one type and move on.
    for exc_type in (
        UnauthorizedError,
        ForbiddenError,
        ValidationRejectedError,
        NotFoundError,
        PaymentRequiredError,
        RetryableAPIError,
        TransportError,
        AmbiguousDeleteError,
    ):
        assert issubclass(exc_type, Exception)
    print("  all of them share a common base, for callers that want one except")


if __name__ == "__main__":
    main()
