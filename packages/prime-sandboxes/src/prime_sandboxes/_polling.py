"""Poll pacing shared by long-running client waits.

Sandbox creation waits, checkpoint (and volume-merge) waits, and volume-sync
waits all poll with the same law: a fixed fast cadence for the first seconds
of a phase, then jittered exponential backoff.
"""

import random

CREATION_POLL_INITIAL_DELAY = 1.0
CREATION_POLL_MAX_DELAY = 10.0
CREATION_POLL_BACKOFF_FACTOR = 1.5
# Fraction of the computed delay applied as +/- jitter.
CREATION_POLL_JITTER = 0.25

# Fast-cadence pacing for short server phases (checkpoint uploads, volume
# merges, volume syncs): waiting clients poll at a fixed fast delay for the
# first seconds of a phase and only then fall back to the backoff law.
CHECKPOINT_POLL_FAST_DELAY = 1.0
CHECKPOINT_POLL_FAST_WINDOW = 10.0


def _jittered_poll_delay(delay: float) -> float:
    """Apply the poll jitter law to a base delay."""
    jitter = delay * CREATION_POLL_JITTER
    return max(0.0, delay + random.uniform(-jitter, jitter))


def _creation_poll_delay(poll_index: int) -> float:
    """Jittered exponential backoff delay for the Nth status poll."""
    delay = min(
        CREATION_POLL_INITIAL_DELAY * (CREATION_POLL_BACKOFF_FACTOR**poll_index),
        CREATION_POLL_MAX_DELAY,
    )
    return _jittered_poll_delay(delay)
