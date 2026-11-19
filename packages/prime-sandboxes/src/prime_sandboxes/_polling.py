"""Poll pacing shared by long-running client waits.

Sandbox creation waits, checkpoint (and volume-merge) waits, and volume-sync
waits all poll with the same law: a fixed fast cadence for the first seconds
of a phase, then jittered exponential backoff (capped lower for the checkpoint
and volume-sync waits).
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
# Backoff cap for those waits. A server phase that finishes after the fast
# window is seen at most this late; the 10 s creation cap would add up to ~10 s
# to every sync. Creation keeps its cap: it is the wait many clients run at once.
CHECKPOINT_POLL_MAX_DELAY = 2.0


def _jittered_poll_delay(delay: float) -> float:
    """Apply the poll jitter law to a base delay."""
    jitter = delay * CREATION_POLL_JITTER
    return max(0.0, delay + random.uniform(-jitter, jitter))


def _creation_poll_delay(poll_index: int, max_delay: float = CREATION_POLL_MAX_DELAY) -> float:
    """Jittered exponential backoff delay for the Nth status poll."""
    delay = min(
        CREATION_POLL_INITIAL_DELAY * (CREATION_POLL_BACKOFF_FACTOR**poll_index),
        max_delay,
    )
    return _jittered_poll_delay(delay)


def _checkpoint_poll_delay(poll_index: int) -> float:
    """Backoff delay for the Nth poll after the fast window of a checkpoint or sync wait."""
    return _creation_poll_delay(poll_index, CHECKPOINT_POLL_MAX_DELAY)
