"""One process-wide ``os.register_at_fork`` hook, shared by everything stateful.

Hosted evals fork after the SDK is initialized, and a forked child inherits far
more than the queue: it gets copies of every open socket and every buffered file
handle. Using those is not merely untidy — two processes writing the same TCP
connection interleave bytes into one HTTP stream, and a duplicated write buffer
gets flushed twice, once from each side.

So anything holding a connection or a file registers here and gets told to start
over in the child. Two rules for a ``reset_after_fork`` implementation:

- **Drop, do not close.** Closing an inherited transport can send bytes — a TLS
  ``close_notify``, an HTTP ``Connection: close`` — down a socket the parent is
  still using. Release the reference and let the child's copies of the
  descriptors go when it exits.
- **Do not flush.** A buffer inherited from the parent holds records the parent
  has not written yet and will write itself. Flushing it in the child writes
  them a second time.

Registration is weak and the hook is installed once. A per-object
``register_at_fork`` call cannot be undone, so registering per instance would
pin every run the process ever opened in memory and re-run hooks for runs that
finished hours ago.
"""

import logging
import os
import threading
import weakref
from typing import Any

logger = logging.getLogger(__name__)

_registry: "weakref.WeakSet[Any]" = weakref.WeakSet()
_lock = threading.Lock()
_installed = False


def register(obj: Any) -> None:
    """Have ``obj.reset_after_fork()`` called in any child forked from here."""
    global _installed
    with _lock:
        _registry.add(obj)
        if _installed or not hasattr(os, "register_at_fork"):  # pragma: no cover - Windows
            return
        os.register_at_fork(after_in_child=_reset_all)
        _installed = True


def _reset_all() -> None:
    # A fresh lock: the child inherits the parent's, which may have been held by
    # a thread that does not exist here. Nothing else runs at this point, so the
    # swap is safe.
    global _lock
    _lock = threading.Lock()
    for obj in list(_registry):
        try:
            obj.reset_after_fork()
        except Exception as exc:  # noqa: BLE001 - a fork hook must never raise
            logger.debug("reset_after_fork failed for %r: %s", obj, exc)
