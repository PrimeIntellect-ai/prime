"""One process-wide ``os.register_at_fork`` hook, shared by everything stateful.

A forked child inherits copies of every open socket and buffered file handle.
Anything holding a connection or a file registers here and gets told to start
over in the child. Two rules for ``reset_after_fork``:

- **Drop, do not close.** Closing an inherited transport can send bytes down a
  socket the parent is still using.
- **Do not flush.** An inherited buffer holds records the parent will write.

Registration is weak and the hook is installed once: ``register_at_fork``
cannot be undone, so per-instance registration would pin every run forever.
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
