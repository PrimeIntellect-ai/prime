"""One process-wide ``os.register_at_fork`` hook for everything holding a
connection. A child must *drop* inherited transports (closing one can write to
the parent's socket) and never flush inherited buffers (those records are the
parent's). Registration is weak and the hook is installed once, since
``register_at_fork`` cannot be undone."""

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
    global _lock
    _lock = threading.Lock()  # the inherited one may be held by a thread that is not here
    for obj in list(_registry):
        try:
            obj.reset_after_fork()
        except Exception as exc:  # noqa: BLE001 - a fork hook must never raise
            logger.debug("reset_after_fork failed for %r: %s", obj, exc)
