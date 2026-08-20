"""Types shared across backends, sinks and the ``Run`` handle.

Only the values that cross a module boundary live here. Response bodies are
deliberately *not* modeled: the platform returns more fields than any producer
reads, and freezing them in pydantic here would make every backend addition a
breaking SDK release. Backends pull the two or three fields they need and hand
back a ``RunHandle``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

RunKind = Literal["eval", "train"]
"""Which run system owns the lifecycle. Selects the backend."""

Mode = Literal["online", "offline", "disabled"]
"""``online`` talks to the platform, ``offline`` writes a local run directory
that can be synced later, ``disabled`` makes every call a no-op while keeping
the same object shape so producer code needs no branching."""

OnError = Literal["warn", "raise"]


class RunStatus(str, Enum):
    """Terminal state a producer can report.

    ``crashed`` is distinct from ``failed``: ``failed`` means the producer
    decided the run failed, ``crashed`` means the process exited without ever
    saying. Only the second one is inferred by the SDK (atexit / signal), and
    the distinction is what tells an operator whether to look at the run's own
    error or at the machine it ran on.
    """

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CRASHED = "crashed"

    def is_terminal(self) -> bool:
        return self is not RunStatus.RUNNING


@dataclass
class EnvironmentRef:
    """An environment as a producer names it, before hub resolution.

    ``id`` short-circuits resolution; ``name`` goes through the hub's
    get-or-create so a local run uploads without a prior ``prime env push``;
    ``slug`` looks up an already-published ``owner/name`` environment.
    """

    name: Optional[str] = None
    id: Optional[str] = None
    version_id: Optional[str] = None
    slug: Optional[str] = None

    @classmethod
    def coerce(cls, value: Any) -> "EnvironmentRef":
        if isinstance(value, EnvironmentRef):
            return value
        if isinstance(value, str):
            return cls(slug=value) if "/" in value else cls(name=value)
        if isinstance(value, dict):
            return cls(
                name=value.get("name"),
                slug=value.get("slug"),
                id=value.get("id"),
                version_id=value.get("version_id"),
            )
        raise TypeError(
            "environments entries must be a str, dict or EnvironmentRef, "
            f"got {type(value).__name__}"
        )

    def __post_init__(self) -> None:
        if not self.name and not self.slug and not self.id:
            raise ValueError("EnvironmentRef needs a name, slug or id")
        if self.slug:
            owner, name = self.slug.split("/", 1) if "/" in self.slug else ("", "")
            if not owner or not name:
                raise ValueError("EnvironmentRef slug must use owner/name format")


@dataclass
class RunSpec:
    """Everything a backend needs to open a run, in producer vocabulary.

    This is the argument surface of ``init()`` after normalization — backends
    translate it into whatever their API family calls these things, which is
    the whole reason eval and training runs can share one handle.
    """

    name: Optional[str] = None
    kind: RunKind = "eval"
    environments: List[EnvironmentRef] = field(default_factory=list)
    model: Optional[str] = None
    framework: Optional[str] = None
    dataset: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    team_id: Optional[str] = None
    # W&B's split, which maps cleanly onto the platform's existing columns:
    # `config` is what you set going in (-> metadata), `summary` is what came
    # out (-> metrics).
    config: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunHandle:
    """What a backend returns once the run exists on the other side."""

    id: str
    name: Optional[str] = None
    url: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
