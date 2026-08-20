"""Types shared across backends, sinks and the ``Run`` handle.

Only the values that cross a module boundary live here. Response bodies are
deliberately *not* modeled: the platform returns more fields than any producer
reads, and freezing them in pydantic here would make every backend addition a
breaking SDK release. Backends pull the two or three fields they need and hand
back a ``RunHandle``.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from .exceptions import ConfigurationError

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


CONFIG_SOURCE_KEY = "config_source"
"""Reserved key inside a run's config holding :class:`ConfigSource` as a dict.

It rides *inside* the config rather than beside it so that every path which
already carries the config carries the source too — create, the periodic update,
finalize, the failure fallback, and the offline archive — with no extra
plumbing and no chance of one of them forgetting it.
"""

MAX_CONFIG_SOURCE_BYTES = 256 * 1024
"""Ceiling on a stored config file. A hand-written run config is single-digit
kilobytes; anything past this is a dataset or a log that would bloat the run's
metadata document, so it is refused loudly at ``init()`` rather than truncated."""

_CONFIG_SOURCE_FORMATS = {
    ".toml": "toml",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}


@dataclass
class ConfigSource:
    """The config file a run was started from, kept byte-for-byte.

    Both producers are now launched from one user-authored file — ``uv run eval
    @ eval.toml``, ``uv run rl @ train.toml`` — and that file *is* the run's
    real configuration. A resolved model dump is a different artifact: it
    answers "what did every knob end up as", not "what did someone write", and
    it loses comments, key order and section grouping on the way through.

    So this is stored verbatim, next to (not instead of) the structured config.
    The structured form stays queryable; this form stays readable.

    Nothing here is redacted. A config file that carries a secret will carry it
    onto the run's page, the same way it already reaches anyone who can read the
    repository it lives in — keep credentials in the environment, not the file.
    """

    text: str
    format: str = "toml"
    filename: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"format": self.format, "text": self.text}
        if self.filename:
            data["filename"] = self.filename
        return data

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Optional["ConfigSource"]:
        """Rebuild from stored metadata. ``None`` if the mapping is not one of ours."""
        text = value.get("text")
        if not isinstance(text, str):
            return None
        raw_format = value.get("format")
        raw_filename = value.get("filename")
        return cls(
            text=text,
            format=str(raw_format) if raw_format else "toml",
            filename=str(raw_filename) if raw_filename else None,
        )

    @classmethod
    def from_file(cls, path: Union[str, "os.PathLike[str]"]) -> "ConfigSource":
        """Read a config file, inferring its format from the suffix."""
        resolved = Path(path)
        try:
            raw = resolved.read_bytes()
        except FileNotFoundError as exc:
            raise ConfigurationError(
                f"config_source={str(path)!r} does not exist. Pass the path to the file the "
                "run was started from, or a ConfigSource(text=...) if it is already in memory."
            ) from exc
        except OSError as exc:
            raise ConfigurationError(
                f"config_source={str(path)!r} could not be read: {exc}"
            ) from exc
        if len(raw) > MAX_CONFIG_SOURCE_BYTES:
            raise ConfigurationError(
                f"config_source={str(path)!r} is {len(raw)} bytes, over the "
                f"{MAX_CONFIG_SOURCE_BYTES}-byte limit for a stored run config."
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigurationError(
                f"config_source={str(path)!r} is not UTF-8 text; a run config must be readable."
            ) from exc
        return cls(
            text=text,
            format=_CONFIG_SOURCE_FORMATS.get(resolved.suffix.lower(), "text"),
            filename=resolved.name,
        )

    @classmethod
    def coerce(cls, value: Any) -> Optional["ConfigSource"]:
        """Normalize whatever ``init(config_source=...)`` was given.

        A ``str`` or ``PathLike`` is a *path*, never inline text — that is how
        every caller will reach for it, and guessing between the two would turn
        a mistyped filename into a run whose config tab shows the filename.
        Inline text goes through ``ConfigSource(text=...)`` explicitly.
        """
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            source = cls.from_mapping(value)
            if source is None:
                raise ValueError("config_source mapping must contain a 'text' string")
            return source
        if isinstance(value, (str, os.PathLike)):
            return cls.from_file(value)
        raise TypeError(
            f"config_source must be a path, a ConfigSource or a mapping, got {type(value).__name__}"
        )

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("ConfigSource.text must be a string")
        if len(self.text.encode("utf-8")) > MAX_CONFIG_SOURCE_BYTES:
            raise ConfigurationError(
                f"config_source is over the {MAX_CONFIG_SOURCE_BYTES}-byte limit "
                "for a stored run config."
            )


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
