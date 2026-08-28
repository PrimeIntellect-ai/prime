"""Types shared across backends, sinks and the ``Run`` handle.

Response bodies are deliberately not modeled: backends pull the two or three
fields they need and hand back a ``RunHandle``.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from .exceptions import ConfigurationError

Mode = Literal["online", "disabled"]
"""``online`` talks to the platform; ``disabled`` makes every call a no-op with
the same object shape."""

OnError = Literal["warn", "raise"]

RUN_KIND = "eval"
"""Stamped as ``run.type`` on records and sent as upload provenance."""


class RunStatus(str, Enum):
    """Terminal state a producer can report.

    ``failed`` means the producer said the run failed; ``cancelled`` means
    somebody stopped it on purpose (Ctrl-C, a cancelled task) — a decision,
    not a fault; ``crashed`` means the process exited without saying (only the
    SDK's atexit hook reports it).
    """

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CRASHED = "crashed"

    def is_terminal(self) -> bool:
        return self is not RunStatus.RUNNING


@dataclass
class EnvironmentRef:
    """An environment as a producer names it, before hub resolution.

    ``id`` short-circuits resolution; ``name`` goes through the hub's
    get-or-create; ``slug`` looks up a published ``owner/name`` environment.
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
"""Where a config file lands inside a run's config, as a :class:`ConfigSource`
dict. Present means "render this verbatim"; absent means "render the structure"."""

MAX_CONFIG_SOURCE_BYTES = 256 * 1024
"""A hand-written run config is kilobytes; anything past this is refused at
``init()`` rather than truncated."""

_CONFIG_SOURCE_FORMATS = {
    ".toml": "toml",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}


@dataclass
class ConfigSource:
    """The config file a run was started from, kept byte-for-byte.

    That file *is* the run's configuration — comments, key order and section
    grouping included — where a resolved model dump is a different artifact.
    Nothing here is redacted: keep credentials in the environment, not the file.
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
                f"config={str(path)!r} does not exist. Pass the path to the file the run was "
                "started from, or a ConfigSource(text=...) if it is already in memory."
            ) from exc
        except OSError as exc:
            raise ConfigurationError(f"config={str(path)!r} could not be read: {exc}") from exc
        if len(raw) > MAX_CONFIG_SOURCE_BYTES:
            raise ConfigurationError(
                f"config={str(path)!r} is {len(raw)} bytes, over the "
                f"{MAX_CONFIG_SOURCE_BYTES}-byte limit for a stored run config."
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigurationError(
                f"config={str(path)!r} is not UTF-8 text; a run config must be readable."
            ) from exc
        return cls(
            text=text,
            format=_CONFIG_SOURCE_FORMATS.get(resolved.suffix.lower(), "text"),
            filename=resolved.name,
        )

    @classmethod
    def coerce(cls, value: Any) -> Optional["ConfigSource"]:
        """Normalize the config-file form of ``init(config=...)``.

        A ``str`` or ``PathLike`` is a *path*, never inline text; inline text
        goes through ``ConfigSource(text=...)`` explicitly.
        """
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            source = cls.from_mapping(value)
            if source is None:
                raise ValueError("a config-source mapping must contain a 'text' string")
            return source
        if isinstance(value, (str, os.PathLike)):
            return cls.from_file(value)
        raise TypeError(
            "a config source must be a path, a ConfigSource or a mapping, "
            f"got {type(value).__name__}"
        )

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("ConfigSource.text must be a string")
        if len(self.text.encode("utf-8")) > MAX_CONFIG_SOURCE_BYTES:
            raise ConfigurationError(
                f"the config source is over the {MAX_CONFIG_SOURCE_BYTES}-byte limit "
                "for a stored run config."
            )


@dataclass
class RunSpec:
    """Everything a backend needs to open a run: ``init()``'s arguments after
    normalization. ``config`` is the run's inputs; outputs accumulate on the
    handle as ``summary``."""

    name: Optional[str] = None
    environments: List[EnvironmentRef] = field(default_factory=list)
    model: Optional[str] = None
    framework: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    team_id: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunHandle:
    """What a backend returns once the run exists on the other side."""

    id: str
    name: Optional[str] = None
    url: Optional[str] = None
