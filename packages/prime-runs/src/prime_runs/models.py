"""Types shared across backends, sinks and the ``Run`` handle. Response bodies
are not modeled: backends pull the fields they need into a ``RunHandle``."""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from .exceptions import ConfigurationError

Mode = Literal["online", "disabled"]
OnError = Literal["warn", "raise"]

RunKind = Literal["eval", "train"]
"""Stamped as ``run.type`` on records and sent as upload provenance; the traces
service's vocabulary, matching verifiers' ``EvalRunInfo`` / ``TrainRunInfo``."""

RUN_KIND: RunKind = "eval"


class RunStatus(str, Enum):
    """``failed``: the producer said so. ``cancelled``: stopped on purpose (an
    interrupt). ``crashed``: the process exited without saying (only the
    atexit hook reports it)."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CRASHED = "crashed"

    def is_terminal(self) -> bool:
        return self is not RunStatus.RUNNING


@dataclass
class EnvironmentRef:
    """``id`` short-circuits resolution; ``name`` goes through the hub's
    get-or-create; ``slug`` looks up a published ``owner/name`` environment."""

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
"""Where a config file lands inside a run's config, as a ``ConfigSource`` dict."""

MAX_CONFIG_SOURCE_BYTES = 256 * 1024

_CONFIG_SOURCE_FORMATS = {".toml": "toml", ".json": "json", ".yaml": "yaml", ".yml": "yaml"}


@dataclass
class ConfigSource:
    """The config file a run was started from, kept byte for byte. Nothing is
    redacted: keep credentials in the environment, not the file."""

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
        """Rebuild from stored metadata; ``None`` if the mapping is not one of ours."""
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
        """A ``str`` or ``PathLike`` is a *path*, never inline text."""
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, (str, os.PathLike)):
            return cls.from_file(value)
        raise TypeError(
            f"a config source must be a path or a ConfigSource, got {type(value).__name__}"
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
class TrainingSpec:
    """Display fields a training run is registered with. ``max_steps`` is
    required by the platform; ``0`` means unknown."""

    max_steps: int = 0
    batch_size: Optional[int] = None
    rollouts_per_example: Optional[int] = None
    seq_len: Optional[int] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")


@dataclass
class RunSpec:
    """``init()``'s arguments after normalization."""

    name: Optional[str] = None
    environments: List[EnvironmentRef] = field(default_factory=list)
    model: Optional[str] = None
    framework: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    team_id: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    kind: RunKind = RUN_KIND
    training: Optional[TrainingSpec] = None


@dataclass
class RunHandle:
    id: str
    name: Optional[str] = None
    url: Optional[str] = None
