"""Local evaluation run records for the Lab viewer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import LabItem


def legacy_row(data: dict[str, Any]) -> dict[str, Any]:
    """Back-translate a native v1 trace row into the legacy sample shape the viewer
    renders (prompt/completion/reward/trajectory); legacy rows pass through."""
    if "nodes" not in data and "rewards" not in data:
        return data
    from verifiers.v1.cli.output import convert_results_for_upload

    converted = convert_results_for_upload([data])
    return converted[0] if converted else {}


@dataclass
class LocalEvalRun:
    """A local evaluation run discovered from a verifiers output directory."""

    env_id: str
    model: str
    run_id: str
    path: Path
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_item(cls, item: LabItem) -> "LocalEvalRun":
        raw = item.raw
        return cls(
            env_id=str(raw.get("env_id") or "-"),
            model=str(raw.get("model") or "-"),
            run_id=str(raw.get("run_id") or item.title),
            path=Path(str(raw.get("path") or ".")),
            metadata=raw.get("metadata") if isinstance(raw.get("metadata"), dict) else None,
        )

    def load_metadata(self) -> dict[str, Any]:
        if self.metadata is not None:
            return self.metadata
        meta_path = self.path / "metadata.json"
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        self.metadata = loaded if isinstance(loaded, dict) else {}
        return self.metadata


@dataclass(frozen=True)
class MetricSummary:
    """Summary statistics for a numeric rollout metric."""

    name: str
    count: int
    avg: float
    min_value: float
    max_value: float


@dataclass(frozen=True)
class RunOverviewStats:
    """Aggregate statistics for a local eval run."""

    rewards: list[float]
    metric_summaries: list[MetricSummary]
    metric_values: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class HistorySectionData:
    """Renderable section in a rollout transcript."""

    title: str
    body: str
    column: str
    collapsed: bool = True
    classes: str = "history-section"
    nested_sections: tuple["HistorySectionData", ...] = ()
    body_first: bool = True


@dataclass(frozen=True)
class SearchHit:
    """A searchable line in rollout history or logs."""

    column: str
    line_index: int
    line_text: str
    section_index: int
    nested_index: int = -1


@dataclass(frozen=True)
class SearchResult:
    """Selected search hit metadata."""

    column: str
    pattern: str
    section_index: int
    nested_index: int = -1


@dataclass(frozen=True)
class RolloutCopyItem:
    """Copy target for rollout viewer modals."""

    key: str
    label: str
    body: str


class LazyRunResults:
    """Lazy loader for a verifiers results.jsonl file."""

    def __init__(self, run: LocalEvalRun):
        self._path = run.path / "results.jsonl"
        self._fh = self._path.open("r", encoding="utf-8")
        self._offsets: list[int] = []
        self._cache: dict[int, dict[str, Any]] = {}
        self._eof = False
        self._count_hint: int | None = None
        self._count: int | None = None

        metadata = run.load_metadata()
        num_examples = metadata.get("num_examples")
        rollouts_per_example = metadata.get("rollouts_per_example")
        if isinstance(num_examples, int) and num_examples >= 0:
            if isinstance(rollouts_per_example, int) and rollouts_per_example >= 0:
                self._count_hint = num_examples * rollouts_per_example
            else:
                self._count_hint = num_examples

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def _read_next_line(self) -> str | None:
        if self._eof:
            return None
        position = self._fh.tell()
        line = self._fh.readline()
        if not line:
            self._eof = True
            self._count = len(self._offsets)
            return None
        self._offsets.append(position)
        return line

    def _ensure_index(self, index: int) -> bool:
        if index < 0:
            return False
        while len(self._offsets) <= index and not self._eof:
            if self._read_next_line() is None:
                break
        return index < len(self._offsets)

    def _ensure_count(self) -> int:
        if self._count is not None:
            return self._count
        while not self._eof:
            if self._read_next_line() is None:
                break
        self._count = len(self._offsets)
        return self._count

    def get(self, index: int) -> dict[str, Any]:
        if index in self._cache:
            return self._cache[index]
        if not self._ensure_index(index):
            return {}
        position = self._fh.tell()
        try:
            self._fh.seek(self._offsets[index])
            line = self._fh.readline()
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                data = {}
        finally:
            self._fh.seek(position)
        data = legacy_row(data) if isinstance(data, dict) else {}
        self._cache[index] = data
        return data

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get(index)

    def __len__(self) -> int:
        return self._ensure_count()

    def __bool__(self) -> bool:
        if self._count is not None:
            return self._count > 0
        if self._offsets:
            return True
        if self._eof:
            return False
        return self._read_next_line() is not None

    def count_hint(self) -> int | None:
        return self._count if self._count is not None else self._count_hint


class LazyLogFile:
    """Lazy loader for a log file with line-level random access."""

    MAX_DISPLAY_LINES = 10_000

    def __init__(self, path: Path):
        self._path = path
        self._fh = path.open("r", encoding="utf-8", errors="replace")
        self._offsets: list[int] = []
        self._cache: dict[int, str] = {}
        self._eof = False
        self._count: int | None = None

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def _read_next_line(self) -> str | None:
        if self._eof:
            return None
        position = self._fh.tell()
        line = self._fh.readline()
        if not line:
            self._eof = True
            self._count = len(self._offsets)
            return None
        self._offsets.append(position)
        return line

    def _ensure_index(self, index: int) -> bool:
        if index < 0:
            return False
        while len(self._offsets) <= index and not self._eof:
            if self._read_next_line() is None:
                break
        return index < len(self._offsets)

    def _ensure_count(self) -> int:
        if self._count is not None:
            return self._count
        while not self._eof:
            if self._read_next_line() is None:
                break
        self._count = len(self._offsets)
        return self._count

    def get_line(self, index: int) -> str:
        if index in self._cache:
            return self._cache[index]
        if not self._ensure_index(index):
            return ""
        position = self._fh.tell()
        try:
            self._fh.seek(self._offsets[index])
            line = self._fh.readline().rstrip("\n\r")
        finally:
            self._fh.seek(position)
        self._cache[index] = line
        return line

    def __len__(self) -> int:
        return self._ensure_count()

    def __bool__(self) -> bool:
        if self._count is not None:
            return self._count > 0
        if self._offsets:
            return True
        if self._eof:
            return False
        return self._read_next_line() is not None


def discover_log_files(run_path: Path) -> list[Path]:
    """Find log files in a run directory, sorted by likely usefulness."""

    log_files = sorted(run_path.glob("*.log"))
    server_logs = [path for path in log_files if path.name == "env_server.log"]
    worker_logs = sorted(
        [path for path in log_files if path.name.startswith("env_worker_")],
        key=lambda path: (
            int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else 0
        ),
    )
    other_logs = [path for path in log_files if path not in server_logs and path not in worker_logs]
    return server_logs + worker_logs + other_logs


def log_tab_label(path: Path) -> str:
    """Derive a short tab label from a log file path."""

    stem = path.stem
    return stem[4:] if stem.startswith("env_") else stem


def merge_log_files(log_files: list[Path]) -> list[str]:
    """Merge log lines from multiple files by timestamp when possible."""

    entries: list[tuple[str, int, str]] = []
    for file_idx, path in enumerate(log_files):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        current_ts = ""
        for line in lines:
            parsed = parse_log_header(line)
            if parsed is not None:
                current_ts = parsed[0]
            entries.append((current_ts, file_idx, line))
    entries.sort(key=lambda entry: entry[0])
    return [line for _, _, line in entries]


def parse_log_header(line: str) -> tuple[str, str, str, str] | None:
    """Parse a verifiers log line into timestamp, source, level, and message."""

    if len(line) < 22 or line[19:22] != " - ":
        return None
    rest = line[22:]
    separator_idx = rest.find(" - ")
    if separator_idx < 0:
        return None
    source = rest[:separator_idx]
    after_source = rest[separator_idx + 3 :]
    space_idx = after_source.find(" ")
    if space_idx < 0:
        level = after_source
        message = ""
    else:
        level = after_source[:space_idx]
        message = after_source[space_idx:]
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        return None
    return line[:19], source, level, message
