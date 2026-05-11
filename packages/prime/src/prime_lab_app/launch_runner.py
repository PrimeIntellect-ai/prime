"""Shared launch runner for config-backed Lab actions."""

from __future__ import annotations

import re
import shlex
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .palette import PRIMARY, STATUS_ERROR, STATUS_SUCCESS, STATUS_WARNING

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{5,}")
_TRAINING_URL_RE = re.compile(r"/dashboard/training/([A-Za-z0-9][A-Za-z0-9_-]{5,})")
_RUN_ID_HINT_RE = re.compile(
    r"\b(?:run[_ -]?id|training[_ -]?run(?:\s+id)?|created\s+run)\b\s*[:=]?\s*"
    r"([A-Za-z0-9][A-Za-z0-9_-]{5,})",
    re.IGNORECASE,
)
_LOG_RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0, 12.0)

LaunchOutputCallback = Callable[[str], None]
LaunchStatusCallback = Callable[[str, str], None]
LaunchDoneCallback = Callable[[str, int | None], None]
TrainingRunCreatedCallback = Callable[[str], None]


@dataclass(frozen=True)
class LogFollowCommand:
    run_id: str
    argv: tuple[str, ...]
    display: str


class ConfigLaunchRunner:
    """Run a Lab launch command and optionally hand off to live training logs."""

    def __init__(
        self,
        *,
        command: str,
        workspace: Path,
        follow_training_logs: bool = False,
        append_output: LaunchOutputCallback,
        update_status: LaunchStatusCallback,
        finish: LaunchDoneCallback,
        training_run_created: TrainingRunCreatedCallback | None = None,
        popen_factory: Callable[..., subprocess.Popen[str]] | None = None,
    ) -> None:
        self._command = command
        self._workspace = workspace
        self._follow_training_logs = follow_training_logs
        self._append_output = append_output
        self._update_status = update_status
        self._finish = finish
        self._training_run_created = training_run_created
        self._popen_factory = popen_factory or subprocess.Popen
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._completed = threading.Event()
        self._opened_training_run_id = ""

    def run(self) -> None:
        try:
            returncode, logs_command = self._stream_process(
                shlex.split(self._command),
                detect_training_logs=self._follow_training_logs,
            )
        except OSError as exc:
            self._append_output(f"Launch failed: {exc}\n")
            self._finish_once("launch", 1)
            return
        if self._stop_requested.is_set():
            self._finish_once("stopped", None)
            return
        if returncode == 0 and logs_command is not None and not self._stop_requested.is_set():
            if self._training_run_created is not None:
                self._finish_once("launch", returncode)
                return
            self._follow_logs_with_backoff(logs_command)
            return
        if returncode == 0 and self._follow_training_logs and logs_command is None:
            self._append_output(
                "\nTraining launch completed, but no live log command was detected.\n"
            )
        self._finish_once("launch", returncode)

    def stop(self) -> None:
        self._stop_requested.set()
        with self._process_lock:
            process = self._process
        if process is None or process.poll() is not None:
            self._finish_once("stopped", None)
            return
        process.terminate()
        self._append_output("\nStopped command.\n")

    def _follow_logs_with_backoff(self, logs_command: LogFollowCommand) -> None:
        self._append_output(f"\nFollowing run logs with: {logs_command.display}\n\n")
        self._update_status("Following live training logs", STATUS_SUCCESS)
        retry_count = 0
        while not self._stop_requested.is_set():
            try:
                returncode, _unused = self._stream_process(list(logs_command.argv))
            except OSError as exc:
                self._append_output(f"Log follow failed: {exc}\n")
                self._finish_once("launch", 1)
                return
            if self._stop_requested.is_set():
                self._finish_once("stopped", None)
                return
            if returncode == 0:
                self._finish_once("logs", 0)
                return
            delay = log_retry_delay(retry_count)
            retry_count += 1
            self._append_output(f"\nLogs are not ready yet; retrying in {delay:g}s.\n")
            self._stop_requested.wait(delay)
        self._finish_once("stopped", None)

    def _stream_process(
        self,
        command: list[str],
        *,
        detect_training_logs: bool = False,
    ) -> tuple[int, LogFollowCommand | None]:
        process = self._popen_factory(
            command,
            cwd=self._workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with self._process_lock:
            self._process = process
        logs_command: LogFollowCommand | None = None
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    if detect_training_logs and logs_command is None:
                        logs_command = extract_training_log_follow_command(line)
                        if logs_command is not None:
                            if self._training_run_created is None:
                                self._update_status("Run created. Preparing live logs", PRIMARY)
                            else:
                                self._update_status("Run created. Opening training view", PRIMARY)
                                self._notify_training_run_created(logs_command.run_id)
                    self._append_output(line)
            return process.wait(), logs_command
        finally:
            with self._process_lock:
                if self._process is process:
                    self._process = None

    def _notify_training_run_created(self, run_id: str) -> None:
        if self._training_run_created is None or not run_id:
            return
        if self._opened_training_run_id == run_id:
            return
        self._opened_training_run_id = run_id
        self._training_run_created(run_id)

    def _finish_once(self, kind: str, returncode: int | None) -> None:
        if self._completed.is_set():
            return
        self._completed.set()
        if kind == "stopped":
            self._update_status("Stopped", STATUS_WARNING)
        elif kind == "logs":
            self._update_status("Live log stream completed", STATUS_SUCCESS)
        elif returncode == 0:
            self._update_status("Launch command completed", STATUS_SUCCESS)
        else:
            self._update_status(f"Launch command exited with {returncode}", STATUS_ERROR)
        self._finish(kind, returncode)


def extract_training_log_follow_command(text: str) -> LogFollowCommand | None:
    cleaned = _ANSI_ESCAPE_RE.sub("", text).replace("`", " ")
    try:
        tokens = shlex.split(cleaned)
    except ValueError:
        tokens = cleaned.split()
    for index in range(len(tokens)):
        command_head = tokens[index : index + 3]
        if command_head in (["prime", "rl", "logs"], ["prime", "train", "logs"]):
            run_id = _first_run_id(tokens[index + 3 : index + 8])
            if run_id:
                return _training_log_follow_command(run_id)
        if tokens[index : index + 2] == ["prime", "logs"]:
            run_id = _first_run_id(tokens[index + 2 : index + 7])
            if run_id:
                return _training_log_follow_command(run_id)
    url_match = _TRAINING_URL_RE.search(cleaned)
    if url_match:
        return _training_log_follow_command(url_match.group(1))
    hint_match = _RUN_ID_HINT_RE.search(cleaned)
    if hint_match:
        return _training_log_follow_command(hint_match.group(1))
    return None


def log_retry_delay(retry_count: int) -> float:
    if retry_count < len(_LOG_RETRY_DELAYS):
        return _LOG_RETRY_DELAYS[retry_count]
    return _LOG_RETRY_DELAYS[-1]


def _first_run_id(tokens: list[str]) -> str:
    for token in tokens:
        stripped = token.strip().strip(".,;:)")
        if stripped.startswith("-"):
            continue
        if _RUN_ID_RE.fullmatch(stripped):
            return stripped
    return ""


def _training_log_follow_command(run_id: str) -> LogFollowCommand:
    argv = ("prime", "train", "logs", run_id, "-f")
    return LogFollowCommand(run_id, argv, " ".join(argv))
