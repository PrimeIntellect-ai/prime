"""Prime-owned environment config command support."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import typer
from rich.table import Table

from .utils import get_console, is_plain_mode, output_data_as_json
from .verifiers_bridge import (
    DEFAULT_ENV_DIR_PATH,
    ResolvedEnvironment,
    _prepare_single_environment,
    run_eval_passthrough,
)
from .verifiers_plugin import load_verifiers_prime_plugin

console = get_console()

ENV_CONFIG_FORMATS = {"text", "toml", "json"}

CONFIG_API_MODULE_CANDIDATES = (
    "verifiers",
    "verifiers.config_introspection",
    "verifiers.utils.config_introspection",
    "verifiers.utils.env_config_introspection",
    "verifiers.utils.env_config_utils",
    "verifiers.v1.config_introspection",
)

CONFIG_SURFACE_FUNCTIONS = (
    "inspect_environment_config",
    "inspect_env_config",
    "get_environment_config_surface",
    "get_env_config_surface",
    "environment_config_surface",
    "env_config_surface",
    "discover_environment_config",
    "discover_env_config",
)

DOCTOR_FUNCTIONS = (
    "doctor_environment",
    "doctor_env",
    "diagnose_environment",
    "diagnose_env",
    "diagnose_environment_config",
    "diagnose_env_config",
    "check_environment_config",
    "check_env_config",
)

PYPROJECT_FUNCTIONS = (
    "diagnose_pyproject_config",
    "validate_pyproject_config",
    "diagnose_pyproject",
    "validate_pyproject",
)


@dataclass(frozen=True)
class EnvDoctorCheck:
    name: str
    status: str
    detail: str = ""
    source: str = "prime-cli"


class ConfigIntrospectionUnavailable(RuntimeError):
    pass


def validate_env_config_format(output_format: str) -> None:
    if output_format not in ENV_CONFIG_FORMATS:
        console.print("[red]Error:[/red] --format must be one of: text, toml, json")
        raise typer.Exit(1)


@contextmanager
def _resolution_console(messages_to_stderr: bool):
    if not messages_to_stderr:
        yield
        return

    import prime_cli.commands.env as env_commands
    import prime_cli.verifiers_bridge as bridge

    stderr_console = get_console(stderr=True, force_terminal=sys.stderr.isatty())
    old_bridge_console = bridge.console
    old_env_console = env_commands.console
    bridge.console = stderr_console
    env_commands.console = stderr_console
    try:
        yield
    finally:
        bridge.console = old_bridge_console
        env_commands.console = old_env_console


def _resolve_for_eval(
    env_ref: str,
    env_dir_path: str,
    *,
    messages_to_stderr: bool,
) -> ResolvedEnvironment:
    status_console = (
        get_console(stderr=True, force_terminal=sys.stderr.isatty())
        if messages_to_stderr
        else console
    )
    plugin = load_verifiers_prime_plugin(console=status_console)
    with _resolution_console(messages_to_stderr):
        return _prepare_single_environment(plugin, env_ref, env_dir_path)


def _load_verifiers_config_introspection() -> Any:
    for module_name in CONFIG_API_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if any(callable(getattr(module, name, None)) for name in CONFIG_SURFACE_FUNCTIONS):
            return module

    raise ConfigIntrospectionUnavailable(
        "verifiers config introspection API is not available. "
        "Install verifiers from the companion config-introspection PR."
    )


def _find_callable(api: Any, names: Iterable[str], purpose: str) -> Callable[..., Any]:
    for name in names:
        candidate = getattr(api, name, None)
        if callable(candidate):
            return candidate
    raise ConfigIntrospectionUnavailable(
        f"verifiers config introspection API does not expose {purpose}."
    )


def _call_with_supported_kwargs(
    func: Callable[..., Any],
    kwargs: Mapping[str, Any],
    *,
    positional_env_id: str | None = None,
) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return func(**kwargs)

    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    positional_args: list[Any] = []
    for param in parameters.values():
        if param.default is not inspect.Parameter.empty:
            continue
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            continue
        if param.name in accepted:
            continue
        if positional_env_id is not None:
            positional_args.append(positional_env_id)
            break

    return func(*positional_args, **accepted)


def _env_config_instance(raw: Mapping[str, object] | None = None) -> Any:
    try:
        import verifiers as vf

        env_config_type = getattr(vf, "EnvConfig", None)
    except Exception:
        env_config_type = None

    if env_config_type is None:
        from verifiers.v1 import EnvConfig as env_config_type

    return env_config_type(raw or {})


def _module_name_for_env(env_name: str) -> str:
    return env_name.replace("-", "_").split("/")[-1]


def _pyproject_path_for_resolved_env(resolved: ResolvedEnvironment) -> Path | None:
    if resolved.local_env_path is not None:
        pyproject = resolved.local_env_path / "pyproject.toml"
        return pyproject if pyproject.exists() else None

    module_name = _module_name_for_env(resolved.env_name)
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None

    path = Path(spec.origin).resolve()
    for parent in (path.parent, *path.parents):
        pyproject = parent / "pyproject.toml"
        if pyproject.exists():
            return pyproject
        if parent == parent.parent:
            break
    return None


def _surface_kwargs(
    resolved: ResolvedEnvironment,
    env_config: Any,
    pyproject_path: Path | None,
    *,
    resolved_output: bool,
) -> dict[str, Any]:
    return {
        "env_id": resolved.env_name,
        "env_name": resolved.env_name,
        "environment": resolved.env_name,
        "environment_id": resolved.env_name,
        "reference": resolved.original,
        "config": env_config,
        "env_config": env_config,
        "resolved": resolved_output,
        "pyproject_path": pyproject_path,
        "path": resolved.local_env_path,
        "env_path": resolved.local_env_path,
        "resolved_environment": resolved,
    }


def _get_config_surface(
    api: Any,
    resolved: ResolvedEnvironment,
    env_config: Any,
    *,
    pyproject_path: Path | None,
    resolved_output: bool,
) -> Any:
    func = _find_callable(api, CONFIG_SURFACE_FUNCTIONS, "an environment config surface")
    return _call_with_supported_kwargs(
        func,
        _surface_kwargs(
            resolved,
            env_config,
            pyproject_path,
            resolved_output=resolved_output,
        ),
        positional_env_id=resolved.env_name,
    )


def _call_renderer(surface: Any, names: Sequence[str], **kwargs: Any) -> Any:
    for name in names:
        renderer = getattr(surface, name, None)
        if not callable(renderer):
            continue
        try:
            signature = inspect.signature(renderer)
        except (TypeError, ValueError):
            return renderer()
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        ):
            return renderer(**kwargs)
        accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return renderer(**accepted)
    return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(getattr(value, key)) for key in value.__dataclass_fields__.keys()}
    return str(value)


def _surface_json(surface: Any) -> Any:
    rendered = _call_renderer(surface, ("to_json", "json"))
    if isinstance(rendered, str):
        try:
            return json.loads(rendered)
        except json.JSONDecodeError:
            return rendered
    if rendered is not None:
        return _jsonable(rendered)
    return _jsonable(surface)


def _surface_text(surface: Any) -> str:
    rendered = _call_renderer(surface, ("to_text", "text", "render_text", "format_text"))
    if rendered is not None:
        return str(rendered).rstrip() + "\n"

    data = _surface_json(surface)
    if not isinstance(data, Mapping):
        return f"{data}\n"

    lines = ["Environment config"]
    for section_name, value in _iter_config_sections(data):
        lines.append("")
        lines.append(section_name)
        if not value:
            lines.append("  (no fields)")
            continue
        if isinstance(value, Mapping):
            for key, item in value.items():
                lines.append(f"  {key}: {_field_summary(item)}")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping) and "name" in item:
                    lines.append(f"  {item['name']}: {_field_summary(item)}")
                else:
                    lines.append(f"  - {item}")
        else:
            lines.append(f"  {value}")
    return "\n".join(lines).rstrip() + "\n"


def _iter_config_sections(data: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    env_data = data.get("env")
    config_data = data.get("config")
    candidates = (
        env_data if isinstance(env_data, Mapping) else None,
        config_data if isinstance(config_data, Mapping) else None,
        data,
    )
    for key, label in (
        ("args", "env.args"),
        ("taskset", "env.taskset"),
        ("harness", "env.harness"),
    ):
        for candidate in candidates:
            if candidate is not None and key in candidate:
                yield label, candidate[key]
                break


def _field_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return str(value)
    pieces: list[str] = []
    if value.get("required"):
        pieces.append("required")
    if value.get("type") is not None:
        pieces.append(str(value["type"]))
    if "default" in value:
        pieces.append(f"default={value['default']!r}")
    if "value" in value:
        pieces.append(f"value={value['value']!r}")
    description = value.get("description")
    if description:
        pieces.append(str(description))
    return ", ".join(pieces) if pieces else json.dumps(_jsonable(value), sort_keys=True)


def _surface_toml(surface: Any, *, resolved_output: bool) -> str:
    rendered = _call_renderer(
        surface,
        ("to_toml", "toml", "toml_template", "render_toml"),
        resolved=resolved_output,
    )
    if rendered is None:
        raise ConfigIntrospectionUnavailable(
            "verifiers config introspection API did not provide a TOML renderer."
        )
    return str(rendered).rstrip() + "\n"


def print_env_config(
    env_ref: str,
    *,
    output_format: str,
    resolved_output: bool,
    env_dir_path: str = DEFAULT_ENV_DIR_PATH,
) -> None:
    validate_env_config_format(output_format)
    messages_to_stderr = output_format == "json"
    resolved = _resolve_for_eval(env_ref, env_dir_path, messages_to_stderr=messages_to_stderr)
    env_config = _env_config_instance()
    pyproject_path = _pyproject_path_for_resolved_env(resolved)
    api = _load_verifiers_config_introspection()
    surface = _get_config_surface(
        api,
        resolved,
        env_config,
        pyproject_path=pyproject_path,
        resolved_output=resolved_output,
    )

    if output_format == "json":
        output_data_as_json(_surface_json(surface), console)
    elif output_format == "toml":
        sys.stdout.write(_surface_toml(surface, resolved_output=resolved_output))
    else:
        sys.stdout.write(_surface_text(surface))


def _diagnostics_from_result(result: Any) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, Mapping):
        diagnostics = result.get("diagnostics")
        if diagnostics is None:
            diagnostics = result.get("checks")
        if isinstance(diagnostics, Mapping):
            return list(diagnostics.values())
        if isinstance(diagnostics, Iterable) and not isinstance(diagnostics, (str, bytes)):
            return list(diagnostics)
        return [result]
    diagnostics = getattr(result, "diagnostics", None)
    if diagnostics is not None:
        if isinstance(diagnostics, Mapping):
            return list(diagnostics.values())
        return list(diagnostics)
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, Mapping)):
        return list(result)
    return [result]


def _value_from_diagnostic(diagnostic: Any, *names: str) -> Any:
    if isinstance(diagnostic, Mapping):
        for name in names:
            if name in diagnostic:
                return diagnostic[name]
        return None
    for name in names:
        if hasattr(diagnostic, name):
            return getattr(diagnostic, name)
    return None


def _check_from_diagnostic(diagnostic: Any) -> EnvDoctorCheck:
    severity = str(
        _value_from_diagnostic(diagnostic, "severity", "level", "status") or "error"
    ).lower()
    if severity in {"ok", "pass", "passed", "success", "info"}:
        status = "pass"
    elif severity in {"warn", "warning"}:
        status = "warn"
    else:
        status = "fail"

    code = _value_from_diagnostic(diagnostic, "code", "name", "check")
    message = _value_from_diagnostic(diagnostic, "message", "detail", "description")
    location = _value_from_diagnostic(diagnostic, "path", "section", "namespace")
    detail = str(message) if message is not None else str(diagnostic)
    if location:
        detail = f"{location}: {detail}"
    return EnvDoctorCheck(
        name=str(code or "Verifiers diagnostic"),
        status=status,
        detail=detail,
        source="verifiers",
    )


def _run_verifiers_doctor(
    api: Any,
    resolved: ResolvedEnvironment,
    env_config: Any,
    pyproject_path: Path | None,
) -> list[EnvDoctorCheck]:
    diagnostics: list[Any] = []
    doctor_func = next(
        (getattr(api, name) for name in DOCTOR_FUNCTIONS if callable(getattr(api, name, None))),
        None,
    )
    if doctor_func is not None:
        result = _call_with_supported_kwargs(
            doctor_func,
            _surface_kwargs(
                resolved,
                env_config,
                pyproject_path,
                resolved_output=True,
            ),
            positional_env_id=resolved.env_name,
        )
        diagnostics.extend(_diagnostics_from_result(result))

    if pyproject_path is not None:
        pyproject_func = next(
            (
                getattr(api, name)
                for name in PYPROJECT_FUNCTIONS
                if callable(getattr(api, name, None))
            ),
            None,
        )
        if pyproject_func is not None:
            result = _call_with_supported_kwargs(
                pyproject_func,
                {
                    "env_id": resolved.env_name,
                    "env_name": resolved.env_name,
                    "environment": resolved.env_name,
                    "pyproject_path": pyproject_path,
                    "path": pyproject_path,
                },
                positional_env_id=resolved.env_name,
            )
            diagnostics.extend(_diagnostics_from_result(result))

    if not diagnostics:
        return [
            EnvDoctorCheck(
                name="Verifiers diagnostics",
                status="pass",
                detail="No framework diagnostics reported.",
                source="verifiers",
            )
        ]
    return [_check_from_diagnostic(diagnostic) for diagnostic in diagnostics]


def _load_environment_with_typed_config(env_name: str, env_config: Any) -> Any:
    from verifiers.utils.env_utils import load_environment

    return load_environment(env_name, config=env_config)


def _run_smoke_eval(env_name: str, env_dir_path: str) -> None:
    run_eval_passthrough(
        env_name,
        ["-n", "1", "-r", "1", "--disable-tui", "--env-dir-path", env_dir_path],
        skip_upload=True,
        env_path=None,
    )


def run_env_doctor(
    env_ref: str,
    *,
    smoke: bool = False,
    env_dir_path: str = DEFAULT_ENV_DIR_PATH,
) -> int:
    checks: list[EnvDoctorCheck] = []
    resolved: ResolvedEnvironment | None = None

    try:
        resolved = _resolve_for_eval(env_ref, env_dir_path, messages_to_stderr=False)
        checks.append(
            EnvDoctorCheck(
                name="Environment resolves",
                status="pass",
                detail=resolved.env_display_id or resolved.env_name,
            )
        )
    except Exception as exc:
        checks.append(EnvDoctorCheck("Environment resolves", "fail", str(exc)))
        _print_doctor_checks(env_ref, checks)
        return 1

    module_name = _module_name_for_env(resolved.env_name)
    module: Any | None = None
    try:
        module = importlib.import_module(module_name)
        checks.append(EnvDoctorCheck("Package imports", "pass", module_name))
    except Exception as exc:
        checks.append(EnvDoctorCheck("Package imports", "fail", f"{module_name}: {exc}"))

    if module is not None:
        loader = getattr(module, "load_environment", None)
        if callable(loader):
            checks.append(EnvDoctorCheck("load_environment exists", "pass", module_name))
        else:
            checks.append(
                EnvDoctorCheck(
                    "load_environment exists",
                    "fail",
                    f"{module_name}.load_environment is missing or not callable.",
                )
            )

    env_config: Any | None = None
    if module is not None and callable(getattr(module, "load_environment", None)):
        try:
            env_config = _env_config_instance()
            _load_environment_with_typed_config(resolved.env_name, env_config)
            checks.append(
                EnvDoctorCheck("load_environment(config) loads", "pass", "Loaded with EnvConfig.")
            )
        except Exception as exc:
            checks.append(EnvDoctorCheck("load_environment(config) loads", "fail", str(exc)))

    api: Any | None = None
    pyproject_path = _pyproject_path_for_resolved_env(resolved)
    try:
        api = _load_verifiers_config_introspection()
        checks.append(
            EnvDoctorCheck(
                "Config introspection API",
                "pass",
                api.__name__ if hasattr(api, "__name__") else type(api).__name__,
                source="verifiers",
            )
        )
    except Exception as exc:
        checks.append(EnvDoctorCheck("Config introspection API", "fail", str(exc), "verifiers"))

    if api is not None:
        if env_config is None:
            try:
                env_config = _env_config_instance()
            except Exception as exc:
                checks.append(EnvDoctorCheck("EnvConfig constructs", "fail", str(exc), "verifiers"))

        if env_config is not None:
            try:
                surface = _get_config_surface(
                    api,
                    resolved,
                    env_config,
                    pyproject_path=pyproject_path,
                    resolved_output=True,
                )
                _surface_text(surface)
                _surface_toml(surface, resolved_output=False)
                checks.append(
                    EnvDoctorCheck(
                        "Config surface renders",
                        "pass",
                        "Text and TOML renderers returned successfully.",
                        source="verifiers",
                    )
                )
            except Exception as exc:
                checks.append(
                    EnvDoctorCheck("Config surface renders", "fail", str(exc), "verifiers")
                )

            try:
                checks.extend(_run_verifiers_doctor(api, resolved, env_config, pyproject_path))
            except Exception as exc:
                checks.append(
                    EnvDoctorCheck("Verifiers diagnostics", "fail", str(exc), "verifiers")
                )

    if pyproject_path is None:
        checks.append(EnvDoctorCheck("pyproject config", "warn", "No local pyproject.toml found."))
    else:
        checks.append(EnvDoctorCheck("pyproject config", "pass", str(pyproject_path)))

    if smoke:
        try:
            _run_smoke_eval(resolved.env_name, env_dir_path)
            checks.append(EnvDoctorCheck("Smoke eval", "pass", "1 example x 1 rollout."))
        except Exception as exc:
            checks.append(EnvDoctorCheck("Smoke eval", "fail", str(exc)))

    _print_doctor_checks(env_ref, checks)
    return 1 if any(check.status == "fail" for check in checks) else 0


def _print_doctor_checks(env_ref: str, checks: Sequence[EnvDoctorCheck]) -> None:
    if is_plain_mode():
        console.print(f"Environment doctor: {env_ref}", markup=False, soft_wrap=True)
        for check in checks:
            detail = f" - {check.detail}" if check.detail else ""
            console.print(
                f"{check.status.upper()} {check.name} [{check.source}]{detail}",
                markup=False,
                soft_wrap=True,
            )
        return

    table = Table(title=f"Environment doctor: {env_ref}")
    table.add_column("Status", no_wrap=True)
    table.add_column("Check", overflow="fold")
    table.add_column("Source", style="dim", no_wrap=True)
    table.add_column("Detail", overflow="fold")
    for check in checks:
        status = check.status.upper()
        style = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(status, "white")
        table.add_row(f"[{style}]{status}[/{style}]", check.name, check.source, check.detail)
    console.print(table)
