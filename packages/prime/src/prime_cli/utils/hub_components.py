"""Which verifiers v1 components a package exports, read from its source.

A verifiers v1 package exports one class per kind of component in its
`__all__`: a taskset, and maybe a harness, an env or judges. The push declares
them so the Hub can list the package by kind before it indexes the release.

The source is read, never imported, so a push needs neither the package's
dependencies nor verifiers. Each exported name is followed to its class through
the package's own imports, then up its bases until one comes from verifiers,
whose class names end in the kind: `Taskset`, `Harness`, `Env` or `Judge`
(`vf.SingleAgentEnv`, `ReferenceJudge`). A name re-exported from verifiers (a
built-in harness, say) counts by its own name. Anything that can't be followed
statically is left out: the Hub indexes the release later and corrects the list.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# `*Config` classes (`HarnessConfig`) are never components.
_KIND_SUFFIXES = (
    ("Taskset", "taskset"),
    ("Harness", "harness"),
    ("Env", "env"),
    ("Judge", "judge"),
)
# The Hub's order for a release's components.
KIND_ORDER = tuple(kind for _suffix, kind in _KIND_SUFFIXES)
_MAX_DEPTH = 12


def kind_of_verifiers_name(name: str) -> Optional[str]:
    if name.endswith("Config"):
        return None
    for suffix, kind in _KIND_SUFFIXES:
        if name.endswith(suffix):
            return kind
    return None


@dataclass
class _Module:
    name: str
    path: Path
    is_package: bool
    classes: Dict[str, ast.ClassDef] = field(default_factory=dict)
    # Local name -> (module, name in that module); module is absolute.
    imported_names: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    # Local alias -> module, for `import verifiers.v1 as vf`.
    imported_modules: Dict[str, str] = field(default_factory=dict)
    exports: Optional[List[str]] = None


def _string_list(node: ast.expr) -> Optional[List[str]]:
    if isinstance(node, (ast.List, ast.Tuple)) and all(
        isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.elts
    ):
        return [item.value for item in node.elts]  # type: ignore[attr-defined]
    return None


class _Package:
    def __init__(self, root: Path, import_name: str):
        self.root = root
        self.import_name = import_name
        self._modules: Dict[str, Optional[_Module]] = {}

    def _file_of(self, module: str) -> Optional[Tuple[Path, bool]]:
        parts = module.split(".")
        for base in (self.root, self.root / "src"):
            package_init = base.joinpath(*parts, "__init__.py")
            if package_init.is_file():
                return package_init, True
            single = base.joinpath(*parts[:-1], f"{parts[-1]}.py")
            if single.is_file():
                return single, False
        return None

    def module(self, name: str) -> Optional[_Module]:
        if name in self._modules:
            return self._modules[name]
        self._modules[name] = None
        found = self._file_of(name)
        if found is None:
            return None
        path, is_package = found
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            return None
        parsed = _Module(name=name, path=path, is_package=is_package)
        for node in tree.body:
            self._read_statement(parsed, node)
        self._modules[name] = parsed
        return parsed

    def _absolute(self, module: _Module, level: int, target: Optional[str]) -> Optional[str]:
        if level == 0:
            return target
        parts = module.name.split(".")
        if not module.is_package:
            parts = parts[:-1]
        if level - 1 > len(parts) - 1:
            return None
        base = parts[: len(parts) - (level - 1)]
        return ".".join(base + ([target] if target else []))

    def _read_statement(self, module: _Module, node: ast.stmt) -> None:
        if isinstance(node, ast.ClassDef):
            module.classes[node.name] = node
        elif isinstance(node, ast.ImportFrom):
            source = self._absolute(module, node.level, node.module)
            if source is None:
                return
            for alias in node.names:
                if alias.name != "*":
                    module.imported_names[alias.asname or alias.name] = (source, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    module.imported_modules[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    module.imported_modules[top] = top
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]  # type: ignore[list-item]
            )
            if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                return
            values = _string_list(node.value) if node.value is not None else None
            if values is None:
                return
            if isinstance(node, ast.AugAssign):
                module.exports = (module.exports or []) + values
            else:
                module.exports = values

    def _is_local(self, module: str) -> bool:
        return module == self.import_name or module.startswith(self.import_name + ".")

    def kind_of_name(self, module: _Module, name: str, depth: int) -> Optional[str]:
        """The kind of the class `name` means in `module`."""
        if depth > _MAX_DEPTH:
            return None
        if name in module.classes:
            return self._kind_of_class(module, module.classes[name], depth + 1)
        if name in module.imported_names:
            source, original = module.imported_names[name]
            return self._kind_from_module(source, original, depth + 1)
        return None

    def _kind_from_module(self, source: str, name: str, depth: int) -> Optional[str]:
        if source == "verifiers" or source.startswith("verifiers."):
            return kind_of_verifiers_name(name)
        if not self._is_local(source):
            return None
        target = self.module(source)
        if target is None:
            # `from pkg import sub` names a submodule, not a class.
            return None
        return self.kind_of_name(target, name, depth)

    def _kind_of_class(self, module: _Module, node: ast.ClassDef, depth: int) -> Optional[str]:
        for base in node.bases:
            kind = self._kind_of_base(module, base, depth)
            if kind is not None:
                return kind
        return None

    def _kind_of_base(self, module: _Module, base: ast.expr, depth: int) -> Optional[str]:
        if isinstance(base, ast.Subscript):  # vf.Env[KuhnPokerEnvConfig]
            base = base.value
        if isinstance(base, ast.Name):
            return self.kind_of_name(module, base.id, depth)
        if isinstance(base, ast.Attribute):
            # vf.Taskset, or verifiers.v1.Taskset after `import verifiers.v1`.
            chain: List[str] = []
            current: ast.expr = base
            while isinstance(current, ast.Attribute):
                chain.append(current.attr)
                current = current.value
            if not isinstance(current, ast.Name):
                return None
            chain.reverse()
            if current.id in module.imported_modules:
                prefix = module.imported_modules[current.id]
            elif current.id in module.imported_names:
                source, original = module.imported_names[current.id]
                prefix = f"{source}.{original}"
            else:
                return None
            qualified = ".".join([prefix, *chain[:-1]])
            return self._kind_from_module(qualified, chain[-1], depth + 1)
        return None


def import_name_of(distribution_name: str) -> str:
    """The module verifiers imports for a package, as the Hub derives it."""
    return distribution_name.replace("-", "_").lower()


@dataclass
class DetectedComponents:
    components: List[Dict[str, str]]
    # Kinds with more than one export, left out because a release declares at
    # most one of each (judges excepted).
    ambiguous: Dict[str, List[str]] = field(default_factory=dict)
    found_exports: bool = False


def detect_components(package_dir: Path, distribution_name: str) -> DetectedComponents:
    """The components the package's `__all__` exports, as the Hub's `components`."""
    package = _Package(package_dir, import_name_of(distribution_name))
    module = package.module(package.import_name)
    if module is None or not module.exports:
        return DetectedComponents(components=[])

    by_kind: Dict[str, List[str]] = {}
    for name in dict.fromkeys(module.exports):
        kind = package.kind_of_name(module, name, 0)
        if kind is not None:
            by_kind.setdefault(kind, []).append(name)

    components: List[Dict[str, str]] = []
    ambiguous: Dict[str, List[str]] = {}
    for kind in KIND_ORDER:
        names = by_kind.get(kind, [])
        if len(names) > 1 and kind != "judge":
            ambiguous[kind] = names
            continue
        components.extend({"kind": kind, "name": name} for name in names)
    return DetectedComponents(components=components, ambiguous=ambiguous, found_exports=True)
