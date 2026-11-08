"""Find the runtime files that a set of Python files can import."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

RUNTIME_PACKAGE = "tokenspeed"
RUNTIME_SOURCE_DIRECTORY = "python"


def module_paths(module: str) -> tuple[str, str]:
    """Return the repo-relative file and package paths that can hold ``module``."""
    base = f"{RUNTIME_SOURCE_DIRECTORY}/{module.replace('.', '/')}"
    return f"{base}.py", f"{base}/__init__.py"


def is_runtime_module(module: str) -> bool:
    return module == RUNTIME_PACKAGE or module.startswith(f"{RUNTIME_PACKAGE}.")


def runtime_imports(path: Path) -> set[str]:
    """Runtime modules that any import statement in ``path`` names, including
    imports inside functions. A name after ``from ... import`` counts too, since
    it may be a submodule."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {module for module in modules if is_runtime_module(module)}


def reachable_runtime_paths(repo_root: Path, roots: Iterable[str]) -> frozenset[str]:
    """Return the repo-relative runtime paths that ``roots`` can import.

    Args:
        repo_root: Checkout of the tested revision.
        roots: Repo-relative Python files or directories whose imports start
            the search.

    Returns:
        Both candidate paths of every runtime module reached, whether or not
        the file exists, so a deleted module that is still imported counts.
        Importing a module also runs its parent packages' ``__init__.py``.
        Modules imported by name at run time (``importlib``) are not followed.
    """
    pending = set()
    for root in roots:
        path = repo_root / root
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            pending |= runtime_imports(file)
    seen: set[str] = set()
    while pending:
        module = pending.pop()
        parts = module.split(".")
        for depth in range(1, len(parts) + 1):
            parent = ".".join(parts[:depth])
            if parent in seen:
                continue
            seen.add(parent)
            for candidate in module_paths(parent):
                if (repo_root / candidate).is_file():
                    pending |= runtime_imports(repo_root / candidate)
    return frozenset(path for module in seen for path in module_paths(module))
