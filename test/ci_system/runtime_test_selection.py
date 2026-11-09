"""Select the runtime test files that a pull request's changes can reach.

CI lists a pull request's changed paths in ``TOKENSPEED_CI_CHANGED_PATHS``.
When every change is Markdown, a test file, or a model module (a module
directly under the models package that defines ``EntryClass``), only these
runtime test files run: the changed ones, and those that import a changed
model module or a model module built on one. Any other change runs every test.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

CHANGED_PATHS_ENV = "TOKENSPEED_CI_CHANGED_PATHS"
MODEL_DIRECTORY = "python/tokenspeed/runtime/models"
MODEL_PACKAGE = "tokenspeed.runtime.models"
RUNTIME_TEST_DIRECTORY = "test/runtime"
KERNEL_TEST_DIRECTORY = "tokenspeed-kernel/test"


def imported_models(path: Path) -> set[str]:
    """Model modules (by name) that any import statement in ``path`` names."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    prefix = f"{MODEL_PACKAGE}."
    return {
        module[len(prefix) :].split(".")[0]
        for module in modules
        if module.startswith(prefix)
    }


def model_modules(repo_root: Path) -> dict[str, Path]:
    """The model modules by name: those directly under the models package
    that define ``EntryClass``."""
    return {
        path.stem: path
        for path in sorted((repo_root / MODEL_DIRECTORY).glob("*.py"))
        if re.search(r"^EntryClass\s*=", path.read_text(encoding="utf-8"), re.M)
    }


def models_built_on(changed: set[str], repo_root: Path) -> set[str]:
    """``changed`` and every model module that imports one of them, directly
    or through other model modules."""
    models = model_modules(repo_root)
    importers: dict[str, set[str]] = {}
    for name, path in models.items():
        for imported in imported_models(path):
            importers.setdefault(imported, set()).add(name)
    reached, pending = set(), list(changed)
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        pending.extend(importers.get(name, set()))
    return reached


def select_runtime_tests(
    test_files: list[str], changed_paths: list[str], repo_root: Path
) -> list[str] | None:
    """Return the repository-relative ``test_files`` that the changes can reach,
    or None when a change can reach any of them."""
    models = model_modules(repo_root)
    changed_tests: set[str] = set()
    changed_models: set[str] = set()
    for path in changed_paths:
        name = Path(path).name
        if path.endswith(".md"):
            continue
        if path.startswith(f"{KERNEL_TEST_DIRECTORY}/") and name.startswith("test_"):
            continue
        if path.startswith(f"{RUNTIME_TEST_DIRECTORY}/") and name.startswith("test_"):
            changed_tests.add(path)
            continue
        stem = name.removesuffix(".py")
        if path == f"{MODEL_DIRECTORY}/{name}" and stem in models:
            changed_models.add(stem)
            continue
        return None
    reached = models_built_on(changed_models, repo_root) if changed_models else set()
    return [
        test
        for test in test_files
        if test in changed_tests
        or (reached and imported_models(repo_root / test) & reached)
    ]
