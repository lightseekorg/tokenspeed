"""Select the runtime test files that a pull request's changes can reach.

CI lists a pull request's changed paths in ``TOKENSPEED_CI_CHANGED_PATHS``.
When every change is Markdown, a test file, or a model module (a module
directly under the models package that defines ``EntryClass``), only these
runtime test files run: the changed ones; for a changed model module, the
tests that import it or a module built on it, or that name it or such a module
(``gpt_oss`` matches ``openai/gpt-oss-20b``); and the tests that import any of
these, directly or through other test files. Any other change runs every test.
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


def imported_modules(path: Path) -> set[str]:
    """Modules that any import statement in ``path`` names, including imports
    inside functions; a name after ``from ... import`` counts as a submodule."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def model_names(modules: set[str]) -> set[str]:
    """Modules directly under the models package (by name) among ``modules``."""
    prefix = f"{MODEL_PACKAGE}."
    return {
        module[len(prefix) :].split(".")[0]
        for module in modules
        if module.startswith(prefix)
    }


def test_module_names(test: str) -> set[str]:
    """Names that import a test file: test.runtime.<name>, runtime.<name> or
    <name>."""
    dotted = test.removesuffix(".py").replace("/", ".")
    return {dotted, dotted.removeprefix("test."), Path(test).stem}


def squashed(text: str) -> str:
    """``text`` lowercased without ``_``, ``-`` and ``.``, so that the module
    ``qwen3_5`` matches the checkpoint ``Qwen3.5-35B``."""
    return re.sub(r"[_.-]", "", text.lower())


def model_modules(repo_root: Path) -> dict[str, Path]:
    """The model modules by name: those directly under the models package
    that define ``EntryClass``."""
    return {
        path.stem: path
        for path in sorted((repo_root / MODEL_DIRECTORY).glob("*.py"))
        if re.search(r"^EntryClass\s*=", path.read_text(encoding="utf-8"), re.M)
    }


def models_built_on(changed: set[str], repo_root: Path) -> set[str]:
    """``changed`` and every module directly under the models package that
    imports one of them, directly or through other such modules."""
    importers: dict[str, set[str]] = {}
    for path in sorted((repo_root / MODEL_DIRECTORY).glob("*.py")):
        for imported in model_names(imported_modules(path)):
            importers.setdefault(imported, set()).add(path.stem)
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
    names = {squashed(name) for name in reached}
    # Every runtime test file, so that an unselected one still links others.
    sources = {
        path.relative_to(repo_root).as_posix(): path
        for path in sorted((repo_root / RUNTIME_TEST_DIRECTORY).rglob("*.py"))
    }
    imports = {test: imported_modules(path) for test, path in sources.items()}
    affected = {
        test
        for test, path in sources.items()
        if test in changed_tests
        or model_names(imports[test]) & reached
        or any(name in squashed(path.read_text(encoding="utf-8")) for name in names)
    }
    # A test that imports an affected test file, or a deleted one, is affected too.
    pending = list(affected | changed_tests)
    while pending:
        modules = test_module_names(pending.pop())
        for test in sources:
            if test not in affected and imports[test] & modules:
                affected.add(test)
                pending.append(test)
    return [test for test in test_files if test in affected]
