# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Select the kernel test files that a pull request's changes name.

On pull requests CI lists the changed repository paths in
``TOKENSPEED_CI_CHANGED_PATHS``, one per line. Kernel sources and test files
both map to op families by name: ``ops/<family>/`` (and an attention variant
directory such as ``ops/attention/mla/``), ``thirdparty/<package>/``, the
``numerics/torch/<family>.py`` references, ``benchmark/``, and the AMD
``ops/<arch>/<family>/``. A test file belongs to the families it imports and to
its ``ops/<family>/`` directory. A changed third-party package also reaches the
families whose sources import it.

A test file runs when it changed, when it shares a family with a change, or
when it calls ``load_builtin_kernels``. Every test file runs when a change
names no family: core modules such as the registry and platform, ``conftest.py``,
packaging and requirements, and anything outside the kernel trees other than
Markdown and the runtime. Dependencies between families are not followed;
pushes to ``main`` run every test.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

CHANGED_PATHS_ENV = "TOKENSPEED_CI_CHANGED_PATHS"
TEST_DIRECTORY = "tokenspeed-kernel/test"
PACKAGE_DIRECTORIES = {
    "tokenspeed_kernel": "tokenspeed-kernel/python/tokenspeed_kernel",
    "tokenspeed_kernel_amd": "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd",
}
# Paths whose changes no kernel test can see.
IGNORED_PREFIXES = ("python/tokenspeed/", "test/runtime/")


def _imported_modules(path: Path) -> set[str]:
    """Modules named by any import statement in ``path``, including imports
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


class KernelTestSelection:
    """Which test files a set of changed paths names.

    Args:
        repo_root: Checkout of the tested revision.
        changed_paths: Repository-relative paths the pull request changes.
    """

    def __init__(self, repo_root: Path, changed_paths: list[str]) -> None:
        self.repo_root = repo_root.resolve()
        self.test_root = self.repo_root / TEST_DIRECTORY
        self.changed_tests: set[str] = set()
        self.families: set[str] = set()
        self.run_all = False
        self._thirdparty_users: dict[str, set[str]] | None = None
        # Test helper modules, which tests import by their bare names.
        self.helpers: dict[str, list[Path]] = {}
        for path in sorted(self.test_root.rglob("*.py")):
            if not path.name.startswith("test_") and path.name not in (
                "conftest.py",
                "__init__.py",
            ):
                self.helpers.setdefault(path.stem, []).append(path)
        for path in changed_paths:
            self._add_change(path)

    def module_family(self, module: str) -> str | None:
        """The family a kernel module belongs to, or None for core modules."""
        parts = module.split(".")
        package = PACKAGE_DIRECTORIES.get(parts[0])
        if package is None or len(parts) < 3:
            return None
        if parts[0] == "tokenspeed_kernel_amd":
            # Private modules beside the families (``ops/gfx950/_xcd.py``) are
            # shared by all of them.
            if (
                parts[1] != "ops"
                or not parts[2].startswith("gfx")
                or len(parts) < 4
                or parts[3].startswith("_")
            ):
                return None
            return parts[3]
        if parts[1] == "ops":
            family = parts[2]
            variant = parts[3] if len(parts) > 3 else ""
            variant_dir = self.repo_root / package / "ops" / family / variant
            if variant and not variant.startswith("_") and variant_dir.is_dir():
                return f"{family}/{variant}"
            return family
        if parts[1] == "thirdparty":
            return f"thirdparty/{parts[2]}"
        if parts[1] == "numerics" and parts[2] == "torch" and len(parts) > 3:
            return parts[3]
        if parts[1] == "benchmark":
            return "benchmark"
        return None

    def _source_module(self, path: str) -> str | None:
        """The module a kernel source path defines, or None outside them."""
        for package, directory in PACKAGE_DIRECTORIES.items():
            if not path.startswith(f"{directory}/"):
                continue
            relative = Path(path[len(directory) + 1 :])
            parts = [package, *relative.parent.parts]
            if relative.name != "__init__.py":
                parts.append(relative.name.split(".", 1)[0])
            return ".".join(parts)
        return None

    def _add_change(self, path: str) -> None:
        path = path.rstrip("/")
        if path.endswith(".md") or path.startswith(IGNORED_PREFIXES):
            return
        if path.startswith(f"{TEST_DIRECTORY}/"):
            name = Path(path).name
            if name.startswith("test_") and name.endswith(".py"):
                self.changed_tests.add(path)
            elif name.startswith("bench_"):
                pass
            elif name.endswith(".py") and name not in ("conftest.py", "__init__.py"):
                self.families.add(f"helper/{Path(path).stem}")
            else:
                self.run_all = True
            return
        module = self._source_module(path)
        family = None if module is None else self.module_family(module)
        # Non-Python sources outside a third-party package (LLVM intrinsics,
        # build files) can affect any kernel; a directory names its family.
        source = path.endswith(".py") or (self.repo_root / path).is_dir()
        if family is None or (not source and not family.startswith("thirdparty/")):
            self.run_all = True
            return
        self.families.add(family)
        if family.startswith("thirdparty/"):
            self.families |= self.thirdparty_users().get(family, set())

    def thirdparty_users(self) -> dict[str, set[str]]:
        """Each third-party package's families: those whose sources import it."""
        if self._thirdparty_users is None:
            users: dict[str, set[str]] = {}
            for package, directory in PACKAGE_DIRECTORIES.items():
                for file in sorted((self.repo_root / directory).rglob("*.py")):
                    path = file.relative_to(self.repo_root).as_posix()
                    family = self.module_family(self._source_module(path))
                    if family is None or family.startswith("thirdparty/"):
                        continue
                    for module in _imported_modules(file):
                        used = self.module_family(module)
                        if used is not None and used.startswith("thirdparty/"):
                            users.setdefault(used, set()).add(family)
            self._thirdparty_users = users
        return self._thirdparty_users

    def _import_families(self, file: Path, seen: set[Path]) -> set[str] | None:
        """The families ``file`` imports, through the test helpers it imports
        too; None when it loads every kernel."""
        seen.add(file)
        modules = _imported_modules(file)
        if any(module.endswith(".load_builtin_kernels") for module in modules):
            return None
        families = {self.module_family(module) for module in modules} - {None}
        for module in modules:
            for helper in self.helpers.get(module, []):
                families.add(f"helper/{module}")
                if helper in seen:
                    continue
                helper_families = self._import_families(helper, seen)
                if helper_families is None:
                    return None
                families |= helper_families
        return families

    def test_families(self, test_file: Path) -> set[str] | None:
        """The families a test file belongs to; None when it loads every kernel."""
        families = self._import_families(test_file.resolve(), set())
        if families is None:
            return None
        parts = test_file.resolve().relative_to(self.test_root).parts[:-1]
        if "ops" in parts and parts.index("ops") + 1 < len(parts):
            families.add(parts[parts.index("ops") + 1])
        return families

    def selects(self, test_file: Path) -> bool:
        if self.run_all:
            return True
        if test_file.resolve().relative_to(self.repo_root).as_posix() in (
            self.changed_tests
        ):
            return True
        families = self.test_families(test_file)
        if families is None:
            return True
        return any(
            changed == family
            or changed.startswith(f"{family}/")
            or family.startswith(f"{changed}/")
            for changed in self.families
            for family in families
        )


def selection_from_env(repo_root: Path) -> KernelTestSelection | None:
    """The selection CI requested for this run, or None to run every test."""
    paths = [
        line.strip()
        for line in os.environ.get(CHANGED_PATHS_ENV, "").splitlines()
        if line.strip()
    ]
    return KernelTestSelection(repo_root, paths) if paths else None
