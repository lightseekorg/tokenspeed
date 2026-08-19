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

"""CPU-only guard that the package imports on the Python it claims to support.

``python/pyproject.toml`` declares a minimum Python, but CI only exercises
3.12. A ``typing`` name introduced after that floor therefore imports cleanly
in CI while raising ``ImportError`` for users on the declared minimum. The
convention here is to take such names from ``typing_extensions``, which
backports them; this test pins that convention.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "python" / "tokenspeed"
_PYPROJECT = _REPO_ROOT / "python" / "pyproject.toml"

# Names added to ``typing`` after 3.10, keyed by the version that shipped
# them. Derived from ``typing.__all__`` on real interpreters; regenerate with
#
#   python3.X -c "import typing; print(sorted(typing.__all__))"
#
# for each version and diff consecutive releases.
_TYPING_ADDED_IN = {
    (3, 11): frozenset(
        {
            "LiteralString",
            "Never",
            "NotRequired",
            "Required",
            "Self",
            "TypeVarTuple",
            "Unpack",
            "assert_never",
            "assert_type",
            "clear_overloads",
            "dataclass_transform",
            "get_overloads",
            "reveal_type",
        }
    ),
    (3, 12): frozenset({"TypeAliasType", "override"}),
    (3, 13): frozenset(
        {
            "NoDefault",
            "ReadOnly",
            "TypeIs",
            "get_protocol_members",
            "is_protocol",
        }
    ),
    (3, 14): frozenset({"evaluate_forward_ref"}),
}

_NEWEST_KNOWN = max(_TYPING_ADDED_IN)


def _post_floor_names(floor: tuple[int, int]) -> dict[str, tuple[int, int]]:
    """Map each ``typing`` name unavailable at *floor* to the version adding it."""
    return {
        name: version
        for version, names in _TYPING_ADDED_IN.items()
        if version > floor
        for name in names
    }


def _declared_floor() -> tuple[int, int]:
    text = _PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*"[^"]*?(\d+)\.(\d+)"', text)
    assert match is not None, "requires-python missing from python/pyproject.toml"
    return int(match.group(1)), int(match.group(2))


def _typing_module_aliases(tree: ast.AST) -> set[str]:
    """Names bound to the ``typing`` module by a plain ``import typing[ as x]``."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "typing":
                    aliases.add(alias.asname or "typing")
    return aliases


def _referenced_typing_names(tree: ast.AST) -> list[tuple[int, str]]:
    """Every ``typing`` name the module reaches for, however it is spelled."""
    aliases = _typing_module_aliases(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            found.append((node.lineno, node.attr))
    return found


def test_typing_imports_respect_declared_python_floor() -> None:
    floor = _declared_floor()
    post_floor = _post_floor_names(floor)
    offenders: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, name in _referenced_typing_names(tree):
            added = post_floor.get(name)
            if added is not None:
                offenders.append(
                    f"{path.relative_to(_REPO_ROOT)}:{lineno}: "
                    f"typing.{name} requires Python {added[0]}.{added[1]}"
                )
    assert not offenders, (
        f"these need Python > {floor[0]}.{floor[1]}; take them from "
        "typing_extensions instead:\n" + "\n".join(offenders)
    )


def test_typing_table_covers_the_running_interpreter() -> None:
    """A table older than the interpreter would silently under-enforce."""
    running = sys.version_info[:2]
    assert running <= _NEWEST_KNOWN, (
        f"_TYPING_ADDED_IN stops at {_NEWEST_KNOWN[0]}.{_NEWEST_KNOWN[1]} but this "
        f"is Python {running[0]}.{running[1]}; names added since are unchecked. "
        "Regenerate the table as described above and add the missing entries."
    )
