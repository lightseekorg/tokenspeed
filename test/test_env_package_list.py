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

"""Tests for tokenspeed env dependency reporting."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from tokenspeed.env import PACKAGE_LIST

ROOT = Path(__file__).resolve().parents[1]
KERNEL_REQUIREMENTS = (
    "common.txt",
    "cuda.txt",
    "cuda-thirdparty.txt",
    "rocm.txt",
    "rocm-thirdparty.txt",
)
BUILD_ONLY_PACKAGES = {
    "wheel",
}


def _dependency_name(requirement: str) -> str:
    return re.split(r"\s*(?:==|>=|<=|~=|!=|>|<|;|\[)", requirement, maxsplit=1)[0]


def _read_requirements(path: Path, seen: set[Path] | None = None) -> list[str]:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)

    requirements = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line.startswith("-r ") or line.startswith("--requirement "):
            include = line.split(maxsplit=1)[1]
            requirements.extend(_read_requirements(path.parent / include, seen))
            continue
        requirements.append(line)
    return requirements


def test_env_package_list_matches_pyproject_dependencies() -> None:
    pyproject = tomllib.loads(
        (ROOT / "python" / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = pyproject["project"]["dependencies"]

    kernel_requirements_dir = ROOT / "tokenspeed-kernel" / "python" / "requirements"
    kernel_dependencies = []
    for requirement_file in KERNEL_REQUIREMENTS:
        kernel_dependencies.extend(
            _read_requirements(kernel_requirements_dir / requirement_file)
        )

    expected = {
        "tokenspeed",
        "tokenspeed-kernel",
        *map(_dependency_name, dependencies),
        *(
            dependency
            for dependency in map(_dependency_name, kernel_dependencies)
            if dependency not in BUILD_ONLY_PACKAGES
        ),
    }

    assert set(PACKAGE_LIST) == expected
