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

"""Builtin kernels remain importable when CUDA third-party packages are missing."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_KERNEL_ROOT = Path(__file__).resolve().parents[2]


def _native_modules():
    requirements = _KERNEL_ROOT / "python/requirements/cuda-thirdparty.txt"
    # Distribution names need not match the import names they provide.
    owners = importlib.metadata.packages_distributions()
    modules = {}
    for line in requirements.read_text().splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        name = Requirement(line).name
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        modules[name] = {
            root
            for root, distributions in owners.items()
            if canonicalize_name(name)
            in {canonicalize_name(distribution) for distribution in distributions}
        }
    return modules


# Discovery belongs to pytest collection, not each probe subprocess.
_NATIVE_MODULES = _native_modules() if __name__ != "__main__" else {}


def _exclusion_cases(names):
    # Baseline, each missing package with peers present, and all missing also
    # cover every two-package state. Keep O(n) fresh interpreters for CI cost.
    cases = {(), tuple(names)}
    cases.update((name,) for name in names)
    return [
        pytest.param(
            case, id="all-present" if not case else f"without-{','.join(case)}"
        )
        for case in sorted(cases, key=lambda case: (len(case), case))
    ]


@pytest.mark.parametrize("excluded", _exclusion_cases(tuple(_NATIVE_MODULES)))
def test_optional_imports(excluded):
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_nvidia:
        pytest.skip("CUDA provider imports require an NVIDIA platform")

    unknown = [name for name in excluded if not _NATIVE_MODULES[name]]
    if unknown:
        pytest.skip(
            f"Installed packages supply no import ownership metadata: {unknown}"
        )
    missing = set().union(*(_NATIVE_MODULES[name] for name in excluded))
    # A fresh interpreter isolates cached imports and native initialization.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *sorted(missing)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout


if __name__ == "__main__":
    sys.path.insert(0, str(_KERNEL_ROOT / "python"))
    # None blocks the root and its submodules; dependent packages stay visible.
    for name in sys.argv[1:]:
        sys.modules[name] = None

    from tokenspeed_kernel.registry import load_builtin_kernels

    load_builtin_kernels()
