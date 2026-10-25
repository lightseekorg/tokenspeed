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

"""The Python half of ``ts-control``'s behavior fixtures.

``tokenspeed-engine/crates/ts-control/tests/fixtures/`` records how the Python
engine resolves inputs (sampling parameters off the msgpack wire); the Rust
crate must reproduce every case. This test fails when the Python behavior
changes without the fixtures, and so the Rust port, following it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "tokenspeed-engine"
    / "crates"
    / "ts-control"
    / "tests"
    / "fixtures"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "ts_control_fixture_generator", _FIXTURE_DIR / "generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXPECTED = _load_generator().build_fixtures()


@pytest.mark.parametrize("file_name", sorted(_EXPECTED))
def test_fixture_matches_python_behavior(file_name: str) -> None:
    on_disk = (_FIXTURE_DIR / file_name).read_text()
    assert on_disk == _EXPECTED[file_name], (
        f"{file_name} no longer matches the Python engine. If the behavior "
        "changed on purpose, run "
        "tokenspeed-engine/crates/ts-control/tests/fixtures/generate.py and "
        "update ts-control to match."
    )


def test_no_stale_fixture_files() -> None:
    on_disk = {
        path.name
        for pattern in ("*.hex", "*.json")
        for path in _FIXTURE_DIR.glob(pattern)
    }
    assert on_disk == set(_EXPECTED)
