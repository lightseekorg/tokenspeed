# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU-only contracts for the shared AMD scheduling library."""

import hashlib
import importlib.util
import tomllib
from pathlib import Path

import pytest

_AMD = Path(__file__).resolve().parents[3] / "tokenspeed-kernel-amd"
_PACKAGE = "tokenspeed_kernel_amd"
_DIRECTORY = _AMD / "python" / _PACKAGE


@pytest.fixture
def schedule():
    pytest.importorskip(
        "tokenspeed_kernel_amd", reason="AMD kernel package is optional"
    )
    spec = importlib.util.spec_from_file_location(
        "amd_schedule_policy_test", _DIRECTORY / "_scheduling.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scheduler_library_is_package_data(schedule):
    path = Path(schedule._SCHED_LIBRARY_PATH)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.name == "sched_barrier.ll"
    assert digest == schedule._scheduler_library_hash()
    config = tomllib.loads((_AMD / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"][_PACKAGE]
    assert any(path.match(pattern) for pattern in patterns)
    text = path.read_text()
    assert f"define i32 @{schedule._SCHED_SYMBOL}() alwaysinline" in text
    assert schedule._SCHED_LIBRARY_NAME in schedule._SCHED_SYMBOL


def test_normal_compile_options_pin_library_path(schedule):
    assert schedule.sched_barrier_compile_options() == {
        "SCHED_LIBRARY_HASH": hashlib.sha256(
            Path(schedule._SCHED_LIBRARY_PATH).read_bytes()
        ).hexdigest(),
        "extern_libs": {schedule._SCHED_LIBRARY_NAME: schedule._SCHED_LIBRARY_PATH},
    }


def test_changed_library_cannot_reuse_old_content_key(schedule, monkeypatch, tmp_path):
    path = tmp_path / Path(schedule._SCHED_LIBRARY_PATH).name
    path.write_bytes(Path(schedule._SCHED_LIBRARY_PATH).read_bytes())
    monkeypatch.setattr(schedule, "_SCHED_LIBRARY_PATH", str(path))
    before = schedule.sched_barrier_compile_options()
    path.write_bytes(path.read_bytes() + b"; changed contents, same path\n")
    # Simulate the fresh process required after editing JIT/library source.
    schedule._scheduler_library_hash.cache_clear()
    after = schedule.sched_barrier_compile_options()
    assert before["extern_libs"] == after["extern_libs"]
    assert before["SCHED_LIBRARY_HASH"] != after["SCHED_LIBRARY_HASH"]
    assert after["SCHED_LIBRARY_HASH"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_compile_options_are_not_shared_mutable_state(schedule):
    options = schedule.sched_barrier_compile_options()
    options["extern_libs"].clear()
    options["SCHED_LIBRARY_HASH"] = None
    again = schedule.sched_barrier_compile_options()
    assert again["SCHED_LIBRARY_HASH"] == schedule._scheduler_library_hash()
    assert again["extern_libs"] == {
        schedule._SCHED_LIBRARY_NAME: schedule._SCHED_LIBRARY_PATH
    }
