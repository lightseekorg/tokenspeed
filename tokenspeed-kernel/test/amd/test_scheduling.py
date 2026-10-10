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

import importlib.util
import re
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
    assert path.name == "sched_barrier.ll"
    config = tomllib.loads((_AMD / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"][_PACKAGE]
    assert any(path.match(pattern) for pattern in patterns)
    # Triton links the library only for calls whose symbol contains its name.
    symbols = set(re.findall(r"^define \S+ @(\w+)", path.read_text(), re.M))
    hints = symbols - {schedule._READFIRSTLANE_SYMBOL}
    assert hints and all(schedule._SCHED_LIBRARY_NAME in s for s in hints)


def test_changed_library_changes_compile_key(schedule, tmp_path):
    compiler = pytest.importorskip("tokenspeed_triton.backends.amd.compiler")
    path = tmp_path / Path(schedule._SCHED_LIBRARY_PATH).name
    path.write_bytes(Path(schedule._SCHED_LIBRARY_PATH).read_bytes())
    options = compiler.HIPOptions(
        arch="gfx950", extern_libs={schedule._SCHED_LIBRARY_NAME: str(path)}
    )
    try:
        before = options.hash()
        path.write_bytes(path.read_bytes() + b"; changed contents, same path\n")
        # Triton memoizes file hashes per process; start cold as a new process.
        compiler.file_hash.cache_clear()
        assert options.hash() != before
    finally:
        compiler.file_hash.cache_clear()


def test_compile_options_are_not_shared_mutable_state(schedule):
    options = schedule.sched_compile_options()
    assert options == {
        "extern_libs": {schedule._SCHED_LIBRARY_NAME: schedule._SCHED_LIBRARY_PATH}
    }
    options["extern_libs"].clear()
    assert schedule.sched_compile_options() == {
        "extern_libs": {schedule._SCHED_LIBRARY_NAME: schedule._SCHED_LIBRARY_PATH}
    }


def test_sched_group_mask_ors_named_classes(schedule):
    assert schedule._sched_group_mask("mfma") == 0x8
    assert schedule._sched_group_mask(("valu", "trans")) == 0x402
    with pytest.raises(ValueError, match="unknown sched_group class"):
        schedule._sched_group_mask(("mfma", "lds"))
