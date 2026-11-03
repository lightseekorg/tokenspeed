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

"""CuTe DSL compilations and direct ``@cute.jit`` calls reach the monitor."""

import logging

import pytest
import tokenspeed_kernel.compile_monitor as compile_monitor
import torch
from tokenspeed_kernel.compile_monitor import (
    ServingJitError,
    UnboundedSpecializationError,
)
from utils import assert_no_cute_compile

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")

from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import from_dlpack

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an NVIDIA GPU"
)


@cute.kernel
def _add_kernel(x: cute.Tensor, value: cutlass.Float32):
    tidx, _, _ = cute.arch.thread_idx()
    x[tidx] = x[tidx] + value


@cute.jit
def _add(x: cute.Tensor, value: cutlass.Float32, threads: cutlass.Constexpr):
    _add_kernel(x, value).launch(grid=(1, 1, 1), block=(threads, 1, 1))


class _AddRows:
    """A kernel object whose attribute the trace bakes in."""

    def __init__(self, rows: int) -> None:
        self.rows = rows

    @cute.jit
    def __call__(self, x: cute.Tensor, value: cutlass.Float32):
        _add_kernel(x, value).launch(grid=(1, 1, 1), block=(self.rows, 1, 1))


@pytest.fixture(autouse=True)
def startup(monkeypatch):
    # mark_serving() closes the process-wide compile switch; reopen it afterwards.
    monkeypatch.setattr(compile_monitor, "_serving", False)


@pytest.fixture
def installed():
    compile_monitor.install_compile_monitor("warn")
    # The pinned cutlass-dsl must keep the JIT entry the hooks read; an
    # upgrade that changes it fails here instead of silently going unmonitored.
    assert compile_monitor._cute_dsl_hooks is not None
    yield
    compile_monitor.uninstall_compile_monitor()


def _x(size=64):
    return torch.zeros(size, device="cuda")


def _messages(caplog, level):
    return [r.message for r in caplog.records if r.levelno == level]


def test_compiles_while_serving_are_logged_with_their_change(installed, caplog):
    x = _x()
    cute.compile(_add, from_dlpack(x), 1.0, 32)
    compile_monitor.mark_serving()
    with caplog.at_level(logging.INFO, logger=compile_monitor.__name__):
        compiled = cute.compile(_add, from_dlpack(x), 1.0, 64)
    compiled(from_dlpack(x), 3.0)
    torch.cuda.synchronize()
    assert torch.equal(x, torch.full_like(x, 3.0))
    (message,) = _messages(caplog, logging.INFO)
    assert "CuTe DSL JIT compiled " in message
    assert "._add while serving in " in message
    # The Float32 value is a runtime argument; only the Constexpr changed.
    assert "(threads: 32 -> 64) from " in message
    assert "test/nvidia/test_cute_dsl_compile_monitor.py:" in message
    stats = compile_monitor.compile_stats()
    assert (stats.startup_compiles, stats.serving_compiles) == (1, 1)
    assert stats.serving_seconds > 0


def test_constexpr_that_follows_the_batch_is_named(installed, caplog):
    x = _x()
    compile_monitor.mark_serving()
    with caplog.at_level(logging.WARNING, logger=compile_monitor.__name__):
        for threads in (33, 35, 37, 39, 41):
            cute.compile(_add, from_dlpack(x), 1.0, threads)
    (warning,) = _messages(caplog, logging.WARNING)
    assert "compile-time parameter threads has compiled 5 new values" in warning
    assert "(33, 35, 37, 39, 41)" in warning


def test_static_dimension_that_follows_the_batch_is_named(installed, caplog):
    x = _x()
    compile_monitor.mark_serving()
    with caplog.at_level(logging.WARNING, logger=compile_monitor.__name__):
        # A tensor converted without dynamic marks keys on its shape.
        for rows in (33, 35, 37, 39, 41):
            cute.compile(_add, from_dlpack(x[:rows]), 1.0, 32)
    (warning,) = _messages(caplog, logging.WARNING)
    assert "compile-time parameter x.shape[0] has compiled 5 new values" in warning


def test_kernel_object_attribute_that_follows_the_batch_is_named(installed, caplog):
    x = _x()
    compile_monitor.mark_serving()
    with caplog.at_level(logging.WARNING, logger=compile_monitor.__name__):
        for rows in (33, 35, 37, 39, 41):
            cute.compile(_AddRows(rows), from_dlpack(x), 1.0)
    (warning,) = _messages(caplog, logging.WARNING)
    assert "_AddRows.__call__: compile-time parameter self.rows" in warning


def test_cache_keyed_on_a_runtime_value_is_reported(installed, caplog):
    x = _x()
    cute.compile(_add, from_dlpack(x).mark_layout_dynamic(), 1.0, 32)
    compile_monitor.mark_serving()
    with caplog.at_level(logging.WARNING, logger=compile_monitor.__name__):
        # A dynamic layout compiles one specialization for every row count,
        # yet this caller compiles once per row count.
        for rows in (33, 35, 37, 39, 41):
            cute.compile(_add, from_dlpack(x[:rows]).mark_layout_dynamic(), 1.0, 32)
    (warning,) = _messages(caplog, logging.WARNING)
    assert "._add has compiled specializations it already had 5 times" in warning


def test_direct_call_while_serving_is_reported_once(installed, caplog):
    x = _x()
    _add(from_dlpack(x), 1.0, 16)
    compile_monitor.mark_serving()
    with caplog.at_level(logging.INFO, logger=compile_monitor.__name__):
        for _ in range(3):
            # Reuses the binary CuTe DSL cached, but traces every time.
            _add(from_dlpack(x), 1.0, 16)
    torch.cuda.synchronize()
    assert torch.equal(x[:16], torch.full((16,), 4.0, device="cuda"))
    assert not _messages(caplog, logging.INFO)
    (warning,) = _messages(caplog, logging.WARNING)
    assert "._add was called directly as a @cute.jit function while serving" in (
        warning
    )
    assert "test/nvidia/test_cute_dsl_compile_monitor.py:" in warning
    assert compile_monitor.compile_stats().serving_compiles == 0


def test_error_mode_raises_on_a_direct_call():
    compile_monitor.install_compile_monitor("error")
    try:
        x = _x()
        compile_monitor.mark_serving()
        with pytest.raises(ServingJitError, match="called directly"):
            _add(from_dlpack(x), 1.0, 8)
        with pytest.raises(UnboundedSpecializationError, match="threads"):
            for threads in range(3, 64, 2):
                cute.compile(_add, from_dlpack(x), 1.0, threads)
    finally:
        compile_monitor.uninstall_compile_monitor()


def test_uninstall_restores_the_jit_entry():
    generate_mlir = BaseDSL.generate_mlir
    compile_and_cache = BaseDSL.compile_and_cache
    compile_monitor.install_compile_monitor("warn")
    try:
        assert BaseDSL.generate_mlir is not generate_mlir
        assert BaseDSL.compile_and_cache is not compile_and_cache
    finally:
        compile_monitor.uninstall_compile_monitor()
    assert BaseDSL.generate_mlir is generate_mlir
    assert BaseDSL.compile_and_cache is compile_and_cache


def test_assert_no_cute_compile():
    x = _x()
    compiled = cute.compile(_add, from_dlpack(x), 1.0, 32)
    with assert_no_cute_compile():
        compiled(from_dlpack(x), 1.0)
    with pytest.raises(AssertionError, match="_add"):
        with assert_no_cute_compile():
            cute.compile(_add, from_dlpack(x), 1.0, 32)
    with pytest.raises(AssertionError, match="_add"):
        with assert_no_cute_compile():
            _add(from_dlpack(x), 1.0, 32)
