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

from contextlib import nullcontext
from contextvars import Context

import pytest
from tokenspeed_kernel.ops.tuning import (
    autotune,
    is_autotuning,
    is_warming_up,
    kernel_warmup,
)


@pytest.mark.parametrize("enabled", [False, True])
def test_warmup_does_not_enable_flashinfer_profiling(enabled):
    assert not is_warming_up()
    with autotune(tune_mode=False, tuning_buckets=None, round_up=None):
        with kernel_warmup(enabled=enabled):
            assert is_warming_up() is enabled
            assert not is_autotuning()
        assert not is_warming_up()


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("inner", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_nested_warmup_restores_enclosing_state(outer, inner, failure):
    assert not is_warming_up()
    with kernel_warmup(enabled=outer):
        expected = (
            pytest.raises(RuntimeError, match="warmup failed")
            if failure
            else nullcontext()
        )
        with expected:
            with kernel_warmup(enabled=inner):
                assert is_warming_up() is inner
                if failure:
                    raise RuntimeError("warmup failed")
        assert is_warming_up() is outer
    assert not is_warming_up()


def test_failed_warmup_does_not_leak_into_serving():
    with pytest.raises(RuntimeError, match="warmup failed"):
        with kernel_warmup(enabled=True):
            assert is_warming_up()
            raise RuntimeError("warmup failed")
    assert not is_warming_up()


def test_warmup_is_local_to_the_execution_context():
    independent = Context()
    with kernel_warmup(enabled=True):
        assert is_warming_up()
        assert not independent.run(is_warming_up)
        with kernel_warmup(enabled=False):
            assert not is_warming_up()
        assert is_warming_up()
    assert not independent.run(is_warming_up)
