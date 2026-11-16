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

"""Exercise stack-limit restoration without creating a CUDA context."""

from test.ci_system.ci_register import register_cuda_ci

import pytest

from tokenspeed.runtime.distributed import cuda_stack

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class StackDriver:
    def __init__(self, limit):
        self.limit = limit
        self.get_error = 0
        self.set_error = 0

    def cuCtxGetLimit(self, pointer, kind):
        pointer._obj.value = self.limit
        return self.get_error

    def cuCtxSetLimit(self, kind, limit):
        if not self.set_error:
            self.limit = limit
        return self.set_error


@pytest.mark.parametrize("limit", [0, 2048])
def test_stack_limit_is_restored_across_sleep_cycles(monkeypatch, limit):
    driver = StackDriver(limit)
    monkeypatch.setattr(cuda_stack, "_driver", lambda: driver)
    for _ in range(2):
        saved = cuda_stack.trim_cuda_stack()
        assert saved == limit
        assert driver.limit == 0
        cuda_stack.restore_cuda_stack(saved)
        assert driver.limit == limit


@pytest.mark.parametrize("operation", ["read", "trim", "restore"])
def test_stack_driver_failure_propagates(monkeypatch, operation):
    driver = StackDriver(2048)
    monkeypatch.setattr(cuda_stack, "_driver", lambda: driver)
    if operation == "read":
        driver.get_error = 201
    else:
        driver.set_error = 201
    with pytest.raises(RuntimeError, match="CUresult 201"):
        if operation == "restore":
            cuda_stack.restore_cuda_stack(2048)
        else:
            cuda_stack.trim_cuda_stack()
    assert driver.limit == 2048
