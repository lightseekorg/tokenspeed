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

"""The stack-limit window releases the local memory its kernels reserved."""

from __future__ import annotations

import pytest
import torch
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

from cuda.bindings import driver  # noqa: E402
from tokenspeed_kernel._triton import tl, triton  # noqa: E402
from tokenspeed_kernel.platform import current_platform  # noqa: E402

_STACK = driver.CUlimit.CU_LIMIT_STACK_SIZE


@triton.jit
def _center_rows(x_ptr, out_ptr, ROWS: tl.constexpr, COLS: tl.constexpr):
    offsets = tl.arange(0, ROWS)[:, None] * COLS + tl.arange(0, COLS)[None, :]
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, x - tl.sum(x, 1)[:, None])


def _stack_limit() -> int:
    error, limit = driver.cuCtxGetLimit(_STACK)
    assert error == driver.CUresult.CUDA_SUCCESS
    return limit


@pytest.fixture(autouse=True)
def _later_tests_keep_the_limit():
    # A runtime call makes the primary context current before the driver reads it.
    torch.cuda.synchronize()
    limit = _stack_limit()
    yield
    (error,) = driver.cuCtxSetLimit(_STACK, limit)
    assert error == driver.CUresult.CUDA_SUCCESS


def test_the_window_lowers_the_limit_its_kernels_raised() -> None:
    x = torch.randn(128, 256, device="cuda")
    out = torch.empty_like(x)
    # Not the driver default, so restoring it differs from resetting the limit.
    found = 1536
    (error,) = driver.cuCtxSetLimit(_STACK, found)
    assert error == driver.CUresult.CUDA_SUCCESS
    with current_platform().restore_stack_limit():
        # One warp holds every row, so each thread spills about 9 KiB.
        kernel = _center_rows[(1,)](x, out, ROWS=128, COLS=256, num_warps=1)
        torch.cuda.synchronize()
        raised, free_raised = _stack_limit(), torch.cuda.mem_get_info()[0]
    assert 4 * kernel.n_spills > found and raised > found
    assert _stack_limit() == found
    assert torch.cuda.mem_get_info()[0] > free_raised
    torch.testing.assert_close(out, x - x.sum(1, keepdim=True))
    # The next launch raises the limit again.
    out.zero_()
    _center_rows[(1,)](x, out, ROWS=128, COLS=256, num_warps=1)
    torch.cuda.synchronize()
    assert _stack_limit() == raised
    torch.testing.assert_close(out, x - x.sum(1, keepdim=True))
