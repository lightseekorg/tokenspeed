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

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.workspace import (  # noqa: E402
    WorkspacePool,
    reset_workspace_pools,
    workspace_pool,
)


def _pool() -> WorkspacePool:
    # CPU keeps these runnable without a device; a tiny initial block keeps
    # the growth-path tests meaningful.
    return WorkspacePool("cpu", initial_nbytes=1024)


def test_allocate_returns_non_overlapping_views():
    pool = _pool()
    first, second = pool.allocate(((4, 8), torch.int32), ((16,), torch.float32))
    assert (first.shape, first.dtype) == ((4, 8), torch.int32)
    assert (second.shape, second.dtype) == ((16,), torch.float32)
    # 4*8*4 = 128 B rounded up to the 256-byte stride keeps the two apart.
    assert second.data_ptr() - first.data_ptr() == 256


def test_separate_allocate_calls_share_the_same_bytes():
    # The reuse is the point: it is what makes the block converge to the
    # largest single request rather than the sum of every caller.
    pool = _pool()
    (first,) = pool.allocate(((64,), torch.uint8))
    (second,) = pool.allocate(((64,), torch.uint8))
    assert second.data_ptr() == first.data_ptr()


def test_growth_moves_the_block():
    # This is why the contract forbids holding views across allocate calls.
    pool = _pool()
    (small,) = pool.allocate(((64,), torch.uint8))
    before = small.data_ptr()
    pool.allocate(((1 << 20,), torch.uint8))
    (again,) = pool.allocate(((64,), torch.uint8))
    assert again.data_ptr() != before


def test_block_grows_to_the_largest_request_and_never_shrinks():
    pool = _pool()
    pool.allocate(((1024,), torch.float32))
    grown = pool.allocate(((1,), torch.uint8))[0].data_ptr()
    pool.allocate(((256,), torch.float32))
    assert pool.allocate(((1,), torch.uint8))[0].data_ptr() == grown


def test_frozen_pool_keeps_addresses_stable():
    # A captured CUDA graph records the address of the view it was given;
    # freezing is what makes that address permanent.
    pool = _pool()
    (view,) = pool.allocate(((1024,), torch.uint8))
    address = view.data_ptr()
    pool.freeze()
    (again,) = pool.allocate(((512,), torch.uint8))
    assert again.data_ptr() == address


def test_frozen_pool_refuses_growth_and_names_the_caller():
    pool = _pool()
    pool.allocate(((1024,), torch.uint8))
    pool.freeze()
    with pytest.raises(RuntimeError, match=r"test_workspace\.py:\d+:"):
        pool.allocate(((1 << 22,), torch.uint8))


def test_allocate_without_specs_is_a_no_op_even_when_frozen():
    pool = _pool()
    pool.freeze()
    assert pool.allocate() == []


def test_unfreeze_reopens_the_pool():
    pool = _pool()
    pool.allocate(((16,), torch.uint8))
    pool.freeze()
    pool.unfreeze()
    pool.allocate(((1 << 20,), torch.uint8))  # grows again
    assert not pool.frozen


def test_pools_are_per_device_and_resettable():
    reset_workspace_pools()
    try:
        first = workspace_pool("cpu")
        assert workspace_pool("cpu") is first
        reset_workspace_pools()
        assert workspace_pool("cpu") is not first
    finally:
        reset_workspace_pools()


def test_pool_allocates_on_its_own_device():
    (view,) = _pool().allocate(((16,), torch.uint8))
    assert view.device.type == "cpu"


def test_initial_size_comes_from_env():
    from tokenspeed.runtime.utils.env import envs

    with envs.TOKENSPEED_WORKSPACE_INITIAL_MB.override(1):
        pool = WorkspacePool("cpu")
    (view,) = pool.allocate(((1 << 20,), torch.uint8))  # fits, no growth
    assert view.numel() == 1 << 20
