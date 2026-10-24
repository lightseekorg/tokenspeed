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

"""Integration checks for Host budgeting against the active pinned allocator."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs the pinned GPU allocator"
)
@pytest.mark.parametrize(
    ("config", "exact_large"),
    [
        ("", False),
        ("pinned_max_round_threshold_mb:13", True),
        ("pinned_max_cached_size_mb:8", True),
        ("pinned_max_round_threshold_mb:32,pinned_max_cached_size_mb:8", True),
    ],
)
def test_host_budget_matches_pinned_allocation(config, exact_large):
    # A child process isolates allocator configuration and its pinned cache.
    code = r"""
import os
import sys
import torch
from tokenspeed.runtime.cache.transfer.lanes import (
    check_host_memory, largest_pinned_fit, pinned_host_bytes,
)
exact_large = sys.argv[1] == "True"
blocks = []
for mib in (5, 8, 9, 13, 14):
    before = torch._C._cuda_hostMemoryStats()["allocated_bytes"]["current"]
    block = torch.empty(mib * 2**20, dtype=torch.uint8, pin_memory=True)
    blocks.append(block)
    actual = torch._C._cuda_hostMemoryStats()["allocated_bytes"]["current"] - before
    assert pinned_host_bytes(block.numel()) == actual
room = 14 * 2**20
assert largest_pinned_fit(room, block_bytes=1) == (room if exact_large else 8 * 2**20)
if exact_large:
    check_host_memory(pinned_host_bytes(192_000_000_000),
                      share_bytes=210_000_000_000, purpose="large Host cache")
    from test.runtime.test_host_executor import TestHostMemoryShareTest
    case = TestHostMemoryShareTest()
    case.setup_method()
    executor, _ = case._build(available_gb=900, nprocs_per_node=4,
                             device_lcm_blocks=580, host_size_gb=192)
    assert executor.num_host_pages == 192_001
    if os.environ["PYTORCH_ALLOC_CONF"] == "pinned_max_round_threshold_mb:13":
        room = 13 * 2**20 + 128
        assert largest_pinned_fit(room, block_bytes=10**6) == 8 * 10**6
        capped, _ = case._build(
            available_gb=4 * (case.headroom + room) / 10**9,
            nprocs_per_node=4, device_lcm_blocks=20, host_size_gb=0,
        )
        assert capped.num_host_pages == 9
else:
    try:
        check_host_memory(pinned_host_bytes(192_000_000_000),
                          share_bytes=210_000_000_000, purpose="large Host cache")
    except ValueError:
        pass
    else:
        raise AssertionError("default rounding must exceed this budget")
# The estimator must use effective configuration, not just startup environment.
torch._C._accelerator_setAllocatorSettings(
    "pinned_max_round_threshold_mb:8,pinned_max_cached_size_mb:8")
assert pinned_host_bytes(9 * 2**20) == 9 * 2**20
assert largest_pinned_fit(14 * 2**20, block_bytes=1) == 14 * 2**20
"""
    env = dict(os.environ)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env["PYTORCH_ALLOC_CONF"] = config
    result = subprocess.run(
        [sys.executable, "-c", code, str(exact_large)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs the pinned GPU allocator"
)
def test_host_budget_remains_conservative_without_allocator_snapshot():
    code = r"""
import torch
from tokenspeed.runtime.cache.transfer.lanes import largest_pinned_fit, pinned_host_bytes
before = torch._C._cuda_hostMemoryStats()["allocated_bytes"]["current"]
block = torch.empty(14 * 2**20, dtype=torch.uint8, pin_memory=True)
actual = torch._C._cuda_hostMemoryStats()["allocated_bytes"]["current"] - before
assert pinned_host_bytes(block.numel()) >= actual
room = 15 * 2**20
fit = largest_pinned_fit(room, block_bytes=2**20)
assert pinned_host_bytes(fit) <= room
assert fit <= room
"""
    env = dict(os.environ)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env["PYTORCH_ALLOC_CONF"] = (
        "backend:cudaMallocAsync,pinned_max_round_threshold_mb:13"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
