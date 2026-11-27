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

"""Allocator-policy preservation, aligned buffers and memory budgeting."""

import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest
import torch

from tokenspeed.runtime.cache.host_allocation import (
    allocate_host_buffer,
    host_register_enabled,
)
from tokenspeed.runtime.cache.transfer.lanes import (
    largest_pinned_fit,
    pinned_host_bytes,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-1gpu")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="requires real CUDA allocator",
)
@pytest.mark.parametrize(
    "variable,config,expected",
    [
        ("PYTORCH_CUDA_ALLOC_CONF", None, ("native", False, 1)),
        (
            "PYTORCH_CUDA_ALLOC_CONF",
            "pinned_use_cuda_host_register:False,pinned_num_register_threads:8",
            ("native", False, 8),
        ),
        (
            "PYTORCH_CUDA_ALLOC_CONF",
            "pinned_use_cuda_host_register:True,pinned_num_register_threads:1",
            ("native", True, 1),
        ),
        (
            "PYTORCH_CUDA_ALLOC_CONF",
            "pinned_use_cuda_host_register:True,pinned_num_register_threads:2",
            ("native", True, 2),
        ),
        (
            "PYTORCH_CUDA_ALLOC_CONF",
            "pinned_use_cuda_host_register:True,pinned_num_register_threads:8",
            ("native", True, 8),
        ),
        (
            "PYTORCH_CUDA_ALLOC_CONF",
            "backend:cudaMallocAsync",
            ("cudaMallocAsync", None, None),
        ),
        (
            "PYTORCH_ALLOC_CONF",
            "pinned_use_cuda_host_register:True,pinned_num_register_threads:4",
            ("native", True, 4),
        ),
    ],
    ids=[
        "default",
        "disabled",
        "serial",
        "parallel2",
        "parallel8",
        "async",
        "generic_alias",
    ],
)
def test_allocator_environment_is_respected(variable, config, expected):
    # Each process starts with a fresh allocator and registration thread pool.
    env = dict(os.environ)
    env.pop("PYTORCH_ALLOC_CONF", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if config is not None:
        env[variable] = config
    env["OMP_NUM_THREADS"] = "1"
    script = """
import json
import sys
import torch
import pytest

backend = torch.cuda.get_allocator_backend()
before = torch.cuda.memory._snapshot()["allocator_settings"] if backend == "native" else None
observed = [backend, before["pinned_use_cuda_host_register"] if before else None,
            before["pinned_num_register_threads"] if before else None]
if observed != json.loads(sys.argv[1]):
    print(f"Allocator environment unsupported by this PyTorch build: {observed}")
    raise SystemExit(77)
status = pytest.main(sys.argv[2:])
if before is not None:
    assert torch.cuda.memory._snapshot()["allocator_settings"] == before
raise SystemExit(status)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            json.dumps(expected),
            f"{Path(__file__)}::test_host_buffer_keeps_allocator_policy",
            f"{Path(__file__).with_name('test_host_executor.py')}::TestCompactHostParentTest::test_both_tiers_use_the_parent_rounded_to_whole_kv_pages[True]",
            f"{Path(__file__).with_name('test_host_executor.py')}::TestCompactHostParentTest::test_both_tiers_use_the_parent_rounded_to_whole_kv_pages[disk_cuda]",
            "-q",
            "-rs",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 77:
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout, result.stdout + result.stderr


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="requires pinned memory and CUDA",
)
def test_host_buffer_keeps_allocator_policy():
    torch.cuda.init()
    warmup = torch.empty(1, dtype=torch.uint8, pin_memory=True)
    native = torch.cuda.get_allocator_backend() == "native"
    before = torch.cuda.memory._snapshot()["allocator_settings"] if native else None
    aligned = host_register_enabled()
    before_threads = psutil.Process().num_threads()
    buffer = allocate_host_buffer((2 << 20) + 37, align=aligned)
    if before is not None and (
        not before["pinned_use_cuda_host_register"]
        or before["pinned_num_register_threads"] == 1
    ):
        # Allow driver housekeeping, but never an implicit 128-thread pool.
        assert psutil.Process().num_threads() <= before_threads + 8
    assert buffer.is_pinned() and buffer.numel() == (2 << 20) + 37
    if aligned:
        assert buffer.data_ptr() % 4096 == 0
    buffer.fill_(17)
    tail = buffer[-37:]
    del buffer, warmup
    gc.collect()
    assert torch.all(tail.to("cuda", non_blocking=True) == 17).item()
    snapshot = allocate_host_buffer(1 << 20, align=False)
    snapshot.fill_(29)
    assert torch.all(snapshot.to("cuda", non_blocking=True) == 29).item()
    torch.cuda.synchronize()
    if native:
        assert torch.cuda.memory._snapshot()["allocator_settings"] == before


def test_l2_alignment_padding_fits_the_pinned_reservation():
    assert largest_pinned_fit(8192, block_bytes=1024) == 8192
    payload = largest_pinned_fit(8192, block_bytes=1024, padding_bytes=4095)
    assert payload == 4096
    assert pinned_host_bytes(payload + 4095) <= 8192
    assert pinned_host_bytes(payload + 1024 + 4095) > 8192
    assert largest_pinned_fit(4095, block_bytes=1024, padding_bytes=4095) == 0
    # A roomy non-power-of-two budget needs the padded fallback when rounded.
    payload = largest_pinned_fit(3000, block_bytes=1024, padding_bytes=100)
    assert pinned_host_bytes(payload + 100) <= 3000
    assert pinned_host_bytes(payload + 1024 + 100) > 3000
    with pytest.raises(ValueError, match="non-negative"):
        largest_pinned_fit(8192, block_bytes=1024, padding_bytes=-1)
