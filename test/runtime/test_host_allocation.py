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

"""Real allocator restoration, aligned buffers and memory budgeting."""

import gc
import sys
from pathlib import Path

import pytest
import torch

from tokenspeed.runtime.cache.host_allocation import (
    allocate_host_buffer,
    host_register_settings,
    validate_host_register_threads,
)
from tokenspeed.runtime.cache.transfer.lanes import (
    largest_pinned_fit,
    pinned_host_bytes,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")


@pytest.mark.parametrize("threads", [0, -1, 3, 129, True])
def test_invalid_thread_override_is_rejected(threads):
    with pytest.raises(ValueError, match="power of two"):
        validate_host_register_threads(threads)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="requires real CUDA allocator",
)
@pytest.mark.parametrize(
    "config",
    [
        "pinned_use_cuda_host_register:False,pinned_num_register_threads:1",
        "max_split_size_mb:256,garbage_collection_threshold:0.8,roundup_power2_divisions:[256:1,512:2,>:4],pinned_use_cuda_host_register:False,pinned_num_register_threads:1",
        "max_split_size_mb:128,pinned_use_cuda_host_register:True,pinned_num_register_threads:4",
    ],
)
@pytest.mark.parametrize("fail", [False, True])
def test_settings_restore_after_success_and_exception(config, fail):
    original = torch.cuda.memory._snapshot()["allocator_settings"]
    try:
        torch.cuda.memory._set_allocator_settings(config)
        before = torch.cuda.memory._snapshot()["allocator_settings"]
        try:
            with host_register_settings(8):
                active = torch.cuda.memory._snapshot()["allocator_settings"]
                assert active["pinned_use_cuda_host_register"] is True
                assert active["pinned_num_register_threads"] == 8
                if fail:
                    raise RuntimeError("allocation interrupted")
        except RuntimeError as error:
            assert fail and str(error) == "allocation interrupted"
        assert torch.cuda.memory._snapshot()["allocator_settings"] == before
    finally:
        restore = original["PYTORCH_CUDA_ALLOC_CONF"].rstrip(",")
        restore += f",pinned_use_cuda_host_register:{original['pinned_use_cuda_host_register']},pinned_num_register_threads:{original['pinned_num_register_threads']}"
        torch.cuda.memory._set_allocator_settings(restore.lstrip(","))
        torch.cuda.memory._set_allocator_settings(original["PYTORCH_CUDA_ALLOC_CONF"])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="requires pinned memory and CUDA",
)
def test_aligned_buffer_survives_views_and_restores_later_allocations():
    before = torch.cuda.memory._snapshot()["allocator_settings"]
    buffer = allocate_host_buffer((2 << 20) + 37, register_threads=8)
    assert buffer.is_pinned() and buffer.data_ptr() % 4096 == 0
    assert buffer.numel() == (2 << 20) + 37
    buffer.fill_(17)
    tail = buffer[-37:]
    del buffer
    gc.collect()
    assert torch.all(tail.to("cuda", non_blocking=True) == 17).item()
    assert torch.cuda.memory._snapshot()["allocator_settings"] == before
    snapshot = allocate_host_buffer(1 << 20, register_threads=None)
    snapshot.fill_(29)
    assert torch.all(snapshot.to("cuda", non_blocking=True) == 29).item()
    torch.cuda.synchronize()
    assert torch.cuda.memory._snapshot()["allocator_settings"] == before


def test_l2_alignment_padding_fits_the_pinned_reservation():
    assert largest_pinned_fit(8192, block_bytes=1024) == 8192
    payload = largest_pinned_fit(8192, block_bytes=1024, padding_bytes=4095)
    assert payload == 4096
    assert pinned_host_bytes(payload + 4095) <= 8192
    assert pinned_host_bytes(payload + 1024 + 4095) > 8192
    assert largest_pinned_fit(4095, block_bytes=1024, padding_bytes=4095) == 0
    with pytest.raises(ValueError, match="non-negative"):
        largest_pinned_fit(8192, block_bytes=1024, padding_bytes=-1)
