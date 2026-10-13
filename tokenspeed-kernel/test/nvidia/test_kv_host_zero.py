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

"""Mapped-host zeroing preserves byte boundaries and GPU stream dependencies."""

from unittest.mock import patch

import pytest
import torch
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

from cutlass import cute  # noqa: E402
from tokenspeed_kernel.ops.kvcache import zero_byte_ranges  # noqa: E402


@pytest.mark.parametrize("extra_ranges", [0, 60])
def test_host_zero_preserves_neighbors_and_storage_offset(extra_ranges):
    ranges = [(3, 7), (31, 27648), (30003, 73729), (110001, 786435)]
    ranges.extend((900001 + i * 16, 3) for i in range(extra_ranges))
    base = torch.full((901043,), 173, dtype=torch.uint8, pin_memory=True)
    view = base[5:-13]
    expected = base.clone()
    for offset, size in ranges:
        expected[5 + offset : 5 + offset + size] = 0
    zero_byte_ranges(view, ranges, device="cuda")
    torch.cuda.synchronize()
    assert torch.equal(base, expected)


def test_host_zero_orders_write_clear_and_read_on_side_stream_without_recompile():
    host = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
    source = torch.full((4096,), 173, dtype=torch.uint8, device="cuda")
    seen = torch.empty_like(source)
    zero_byte_ranges(host, [(0, 1)], device=source.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with patch.object(cute, "compile", wraps=cute.compile) as compile_call:
        for ranges in ([(3, 1)], [(7, 3), (16, 33)], [(0, 4096)], []):
            with torch.cuda.stream(stream):
                # A synchronous CPU clear would run before this pending GPU
                # write and be overwritten. A later GPU read checks the order.
                torch.cuda._sleep(10_000_000)
                host.copy_(source, non_blocking=True)
                zero_byte_ranges(host, ranges, device=source.device)
                seen.copy_(host, non_blocking=True)
            stream.synchronize()
            expected = torch.full((4096,), 173, dtype=torch.uint8)
            for offset, size in ranges:
                expected[offset : offset + size] = 0
            assert torch.equal(seen.cpu(), expected)
            assert torch.equal(host, expected)
    assert compile_call.call_count == 0


def test_zero_range_placement_and_validation():
    cpu = torch.full((31,), 173, dtype=torch.uint8)
    zero_byte_ranges(cpu, [(1, 3)], device="cpu")
    assert cpu.tolist() == [173, 0, 0, 0] + [173] * 27
    with pytest.raises(ValueError, match="pinned host"):
        zero_byte_ranges(cpu, [(1, 3)], device="cuda")
    pinned = cpu.pin_memory()
    for ranges in ([(-1, 1)], [(0, 0)], [(30, 2)]):
        with pytest.raises(ValueError, match="within backing"):
            zero_byte_ranges(pinned, ranges, device="cuda")
    with pytest.raises(ValueError, match="contiguous uint8"):
        zero_byte_ranges(pinned[::2], [(0, 1)], device="cuda")
    device = torch.full((31,), 173, dtype=torch.uint8, device="cuda")
    zero_byte_ranges(device, [(1, 3)], device=device.device)
    assert torch.equal(device.cpu(), cpu)


def test_host_metadata_is_not_captured_as_temporary_cpu_storage():
    host = torch.full((32,), 173, dtype=torch.uint8, pin_memory=True)
    zero_byte_ranges(host, [(1, 3)], device="cuda")
    torch.cuda.synchronize()
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        # Metadata is rebuilt by the scheduler before model graph replay.
        # Do not embed a temporary pinned CPU table in a replayable graph.
        with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
            zero_byte_ranges(host, [(7, 3)], device="cuda")
