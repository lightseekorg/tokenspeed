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

"""``zero_byte_ranges`` must zero exactly its ranges without draining the stream.

The range-table upload used to be ``torch.tensor(list, device=cuda)`` -- a
synchronizing pageable H2D that stalled the event loop for every kernel
already queued (tens of ms mid-decode at bs=32). The non-blocking contract is
load-bearing for the engine's overlap scheduling, so it gets its own test.
"""

from __future__ import annotations

import time

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.ops.kvcache.triton import zero_byte_ranges  # noqa: E402


def test_zeroes_exactly_the_requested_ranges():
    torch.manual_seed(0)
    backing = torch.randint(1, 255, (1 << 20,), dtype=torch.uint8, device="cuda")
    ref = backing.clone()
    ranges = [(0, 3), (4096, 1024), (65536, 1), ((1 << 20) - 7, 7), (99999, 4097)]
    zero_byte_ranges(backing, ranges)
    torch.cuda.synchronize()
    for off, size in ranges:
        ref[off : off + size] = 0
    assert torch.equal(backing, ref)


def test_empty_ranges_is_a_noop():
    backing = torch.randint(1, 255, (4096,), dtype=torch.uint8, device="cuda")
    ref = backing.clone()
    zero_byte_ranges(backing, [])
    torch.cuda.synchronize()
    assert torch.equal(backing, ref)


@pytest.mark.parametrize("bad", [[(-1, 4)], [(0, 0)], [(4090, 100)]])
def test_out_of_bounds_ranges_are_refused(bad):
    backing = torch.zeros(4096, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError):
        zero_byte_ranges(backing, bad)


def test_does_not_synchronize_a_busy_stream():
    """The host call must return while the stream still has queued work.

    A synchronizing upload would block here for the full sleep -- exactly the
    event-loop stall this guards against."""
    backing = torch.randint(1, 255, (1 << 16,), dtype=torch.uint8, device="cuda")
    ref = backing.clone()
    torch.cuda.synchronize()
    cycles_per_ms = 2_000_000  # conservative on any modern part
    torch.cuda._sleep(100 * cycles_per_ms)
    t0 = time.perf_counter()
    zero_byte_ranges(backing, [(128, 512)])
    host_ms = (time.perf_counter() - t0) * 1e3
    torch.cuda.synchronize()
    assert host_ms < 20, f"zero_byte_ranges blocked the host for {host_ms:.1f}ms"
    ref[128 : 128 + 512] = 0
    assert torch.equal(backing, ref)
