# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.kvcache import triton as kvcache_triton

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def test_zero_range_staging_min_rows_bypasses_small_tables(monkeypatch):
    monkeypatch.setattr(kvcache_triton, "_ZERO_RANGE_STAGING_ENABLED", True)
    monkeypatch.setattr(kvcache_triton, "_ZERO_RANGE_STAGING_MIN_ROWS", 3)

    assert not kvcache_triton._should_stage_zero_ranges(2)
    assert kvcache_triton._should_stage_zero_ranges(3)


def test_zero_range_stager_rejects_oversize_without_allocating_slots():
    stager = kvcache_triton._ZeroRangeTableStager(capacity=1)

    assert stager.try_stage([(0, 4), (8, 4)], torch.device("cuda", 0)) is None
    assert stager.snapshot_stats() == {
        "staged": 0,
        "busy_fallback": 0,
        "oversize_fallback": 1,
        "capture_fallback": 0,
    }
    assert stager._slots_by_device == {}


def test_zero_range_stager_rejects_capture_without_allocating_slots(monkeypatch):
    stager = kvcache_triton._ZeroRangeTableStager(capacity=1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    assert stager.try_stage([(0, 4)], torch.device("cuda", 0)) is None
    assert stager.snapshot_stats() == {
        "staged": 0,
        "busy_fallback": 0,
        "oversize_fallback": 0,
        "capture_fallback": 1,
    }
    assert stager._slots_by_device == {}


@requires_cuda
def test_zero_byte_ranges_reuses_four_slot_staging_without_data_corruption(
    monkeypatch,
):
    stager = kvcache_triton._ZeroRangeTableStager(capacity=4)
    monkeypatch.setattr(kvcache_triton, "_ZERO_RANGE_STAGING_ENABLED", True)
    monkeypatch.setattr(kvcache_triton, "_ZERO_RANGE_STAGING_MIN_ROWS", 0)
    monkeypatch.setattr(kvcache_triton, "_ZERO_RANGE_TABLE_STAGER", stager)
    backing = torch.full((256,), 7, dtype=torch.uint8, device="cuda")

    for index in range(12):
        offset = index * 8
        kvcache_triton.zero_byte_ranges(backing, [(offset, 4), (128 + offset, 4)])
    torch.cuda.synchronize()

    expected = torch.full((256,), 7, dtype=torch.uint8)
    for index in range(12):
        offset = index * 8
        expected[offset : offset + 4] = 0
        expected[128 + offset : 132 + offset] = 0
    assert torch.equal(backing.cpu(), expected)
    assert stager.snapshot_stats() == {
        "staged": 12,
        "busy_fallback": 0,
        "oversize_fallback": 0,
        "capture_fallback": 0,
    }
