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

import sys
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.cache.l2.storage import HostCacheStorage
from tokenspeed.runtime.cache.transfer.layout import (
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
)


@pytest.fixture(autouse=True)
def _fake_torch(monkeypatch):
    uint8 = object()

    def empty(num_bytes, *, dtype, pin_memory):
        assert dtype is uint8
        assert pin_memory is True
        return bytearray(num_bytes)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(empty=empty, uint8=uint8),
    )


def _layout():
    field = lambda name, size: CacheField(name, 0, 0, size, size)
    return CacheTransferLayout(
        num_lcm_blocks=2,
        groups=(
            CacheGroupLayout("full", 4, (field("k", 32),)),
            CacheGroupLayout("state", 1, (field("state", 80),)),
        ),
        buffers=(object(),),
        consumers=(("k",), ("state",)),
    )


def test_block_offsets_follow_group_packing_without_overlap():
    layout = _layout()
    storage = HostCacheStorage(layout, num_host_lcm_blocks=2)

    assert storage.host_block_offset(0, 1) == 0
    assert storage.host_block_offset(0, 4) == 3 * storage.host_cache_block_bytes[0]
    assert storage.host_block_offset(0, 5) == storage.host_lcm_block_bytes
    assert storage.host_block_offset(1, 1) == 0
    assert storage.host_block_offset(1, 2) == storage.host_lcm_block_bytes


@pytest.mark.parametrize("group_index,block_id", [(0, 0), (0, 9), (1, 3)])
def test_null_and_out_of_range_blocks_are_rejected(group_index, block_id):
    layout = _layout()
    storage = HostCacheStorage(layout, num_host_lcm_blocks=2)

    with pytest.raises(IndexError):
        storage.host_block_offset(group_index, block_id)


def test_host_fields_are_packed_without_padding():
    field = lambda name, stride, payload: CacheField(name, 0, 0, stride, payload)
    layout = CacheTransferLayout(
        num_lcm_blocks=10,
        groups=(
            CacheGroupLayout(
                "full",
                16,
                (
                    field("full.k", 96, 40),
                    field("full.v", 96, 40),
                ),
            ),
            CacheGroupLayout(
                "state",
                1,
                (
                    field("state.ssm", 2048, 1000),
                    field("state.conv", 1024, 200),
                ),
            ),
        ),
        buffers=(object(),),
        consumers=(("full.k", "full.v"), ("state.ssm", "state.conv")),
    )

    storage = HostCacheStorage(layout, num_host_lcm_blocks=1)

    assert storage.host_cache_block_bytes == (80, 1200)
    assert storage.host_field_offsets == ((0, 40), (0, 1000))
    assert storage.host_lcm_block_bytes == 1280
