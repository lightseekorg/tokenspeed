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

import pytest

from tokenspeed.runtime.cache.host_storage import HostCacheStorage
from tokenspeed.runtime.cache.layout import (
    CacheGroupLayout,
    CacheSegment,
    CacheTransferLayout,
)


def _layout():
    segment = lambda name, size: CacheSegment(name, 0, 0, size, size)
    return CacheTransferLayout(
        logical_block_tokens=128,
        groups=(
            CacheGroupLayout("full", 4, 9, (segment("k", 32),)),
            CacheGroupLayout("state", 1, 3, (segment("state", 80),)),
        ),
        buffers=(object(),),
        consumers=(("k",), ("state",)),
    )


def test_child_offsets_follow_group_packing_without_overlap():
    layout = _layout()
    packed = layout.pack()
    storage = HostCacheStorage(
        layout,
        num_lcm_blocks=2,
        backing=bytearray(2 * packed.parent_bytes),
    )

    assert storage.child_offset(0, 1) == 0
    assert storage.child_offset(0, 4) == 3 * packed.child_bytes[0]
    assert storage.child_offset(0, 5) == packed.parent_bytes
    assert storage.child_offset(1, 1) == 0
    assert storage.child_offset(1, 2) == packed.parent_bytes


@pytest.mark.parametrize("group_index,page_id", [(0, 0), (0, 9), (1, 3)])
def test_null_and_out_of_range_pages_are_rejected(group_index, page_id):
    layout = _layout()
    packed = layout.pack()
    storage = HostCacheStorage(
        layout,
        num_lcm_blocks=2,
        backing=bytearray(2 * packed.parent_bytes),
    )

    with pytest.raises(IndexError):
        storage.child_offset(group_index, page_id)


def test_backing_size_must_match_exact_parent_budget():
    layout = _layout()
    with pytest.raises(ValueError, match="backing"):
        HostCacheStorage(layout, num_lcm_blocks=2, backing=bytearray(1))
