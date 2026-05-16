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
import torch

from tokenspeed.runtime.cache.allocator import KVAllocator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_free_group_end_returns_freed_pages_to_free_slots():
    """free_extra_pages_not_cached + free_group_end should push the freed
    page ids back onto free_slots and decrement token_slot_refs."""
    page_size = 4
    max_pages = 8
    total_slots = max_pages * page_size

    allocator = KVAllocator(
        size=total_slots,
        device="cuda",
        max_batch_size=2,
        max_context_len=total_slots,
        page_size=page_size,
    )

    # Assign known pages [1, 5, 6] to request 0
    allocator.req_to_page[0, :3] = torch.tensor(
        [1, 5, 6], dtype=torch.int32, device="cuda"
    )
    for page in (1, 5, 6):
        allocator.token_slot_refs[page * page_size : (page + 1) * page_size] = 1

    initial_free_count = allocator.free_slots.numel()

    # alloced_len=12 covers 3 pages; real_seq_len=4 keeps 1 -> free pages [5, 6]
    allocator.free_extra_pages_not_cached(
        req_pool_index=0, real_seq_len=page_size, alloced_len=3 * page_size
    )
    allocator.free_group_end()

    assert allocator.free_slots.numel() == initial_free_count + 2
    freed_pages = set(allocator.free_slots[initial_free_count:].tolist())
    assert freed_pages == {5, 6}
    for page in (5, 6):
        slot_refs = allocator.token_slot_refs[page * page_size : (page + 1) * page_size]
        assert (slot_refs == 0).all()
