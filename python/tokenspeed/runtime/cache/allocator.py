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

"""Allocators for KV-cache page metadata and slot management."""

import torch

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class KVAllocator:
    """Operate on token slots and block-table metadata only.

    Physical KV storage is managed separately so the same metadata operations
    can work with different memory backends.
    """

    def __init__(
        self,
        size: int,
        device: str,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
    ):
        self.free_slots = None
        self.token_slot_refs = None
        self.size = size
        self.is_not_in_free_group = True
        self.free_group = []
        self.page_size = page_size
        self.device = device
        self.last_slot = torch.ones(max_batch_size, dtype=torch.int32) * (page_size - 1)
        self.num_pages = torch.zeros(max_batch_size, dtype=torch.int32)
        self.max_context_len = max_context_len
        self.max_page_num = (max_context_len + page_size - 1) // page_size
        self.max_batch_size = max_batch_size
        self.req_to_page = None
        self.req_to_page_cpu = None
        self.clear()

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, req_pool_index: int, need_size: int, alloced_len: int):
        page_offset = alloced_len % self.page_size
        page_num = (alloced_len + self.page_size - 1) // self.page_size
        last_page_remain = page_num * self.page_size - alloced_len
        last_page_id = self.req_to_page_cpu[req_pool_index, page_num - 1].item()
        # if last_page_remain is zero, kv_loc is Tensor([])
        kv_loc = (
            last_page_id * self.page_size
            + page_offset
            + torch.arange(0, min(last_page_remain, need_size), dtype=torch.int32)
        )
        if last_page_remain >= need_size:
            return kv_loc.to(self.device, non_blocking=True)

        remain_size = need_size - last_page_remain
        need_new_page_num = (remain_size + self.page_size - 1) // self.page_size
        if need_new_page_num > len(self.free_slots):
            # do not change self.seq_lens
            return None

        # Check if we have enough space in req_to_page tensor
        if page_num + need_new_page_num > self.max_page_num:
            logger.warning(
                "Requested page range [%s:%s] exceeds max_page_num %s. alloced_len=%s, need_size=%s, page_num=%s",
                page_num,
                page_num + need_new_page_num,
                self.max_page_num,
                alloced_len,
                need_size,
                page_num,
            )
            # Do not change self.seq_lens
            return None

        new_pages = self.free_slots[:need_new_page_num]
        self.free_slots = self.free_slots[need_new_page_num:]
        # update req_to_page
        self.req_to_page[req_pool_index, page_num : page_num + need_new_page_num] = (
            new_pages.to(self.device)
        )
        self.req_to_page_cpu[
            req_pool_index, page_num : page_num + need_new_page_num
        ] = new_pages
        # construct kv_loc
        kv_loc1 = new_pages.unsqueeze(1) * self.page_size
        offsets = torch.arange(0, self.page_size, dtype=torch.int32)
        kv_loc1 = kv_loc1 + offsets
        kv_loc1 = kv_loc1.flatten()[:remain_size]
        final_kv_loc = torch.concat([kv_loc, kv_loc1]).to(
            self.device, non_blocking=True
        )
        return final_kv_loc

    def free_extra_pages_not_cached(
        self, req_pool_index: int, real_seq_len: int, alloced_len: int
    ):
        full_page_num = real_seq_len // self.page_size
        alloced_page_num = (alloced_len + self.page_size - 1) // self.page_size
        page_num_to_free = alloced_page_num - full_page_num
        if page_num_to_free == 0:
            return
        page_ids_to_free = self.req_to_page[
            req_pool_index, full_page_num : full_page_num + page_num_to_free
        ]
        self.need_to_free.append(page_ids_to_free)

    def free_req_cache(self, req_pool_index: int, alloced_len: int):
        """Release all pages of the request when prefix cache is not used."""
        alloced_page_num = (alloced_len + self.page_size - 1) // self.page_size
        if alloced_page_num == 0:
            return
        page_ids_to_free = self.req_to_page[req_pool_index, :alloced_page_num]
        self.need_to_free.append(page_ids_to_free)

    def free_with_diff(self, new_prefix_page_ids, old_page_ids):
        # New KV pages come from the prefix tree and are already cached, so only
        # release the pages that differ from the old allocation.
        assert len(new_prefix_page_ids) == len(
            old_page_ids
        ), "[free with diff] new_prefix_page_ids and old_page_ids should have the same length"
        diff = new_prefix_page_ids != old_page_ids
        if torch.any(diff):
            logger.debug(
                "[DebugTrace] free_with_diff free page=%s", old_page_ids[diff].tolist()
            )
            self.need_to_free.append(old_page_ids[diff])
        else:
            logger.debug(
                "[DebugTrace] free_with_diff: no pages to free, all pages are cached"
            )
        return diff

    def append_to_later_free(self, page_ids: torch.Tensor) -> None:
        self.need_to_free.append(page_ids)

    def free(self, req_pool_index: int, indices=None) -> None:
        if self.is_not_in_free_group:
            num_pages = self.num_pages[req_pool_index]
            pages = self.req_to_page[req_pool_index, :num_pages].cpu()
            free_slots = [self.free_slots]
            for i in range(num_pages):
                page_index = pages[i]
                free_slots.append(
                    torch.arange(
                        page_index * self.page_size,
                        (page_index + 1) * self.page_size,
                        dtype=torch.int32,
                    )
                )
            self.free_slots = torch.concat(free_slots)
            self.num_pages[req_pool_index] = 0
            self.last_slot[req_pool_index] = self.page_size - 1
        else:
            self.free_group.append(req_pool_index)

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.need_to_free:
            pages_need_to_free = torch.concat(self.need_to_free)
            logger.debug(
                "[DebugTrace] free_group_end pages_need_to_free=%s",
                pages_need_to_free.tolist(),
            )
            token_level_offsets = torch.arange(self.page_size, device=self.device)
            slots_to_free = (
                pages_need_to_free[:, None] * self.page_size + token_level_offsets
            ).flatten()
            writted_positions = slots_to_free[self.token_slot_refs[slots_to_free] >= 1]
            self.token_slot_refs[writted_positions] += -1
            self.free_slots = torch.concat([self.free_slots, pages_need_to_free.cpu()])
            self.need_to_free = []

    def clear(self) -> None:
        # Page 0 is used for padding
        self.free_slots = torch.arange(
            1, self.size // self.page_size, dtype=torch.int32
        )
        if self.token_slot_refs is None:
            self.token_slot_refs = torch.zeros(
                self.size, dtype=torch.int32, device=self.device
            )
        else:
            self.token_slot_refs.zero_()
        if self.req_to_page is None:
            self.req_to_page = torch.zeros(
                (self.max_batch_size, self.max_page_num),
                dtype=torch.int32,
                device=self.device,
            )
            self.req_to_page_cpu = torch.zeros(
                (self.max_batch_size, self.max_page_num),
                dtype=torch.int32,
                pin_memory=True,
            )
        else:
            self.req_to_page.zero_()
            self.req_to_page_cpu.zero_()
        self.free_group = []
        self.need_to_free = []
