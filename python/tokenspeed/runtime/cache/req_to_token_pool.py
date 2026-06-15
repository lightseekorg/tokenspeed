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

"""
Memory pool.

ReqToTokenPool maps a request to its token locations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


@dataclass
class ReqToTokenPoolInfo:
    """For chunked prefill"""

    verified_len: int
    alloced_len: int
    alloced_slots: torch.Tensor


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        # Tag as "kv_cache": this per-token page state is invalid once KV is
        # discarded, so it is released/restored alongside the KV cache.
        with memory_saver_adapter.region(tag="kv_cache", enable_cpu_backup=False):
            self.req_to_token = torch.zeros(
                (size, max_context_len), dtype=torch.int32, device=device
            )
            # verified_lens records the valid historical KV cache length for each request,
            # mainly used to determine the KV cache position to write for this computation
            self.verified_lens = torch.zeros(size, dtype=torch.int32, device=device)
            # alloced_lens records the allocated KV cache length for each request,
            # which can be larger than verified_lens, mainly used to determine the KV cache position for this allocation
            self.alloced_lens = torch.zeros(size, dtype=torch.int32, device=device)
        self.alloced_lens_cpu = torch.zeros(size, dtype=torch.int32, pin_memory=True)
        self.free_slots = list(range(size))[1:]

    def set_req_pool_info(self, req_pool_idx: int, metadata: ReqToTokenPoolInfo):
        self.verified_lens[req_pool_idx] = metadata.verified_len
        self.alloced_lens[req_pool_idx] = metadata.alloced_len
        self.alloced_lens_cpu[req_pool_idx] = metadata.alloced_len
        self.req_to_token[req_pool_idx, : metadata.alloced_len] = metadata.alloced_slots

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> list[int] | None:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        # During overlap scheduling, after a retracted request frees its req_pool,
        # the forward_thread may still modify its verified_lens, causing errors when
        # reusing this position. Here we ensure that when req_idx is reused, the corresponding resource is empty.
        self.verified_lens[select_index] = 0
        self.alloced_lens[select_index] = 0
        self.alloced_lens_cpu[select_index] = 0

        return select_index

    def free(self, free_index: int | list[int]) -> None:
        free_indices = [free_index] if isinstance(free_index, int) else free_index
        self.free_slots.extend(free_indices)
        for index in free_indices:
            self.verified_lens[index] = 0
            self.alloced_lens[index] = 0
            self.alloced_lens_cpu[index] = 0

    def clear(self):
        # clear method is called during flush_cache
        # slot 0 is used as padding in spec_cuda_graph and is not allocated externally
        self.free_slots = list(range(self.size))[1:]
        self.verified_lens.zero_()
        self.alloced_lens.zero_()
        self.alloced_lens_cpu.zero_()
