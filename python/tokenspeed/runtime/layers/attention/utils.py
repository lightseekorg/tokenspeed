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

import torch
import triton
import triton.language as tl

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.utils import get_available_gpu_memory


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


# --- Page table helpers (shared across attention backends) ---


def build_page_table(
    req_pool_indices: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    max_seq_len_k: int,
) -> torch.Tensor:
    """Build page table from a batch-ordered table.

    page_table: [bs, max_pages] page IDs, row i == batch position i (the
    drafter's page table or the idle/warmup placeholder).
    Returns: [bs, max_pages_needed] page table slice.
    """
    max_pages = (max_seq_len_k + page_size - 1) // page_size
    return page_table[: req_pool_indices.shape[0], :max_pages]


# --- Page-based memory profiling ---


def profile_available_cache_memory_bytes(
    attn_config: AttnConfig,
    gpu_id: int,
    tp_size: int,
    gpu_memory_utilization: float,
    total_gpu_memory: int,
    *,
    graph_reserve_bytes: int = 0,
    world_group=None,
) -> int:
    cpu_group = (
        pg_manager.get_process_group("gloo", world_group)
        if world_group is not None
        else None
    )
    available_gpu_memory = get_available_gpu_memory(
        attn_config.device,
        gpu_id,
        distributed=tp_size > 1,
        cpu_group=cpu_group,
    )
    cache_memory = (
        available_gpu_memory
        - total_gpu_memory * (1 - gpu_memory_utilization)
        - graph_reserve_bytes / (1 << 30)
    )
    return int(cache_memory * (1 << 30))
