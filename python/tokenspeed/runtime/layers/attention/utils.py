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
from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
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
    req_to_page: torch.Tensor,
    page_size: int,
    max_seq_len_k: int,
) -> torch.Tensor:
    """Build page table from req_to_page.

    req_to_page: [req_pool_size+1, max_pages] containing page IDs.
    Returns: [bs, max_pages_needed] page table slice.
    """
    max_pages = (max_seq_len_k + page_size - 1) // page_size
    return req_to_page[req_pool_indices, :max_pages]


def update_page_table_inplace(
    page_table_buf: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_page: torch.Tensor,
    page_size: int,
    max_seq_len_k: int,
):
    """Copy page table from req_to_page into pre-allocated CUDA graph buffer."""
    max_pages = (max_seq_len_k + page_size - 1) // page_size
    page_table_buf[:, :max_pages].copy_(req_to_page[req_pool_indices, :max_pages])


def token_indices_from_pages(
    req_pool_indices: torch.Tensor,
    token_positions: torch.Tensor,
    req_to_page: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Convert token positions to KV slot indices using req_to_page.

    token_positions: [bs, num_tokens] — token offsets within each request.
    Returns: [bs, num_tokens] — KV cache slot IDs (page_id * page_size + offset).
    """
    page_indices = token_positions // page_size
    offsets = token_positions % page_size
    page_ids = req_to_page[req_pool_indices].gather(1, page_indices)
    return page_ids * page_size + offsets


# --- Page-based memory profiling ---

from typing import Optional, Tuple

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
from tokenspeed.runtime.utils import get_available_gpu_memory


def profile_max_num_pages(
    attn_config: BaseAttnConfig,
    gpu_id: int,
    tp_size: int,
    gpu_memory_utilization: float,
    page_size: int,
    num_attention_layers: int,
    total_gpu_memory: int,
    world_group=None,
    draft_attn_config: BaseAttnConfig | None = None,
    draft_num_attention_layers: int | None = None,
    cache_cell_size: int | None = None,
):
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

    rest_memory = available_gpu_memory - total_gpu_memory * (1 - gpu_memory_utilization)
    if cache_cell_size is None:
        cell_size = attn_config.cache_cell_size() * num_attention_layers
    else:
        cell_size = cache_cell_size
    if draft_attn_config is not None:
        cell_size += draft_attn_config.cache_cell_size() * draft_num_attention_layers
    if cell_size <= 0:
        raise ValueError(f"KV cache cell size must be positive, got {cell_size}")
    max_num_token = int(rest_memory * (1 << 30) // cell_size)
    max_num_pages = (max_num_token + page_size - 1) // page_size
    return max_num_pages


def profile_cache_budget(
    attn_config: BaseAttnConfig,
    gpu_id: int,
    tp_size: int,
    mem_fraction_static: float,
    page_size: int,
    num_attention_layers: int,
    total_gpu_memory: int,
    mamba_memory_per_chunk: int,
    mamba_ratio: float,
    world_group=None,
    draft_attn_config: Optional[BaseAttnConfig] = None,
    draft_num_attention_layers: Optional[int] = None,
) -> Tuple[int, int]:
    """Profile GPU memory and split between KV pages and mamba chunks.

    Returns:
        (kv_max_num_pages, mamba_pool_total_chunks)
    """
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

    rest_memory = available_gpu_memory - total_gpu_memory * (1 - mem_fraction_static)
    cell_size = attn_config.cache_cell_size() * num_attention_layers
    if draft_attn_config is not None:
        cell_size += draft_attn_config.cache_cell_size() * draft_num_attention_layers

    total_cache_memory = rest_memory * (1 << 30)
    kv_memory = int(total_cache_memory / (1 + mamba_ratio))
    mamba_memory = total_cache_memory - kv_memory

    kv_cell_size = cell_size * page_size
    kv_max_num_pages = int(kv_memory // kv_cell_size)
    mamba_pool_total_chunks = max(int(mamba_memory // mamba_memory_per_chunk), 2)

    return kv_max_num_pages, mamba_pool_total_chunks
