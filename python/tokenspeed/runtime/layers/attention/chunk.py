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

from dataclasses import dataclass
from functools import cached_property

import torch

from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.triton import tl, triton


@triton.jit(
    do_not_specialize=["page_table_ptr_stride"],
    do_not_specialize_on_alignment=[
        "chunk_start_idx_ptr",
        "chunk_seq_lens_ptr",
        "chunk_cum_seq_lens_ptr",
    ],
)
def create_chunked_cache_kv_indices_paged(
    page_table_ptr,  # [bs, max_pages], row i == batch position i
    chunk_start_idx_ptr,  # (batch_size,)
    chunk_seq_lens_ptr,  # (batch_size,)
    chunk_cum_seq_lens_ptr,  # (batch_size + 1,)
    chunk_kv_indices_ptr,  # (num_chunk_tokens,)
    page_table_ptr_stride,
    PAGE_SIZE: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    chunk_kv_indices_offset = tl.load(chunk_cum_seq_lens_ptr + pid)

    chunk_start_pos = tl.load(chunk_start_idx_ptr + pid).to(tl.int32)
    chunk_seq_len = tl.load(chunk_seq_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(chunk_seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < chunk_seq_len
        token_pos = chunk_start_pos + offset
        page_idx = token_pos // PAGE_SIZE
        page_id = tl.load(
            page_table_ptr + pid * page_table_ptr_stride + page_idx,
            mask=mask,
        )
        kv_slot = page_id * PAGE_SIZE + token_pos % PAGE_SIZE
        tl.store(
            chunk_kv_indices_ptr + chunk_kv_indices_offset + offset,
            kv_slot,
            mask=mask,
        )


def get_max_chunk_capacity():
    return (
        global_server_args_dict["chunked_prefill_size"]
        * global_server_args_dict["mla_chunk_multiplier"]
    )


@dataclass
class Chunks:
    starts: torch.Tensor
    ends: torch.Tensor
    len_in_chunk: torch.Tensor

    @cached_property
    def boundaries(self) -> torch.Tensor:
        """Per-request positions before the first loop and after each loop."""
        return torch.cat((self.ends.new_zeros((1, self.ends.shape[1])), self.ends))

    @cached_property
    def cum_seq_lens(self):
        num_chunks = self.starts.shape[0]
        bs = self.starts.shape[1]
        result = torch.zeros(
            num_chunks, bs + 1, device=self.starts.device, dtype=torch.int32
        )
        torch.cumsum(self.len_in_chunk, dim=1, out=result[:, 1:])
        return result


def chunking(prefix_lens: torch.Tensor, num_chunks, batch_size, chunk_len):
    starts = (
        torch.arange(num_chunks, device=prefix_lens.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(-1, batch_size)
        * chunk_len
    )
    ends = torch.min(prefix_lens.unsqueeze(0), starts + chunk_len).to(torch.int32)

    chunks = Chunks(
        starts=starts,
        ends=ends,
        len_in_chunk=(ends - starts).clamp(min=0).to(torch.int32),
    )
    return chunks


def plan_prefix_chunks(prefix_lens_cpu: torch.Tensor, capacity: int) -> Chunks:
    """Share a bounded expansion buffer among requests with remaining history.

    Args:
        prefix_lens_cpu: One nonnegative cached-prefix length per request on CPU.
        capacity: Maximum total history rows expanded in one loop.

    Returns:
        CPU chunk boundaries and lengths in request order. Every prefix is read
        exactly once; a finished request contributes no further rows.
    """
    if prefix_lens_cpu.device.type != "cpu" or prefix_lens_cpu.ndim != 1:
        raise ValueError("prefix lengths must be a one-dimensional CPU tensor")
    batch_size = prefix_lens_cpu.numel()
    if capacity <= 0 or capacity < batch_size:
        raise ValueError("history capacity must be positive and cover the batch")
    lengths = prefix_lens_cpu.tolist()
    if any(length < 0 for length in lengths):
        raise ValueError("prefix lengths must be nonnegative")
    if not lengths:
        empty = torch.empty((0, 0), dtype=torch.int32)
        return Chunks(empty, empty, empty)
    share = capacity // batch_size
    # Keep small partitions stable: FP8 attention rounding depends on the
    # partition size, and pooling these short chunks can amplify its error.
    if share < 1024:
        return chunking(
            prefix_lens_cpu, (max(lengths) + share - 1) // share, batch_size, share
        )
    consumed = [0] * batch_size
    starts, ends = [], []
    while consumed != lengths:
        remaining = [length - start for length, start in zip(lengths, consumed)]
        active = sorted(
            (i for i, length in enumerate(remaining) if length),
            key=remaining.__getitem__,
        )
        allocated = [0] * batch_size
        budget = capacity
        for offset, request in enumerate(active):
            share, remainder = divmod(budget, len(active) - offset)
            if remaining[request] <= share:
                allocated[request] = remaining[request]
                budget -= remaining[request]
            else:
                for position, request in enumerate(active[offset:]):
                    allocated[request] = share + (position < remainder)
                break
        starts.append(consumed)
        consumed = [start + size for start, size in zip(consumed, allocated)]
        ends.append(consumed)
    start_tensor = torch.tensor(starts, dtype=torch.int32).reshape(-1, batch_size)
    end_tensor = torch.tensor(ends, dtype=torch.int32).reshape(-1, batch_size)
    return Chunks(start_tensor, end_tensor, end_tensor - start_tensor)


def get_chunks_paged(prefix_lens, prefix_lens_cpu, page_table, page_size):
    """Chunk the cached prefixes and resolve each chunk's KV slots through
    the batch-ordered ``[bs, max_pages]`` kernel page table."""
    device: torch.device = prefix_lens.device
    batch_size = len(prefix_lens_cpu)
    chunks_cpu = plan_prefix_chunks(prefix_lens_cpu, get_max_chunk_capacity())
    num_chunks = chunks_cpu.starts.shape[0]
    arrays = torch.stack(
        (chunks_cpu.starts, chunks_cpu.ends, chunks_cpu.len_in_chunk)
    ).to(device=device)
    chunks = Chunks(*arrays.unbind(0))

    num_tokens_per_forward = chunks_cpu.len_in_chunk.sum(dim=1).tolist()

    chunk_kv_indices_list = []
    for idx in range(num_chunks):
        chunk_kv_indices = torch.empty(
            num_tokens_per_forward[idx], dtype=torch.int32, device=device
        )
        create_chunked_cache_kv_indices_paged[(batch_size,)](
            page_table,
            chunks.starts[idx],
            chunks.len_in_chunk[idx],
            chunks.cum_seq_lens[idx],
            chunk_kv_indices,
            page_table.shape[1],
            page_size,
        )
        chunk_kv_indices_list.append(chunk_kv_indices)

    return chunks, chunk_kv_indices_list, chunks_cpu


def build_chunked_prefill_metadata_arrays(
    extend_prefix_lens,
    extend_prefix_lens_cpu,
    page_table,
    page_size,
):
    """Build the per-prefix-loop arrays for chunked-prefill MLA.

    Run once per chunked-prefill iteration in the backend's
    ``_init_prefill_metadata``; ``page_table`` is the batch-ordered
    ``[num_extends, max_pages]`` kernel page table. Returns:

    - ``chunked_loop_num``: number of prefix loop iterations
    - ``chunk_kv_indices_list``: List[Tensor], paged KV indices per loop_idx
    - ``chunked_seq_len``: (chunked_loop_num, num_extends) int32 GPU — per-seq
      KV length within each loop_idx (zero for seqs whose prefix doesn't
      reach this chunk).
    - ``cu_chunked_seq_len``: (chunked_loop_num, num_extends+1) int32 GPU —
      cumsum along the seq dim, fed to the chunker as ``cum_seq_lens_kv``.
    - ``max_chunk_len_per_loop``: List[int], CPU max-seq-len per loop_idx,
      fed to the chunker as ``max_kv_len``.
    - ``chunk_bounds_cpu``: (chunked_loop_num+1, num_extends) CPU positions,
      shared with the distributed history gather.

    The q-side cumsum / max do not appear here: callers alias them to the
    causal pass's ``cum_extend_seq_lens`` / ``max_extend_seq_len``, since
    every prefix-chunk forward sees the same ``q_lens == extend_seq_lens``.
    """
    chunks, chunk_kv_indices_list, chunks_cpu = get_chunks_paged(
        extend_prefix_lens,
        extend_prefix_lens_cpu,
        page_table,
        page_size,
    )
    chunked_loop_num = chunks.starts.shape[0]
    max_chunk_len_per_loop = [
        chunks_cpu.len_in_chunk[i].max().item() for i in range(chunked_loop_num)
    ]
    return (
        chunked_loop_num,
        chunk_kv_indices_list,
        chunks.len_in_chunk,
        chunks.cum_seq_lens,
        max_chunk_len_per_loop,
        chunks_cpu.boundaries,
    )
