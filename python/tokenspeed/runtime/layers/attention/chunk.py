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
import triton
import triton.language as tl

from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = get_colorful_logger(__name__)


@triton.jit
def create_chunked_cache_kv_indices_paged(
    page_table_ptr,  # (max_batch, max_pages)
    req_pool_indices_ptr,  # (batch_size,)
    chunk_start_idx_ptr,  # (batch_size,)
    chunk_seq_lens_ptr,  # rank-local (batch_size,)
    chunk_cum_seq_lens_ptr,  # rank-local (batch_size + 1,)
    chunk_kv_indices_ptr,  # (max rank-local tokens,)
    page_table_ptr_stride: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    chunk_kv_indices_offset = tl.load(chunk_cum_seq_lens_ptr + pid)

    chunk_start_pos = tl.load(chunk_start_idx_ptr + pid).to(tl.int32)
    chunk_seq_len = tl.load(chunk_seq_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(chunk_seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < chunk_seq_len
        first_owned = (
            chunk_start_pos + (DCP_RANK - chunk_start_pos % DCP_SIZE) % DCP_SIZE
        )
        token_pos = first_owned + offset * DCP_SIZE
        local_pos = token_pos // DCP_SIZE
        page_idx = local_pos // PAGE_SIZE
        page_id = tl.load(
            page_table_ptr + req_pool_index * page_table_ptr_stride + page_idx,
            mask=mask,
            other=0,
        )
        kv_slot = page_id * PAGE_SIZE + local_pos % PAGE_SIZE
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


def build_dcp_compact_reconstruction_plan(
    starts: list[int],
    lengths: list[int],
    dcp_size: int,
    dcp_rank: int,
) -> tuple[list[int], int, list[int]]:
    """Plan compact local prefix reads and global-order reconstruction.

    Returns this rank's per-request row counts, the common padded row count
    used by every rank's all-gather input, and indices that restore the
    gathered rank-major rows to request-major global token order.
    """
    counts_by_rank: list[list[int]] = []
    for rank in range(dcp_size):
        rank_counts = []
        for start, length in zip(starts, lengths):
            first_delta = (rank - int(start)) % dcp_size
            rank_counts.append(
                0
                if first_delta >= int(length)
                else 1 + (int(length) - 1 - first_delta) // dcp_size
            )
        counts_by_rank.append(rank_counts)
    padded_local_tokens = max(sum(counts) for counts in counts_by_rank)

    rank_req_offsets = []
    for counts in counts_by_rank:
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        rank_req_offsets.append(offsets)
    reconstruction = []
    for req, (start, length) in enumerate(zip(starts, lengths)):
        for token_pos in range(int(start), int(start) + int(length)):
            owner = token_pos % dcp_size
            first_owned = int(start) + (owner - int(start)) % dcp_size
            owner_offset = (token_pos - first_owned) // dcp_size
            reconstruction.append(
                owner * padded_local_tokens
                + rank_req_offsets[owner][req]
                + owner_offset
            )
    return counts_by_rank[dcp_rank], padded_local_tokens, reconstruction


# Here we suppose the length of each chunk is equal
# For example, if we have 4 sequences with seq length [256, 512, 768, 1024], chunk_len = 256
# num_chunks = cdiv(1024, 256) = 4
# chunk_starts = [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512], [768, 768, 768, 768]]
# chunk_ends = [[256, 256, 256, 256], [256, 512, 512, 512], [256, 512, 768, 768], [256, 512, 768, 1024]]
# chunk_seq_lens = [[256, 256, 256, 256], [0, 256, 256, 256], [0, 0, 256, 256], [0, 0, 0, 256]]
"""
        seq0 seq1 seq2 seq3
chunk0   --   --   --   --
chunk1   --   --   --   --
chunk2   --   --   --   --
chunk3   --   --   --   --
"""


# starts, ends, len_in_chunk, cum_seq_lens, all satisfy the above layout
@dataclass
class Chunks:
    starts: torch.Tensor
    ends: torch.Tensor
    len_in_chunk: torch.Tensor

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


def get_chunks_paged(
    prefix_lens,
    prefix_lens_cpu,
    page_table,
    req_pool_indices,
    page_size,
    *,
    dcp_size=1,
    dcp_rank=0,
):
    """Page-table aware version of get_chunks."""
    device: torch.device = prefix_lens.device
    batch_size = len(prefix_lens_cpu)

    chunk_capacity = get_max_chunk_capacity()
    chunk_len = chunk_capacity // batch_size
    max_prefix = prefix_lens_cpu.max().item()
    num_chunks = (max_prefix + chunk_len - 1) // chunk_len

    chunks = chunking(prefix_lens, num_chunks, batch_size, chunk_len)
    chunks_cpu = chunking(prefix_lens_cpu, num_chunks, batch_size, chunk_len)

    num_tokens_per_forward = chunks_cpu.len_in_chunk.sum(dim=1).tolist()

    chunk_kv_indices_list = []
    chunk_reconstruction_indices_list = []
    for idx in range(num_chunks):
        if dcp_size == 1:
            local_lens_cpu = chunks_cpu.len_in_chunk[idx]
            padded_local_tokens = num_tokens_per_forward[idx]
            reconstruction_indices = None
        else:
            starts = chunks_cpu.starts[idx].tolist()
            lengths = chunks_cpu.len_in_chunk[idx].tolist()
            local_counts, padded_local_tokens, reconstruction = (
                build_dcp_compact_reconstruction_plan(
                    starts, lengths, dcp_size, dcp_rank
                )
            )
            local_lens_cpu = torch.tensor(local_counts, dtype=torch.int32)
            reconstruction_indices = torch.tensor(
                reconstruction, dtype=torch.int64, device=device
            )

        local_lens = local_lens_cpu.to(device=device)
        local_cum_lens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        torch.cumsum(local_lens, dim=0, out=local_cum_lens[1:])
        chunk_kv_indices = torch.zeros(
            padded_local_tokens, dtype=torch.int32, device=device
        )
        create_chunked_cache_kv_indices_paged[(batch_size,)](
            page_table,
            req_pool_indices,
            chunks.starts[idx],
            local_lens,
            local_cum_lens,
            chunk_kv_indices,
            page_table.shape[1],
            page_size,
            DCP_SIZE=dcp_size,
            DCP_RANK=dcp_rank,
        )
        chunk_kv_indices_list.append(chunk_kv_indices)
        chunk_reconstruction_indices_list.append(reconstruction_indices)

    return (
        chunks,
        chunk_kv_indices_list,
        chunks_cpu,
        chunk_reconstruction_indices_list,
    )


def build_chunked_prefill_metadata_arrays(
    extend_prefix_lens,
    extend_prefix_lens_cpu,
    page_table,
    req_pool_indices,
    page_size,
    *,
    dcp_size=1,
    dcp_rank=0,
):
    """Build the per-prefix-loop arrays for chunked-prefill MLA.

    Run once per chunked-prefill iteration in the backend's
    ``_init_prefill_metadata``. Returns:

    - ``chunked_loop_num``: number of prefix loop iterations
    - ``chunk_kv_indices_list``: List[Tensor], paged KV indices per loop_idx
    - ``chunk_reconstruction_indices_list``: maps gathered compact DCP rows
      back to global request-major order; ``None`` entries when DCP is off.
    - ``chunked_seq_len``: (chunked_loop_num, num_extends) int32 GPU — per-seq
      KV length within each loop_idx (zero for seqs whose prefix doesn't
      reach this chunk).
    - ``cu_chunked_seq_len``: (chunked_loop_num, num_extends+1) int32 GPU —
      cumsum along the seq dim, fed to the chunker as ``cum_seq_lens_kv``.
    - ``max_chunk_len_per_loop``: List[int], CPU max-seq-len per loop_idx,
      fed to the chunker as ``max_kv_len``.

    The q-side cumsum / max do not appear here: callers alias them to the
    causal pass's ``cum_extend_seq_lens`` / ``max_extend_seq_len``, since
    every prefix-chunk forward sees the same ``q_lens == extend_seq_lens``.
    """
    (
        chunks,
        chunk_kv_indices_list,
        chunks_cpu,
        chunk_reconstruction_indices_list,
    ) = get_chunks_paged(
        extend_prefix_lens,
        extend_prefix_lens_cpu,
        page_table,
        req_pool_indices,
        page_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )
    chunked_loop_num = chunks.starts.shape[0]
    max_chunk_len_per_loop = [
        chunks_cpu.len_in_chunk[i].max().item() for i in range(chunked_loop_num)
    ]
    return (
        chunked_loop_num,
        chunk_kv_indices_list,
        chunk_reconstruction_indices_list,
        chunks.len_in_chunk,
        chunks.cum_seq_lens,
        max_chunk_len_per_loop,
    )
