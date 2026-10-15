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

"""Low-level Ascend BF16 DSA operator adapters."""

from __future__ import annotations

import os

import torch
import torch_npu
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _scatter(
    X,
    Y,
    Loc,
    WIDTH: tl.constexpr,
    CAP: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    loc = tl.load(Loc + row).to(tl.int64)
    x = tl.load(X + row * WIDTH + off, off < WIDTH, other=0)
    # Null page padding must not race on or corrupt live cache data.
    tl.store(Y + loc * WIDTH + off, x, (off < WIDTH) & (loc >= PAGE) & (loc < CAP))


def interleave_rope(tensor, positions, cos_sin_cache, *, rope_dim):
    """Run the one-sided interleaved RoPE used by FluentLLM's indexer."""
    if tensor.shape[0] == 0:
        return tensor
    rotary = tensor[..., :rope_dim]
    tail = tensor[..., rope_dim:]
    cos_sin = cos_sin_cache.index_select(0, positions.flatten())
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.repeat(1, 2).view(-1, 1, 1, rope_dim)
    sin = sin.repeat(1, 2).view(-1, 1, 1, rope_dim)
    rotated = torch_npu.npu_interleave_rope(
        rotary.reshape(rotary.shape[0], -1, 1, rope_dim),
        cos,
        sin,
    ).reshape_as(rotary)
    if tail.shape[-1] == 0:
        return rotated
    return torch.cat((rotated, tail), dim=-1)


class AscendDSAKernels:
    def __init__(self):
        import flash_ops

        library = os.environ.get("TOKENSPEED_LONGCAT_DSA_LIBRARY")
        if library:
            torch.ops.load_library(library)
        self._index = getattr(torch.ops.custom, "npu_lightning_indexer", None)
        self._attention = getattr(
            torch.ops.custom, "npu_sparse_flash_attention_decode", None
        )
        self._select_local = getattr(
            torch.ops.custom, "npu_select_local_topk_indices", None
        )
        self._merge_partials = getattr(
            torch.ops.custom, "npu_kvp_attention_merge", None
        )
        if self._index is None or self._attention is None:
            raise RuntimeError(
                "LongCatDSA needs LightningIndexer and "
                "SparseFlashAttentionDecode operators"
            )
        self._full_scan_chunks = {}

    interleave_rope = staticmethod(interleave_rope)

    def require_context_parallel(self):
        missing = []
        if self._select_local is None:
            missing.append("SelectLocalTopkIndices")
        if self._merge_partials is None:
            missing.append("KvpAttentionMerge")
        if missing:
            raise RuntimeError(
                "Ascend DSA context parallelism needs " + ", ".join(missing)
            )

    def _full_scan_chunk_hints(self, indices):
        """Stable scheduling hints for SFAD's full 2048-entry scan.

        LightningIndexer's second output is optional sparse values, not the
        ``valid_chunks`` emitted by FluentLLM's later KVP-local selection.
        This path has no KVP compaction and uses ``sparse_prep_mode=0``, so
        SFAD scans all entries and only needs a conservative scheduling cost.
        Cache the all-16 hint by graph shape to avoid adding a runtime kernel.
        """
        key = (indices.device, indices.shape[0])
        chunks = self._full_scan_chunks.get(key)
        if chunks is None:
            chunks = torch.full(
                (indices.shape[0],),
                indices.shape[-1] // 128,
                dtype=torch.int32,
                device=indices.device,
            )
            self._full_scan_chunks[key] = chunks
        return chunks

    def scatter(self, x, cache, locations):
        width = x.shape[-1]
        if x.dtype != torch.bfloat16 or cache.dtype != x.dtype:
            raise ValueError("LongCatDSA scatter requires BF16")
        if not x.is_contiguous() or not cache.is_contiguous():
            raise ValueError("LongCatDSA cache/input must be contiguous")
        if x.numel() // width != locations.numel():
            raise ValueError("LongCatDSA write locations must cover all tokens")
        _scatter[(locations.numel(),)](
            x,
            cache,
            locations,
            width,
            cache.numel() // width,
            cache.shape[1],
            triton.next_power_of_2(width),
        )

    def index(
        self, query, key, weights, q_ends, kv_lengths, table, topk, initial, local
    ):
        initial_tensor = torch.full_like(q_ends, initial, dtype=torch.int32)
        local_tensor = torch.full(
            (query.shape[0],), local, dtype=torch.int32, device=query.device
        )
        result, _ = self._index(
            query.contiguous(),
            key,
            weights.to(torch.bfloat16).contiguous(),
            actual_seq_lengths_query=q_ends,
            actual_seq_lengths_key=kv_lengths,
            block_table=table,
            init_tensor=initial_tensor,
            local_tensor=local_tensor,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=topk,
            sparse_mode=3,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            return_value=False,
        )
        return result, self._full_scan_chunk_hints(result)

    def index_partial(
        self,
        query,
        key,
        weights,
        q_ends,
        kv_lengths,
        table,
        topk,
        initial_counts,
        local_counts,
        *,
        sparse_mode,
    ):
        result, values = self._index(
            query.contiguous(),
            key,
            weights.to(torch.bfloat16).contiguous(),
            actual_seq_lengths_query=q_ends,
            actual_seq_lengths_key=kv_lengths,
            block_table=table,
            init_tensor=initial_counts,
            local_tensor=local_counts,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=topk,
            sparse_mode=sparse_mode,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            return_value=True,
        )
        return result, values

    def select_local(self, local_indices, global_positions, rank):
        if self._select_local is None:
            raise RuntimeError("LongCat DSA CP needs SelectLocalTopkIndices")
        return self._select_local(local_indices, global_positions, rank)

    def attention(
        self,
        q,
        qr,
        key,
        indices,
        valid_chunks,
        q_ends,
        kv_lengths,
        table,
        scale,
    ):
        output, _, _ = self._attention(
            q,
            key,
            key,
            indices,
            scale,
            valid_chunks=valid_chunks,
            block_table=table,
            actual_seq_lengths_query=q_ends,
            actual_seq_lengths_kv=kv_lengths,
            query_rope=qr,
            key_rope=None,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            attention_mode=2,
            return_softmax_lse=False,
            sparse_prep_mode=0,
        )
        return output.to(q.dtype)

    def attention_partial(
        self,
        packed_query,
        key,
        indices,
        valid_chunks,
        q_ends,
        kv_lengths,
        table,
        scale,
        *,
        sparse_mode,
    ):
        output, softmax_max, softmax_sum = self._attention(
            packed_query,
            key,
            key,
            indices,
            scale,
            valid_chunks=valid_chunks,
            block_table=table,
            actual_seq_lengths_query=q_ends,
            actual_seq_lengths_kv=kv_lengths,
            query_rope=None,
            key_rope=None,
            sparse_block_size=1,
            layout_query="RANK_MAJOR_TND_PACKED_KVP_COMM",
            layout_kv="PA_BSND",
            sparse_mode=sparse_mode,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            attention_mode=2,
            return_softmax_lse=True,
            sparse_prep_mode=1,
        )
        return output, softmax_max, softmax_sum

    def merge_partials(self, output, lse):
        if self._merge_partials is None:
            raise RuntimeError("LongCat DSA CP needs KvpAttentionMerge")
        return self._merge_partials(output, lse)
