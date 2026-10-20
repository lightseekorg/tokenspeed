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
        if self._index is None or self._attention is None:
            raise RuntimeError(
                "LongCatDSA needs LightningIndexer and "
                "SparseFlashAttentionDecode operators"
            )
        self._full_scan_chunks = {}
        self._identity_rotary = {}

    interleave_rope = staticmethod(interleave_rope)

    @staticmethod
    def prepare_projection_weight(value):
        """Copy a projection carrier into fresh NPU NZ29 storage.

        The returned tensor preserves the input shape, dtype and coefficients;
        it owns storage independent of the input. Internal-format allocation
        is disabled again when preparation finishes.
        """
        import torch_npu

        # Allocate a fresh 2D storage descriptor before the NZ conversion.
        plain = torch.empty(tuple(value.shape), dtype=value.dtype, device="npu")
        plain.copy_(value)
        # This TorchNPU option has a setter but no readable attribute.
        # The existing weight loader leaves it disabled after preparing NZ.
        try:
            torch.npu.config.allow_internal_format = True
            result = torch_npu.npu_format_cast(plain, 29)
            if torch_npu.get_npu_format(result) != 29:
                raise RuntimeError(
                    "DSA projection weight conversion did not produce NZ29"
                )
            return result
        finally:
            torch.npu.config.allow_internal_format = False

    def fused_prolog(self, *args, rotary=None):
        """Fused projections/store, optionally GPT-J RoPE (MLA=1, Indexer=2).

        ``rotary`` carries int64 positions, an FP32 [P,64] cos/sin table,
        and the explicit bit mask. Omitting it leaves projections unrotated.
        """
        if rotary is None:
            x = args[0]
            key = (x.device, x.shape[0])
            if key not in self._identity_rotary:
                self._identity_rotary[key] = (
                    torch.zeros(x.shape[0], dtype=torch.int64, device=x.device),
                    torch.zeros((1, 64), dtype=torch.float32, device=x.device),
                    0,
                )
            rotary = self._identity_rotary[key]
        return torch.ops.custom.npu_dsa_prolog_fused(*args, *rotary)

    def initialize_prolog_window(self, rank, rendezvous, *, rank_size):
        name = "dsa_prolog_fused"
        torch.ops.custom.initialize_group_first_rdma(
            name, rank, rank_size, rendezvous, 32 << 20
        )
        return torch.ops.custom.dsa_prolog_window(name)

    def finalize_prolog_window(self):
        torch.ops.custom.finalize_group_first_rdma("dsa_prolog_fused")

    def select_local(self, indices, positions, rank):
        if self._select_local is None:
            raise RuntimeError("Ascend DSA CP requires SelectLocalTopkIndices")
        return self._select_local(indices, positions, rank)

    def merge_partials(self, output, lse):
        """Merge normalized O[P,N,D] with natural-log LSE[P,N].

        Callers must zero O for every empty shard (LSE=-inf) before exchange.
        All-empty rows use finite equal merge weights and return zero; other
        rows retain their LSE weights. Return BF16 output with shape [N,D].
        """
        from tokenspeed_kernel_npu.ops.mla_dcp import merge_dcp_partials

        return merge_dcp_partials(output, lse)

    def merge_topk_fused(self, indices, values, *, group, rank_size):
        return torch.ops.custom.npu_dcp_merge_topk(
            indices, values, group=group, rank_size=rank_size, sorted_input=True
        )

    def merge_lse_direct(
        self, output, maximum, total, peer_inputs, *, group, rank_size, setup_mode
    ):
        return torch.ops.custom.npu_dsa_merge_lse_direct(
            output, maximum, total, peer_inputs, group, rank_size, setup_mode
        )

    def output_projection_partial(self, output, w_vc, w_o):
        return torch.ops.custom.npu_dsa_output_projection_partial(output, w_vc, w_o)

    def require_fused_context_parallel(self, *, rank_size, num_heads):
        """Validate topology against the installed operators without device work."""
        missing = [
            name
            for name in (
                "npu_dcp_merge_topk",
                "npu_dsa_merge_lse_direct",
                "npu_select_local_topk_indices",
                "npu_kvp_attention_merge",
            )
            if getattr(torch.ops.custom, name, None) is None
        ]
        if missing:
            raise RuntimeError(
                "Ascend DSA CP requires fused operators: " + ", ".join(missing)
            )
        # Meta executes the native ABI's shape/topology checks without allocating
        # NPU buffers or touching a communicator. Keep its limits at that boundary.
        rows = 64
        group = "tokenspeed-dsa-contract"
        indices = torch.empty((rows, 1, 2048), dtype=torch.int32, device="meta")
        scores = torch.empty_like(indices, dtype=torch.bfloat16)
        output = torch.empty((num_heads, rows, 512), dtype=torch.float32, device="meta")
        statistics = torch.empty(
            (1, rows, num_heads), dtype=torch.float32, device="meta"
        )
        peers = torch.empty(rank_size, dtype=torch.int64, device="meta")
        try:
            self.merge_topk_fused(indices, scores, group=group, rank_size=rank_size)
            self.merge_lse_direct(
                output,
                statistics,
                statistics,
                peers,
                group=group,
                rank_size=rank_size,
                setup_mode=0,
            )
        except NotImplementedError as exc:
            raise RuntimeError(
                "Installed Ascend DSA fused operators provide no Meta "
                "implementation; upgrade the flash_ops binding"
            ) from exc
        except RuntimeError as exc:
            raise ValueError(
                f"Installed Ascend DSA fused operators reject CP{rank_size} "
                f"with {num_heads} heads: {exc}"
            ) from exc

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
        visible_lengths=None,
    ):
        optional = {}
        if visible_lengths is not None:
            optional["visible_lengths"] = visible_lengths
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
            **optional,
        )
        return result, values

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
        out=None,
    ):
        operation = self._attention if out is None else self._attention.out
        outputs = (
            {}
            if out is None
            else dict(out=out[0], softmax_max=out[1], softmax_sum=out[2])
        )
        output, softmax_max, softmax_sum = operation(
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
            **outputs,
        )
        return output, softmax_max, softmax_sum


def dsa_cp_metadata(
    table,
    lengths,
    *,
    page_size,
    block_granularity,
    virtual_block_count,
    degree,
    rank,
    initial_tokens,
    window_tokens,
    query_width,
    query_group_size,
    slots,
    out,
    local_lengths,
    initial_counts,
    window_counts,
    query_ends,
    index_table,
    index_lengths,
    index_initial_counts,
    index_query_ends,
    local_slots,
):
    """Write causal query metadata, grouped Indexer metadata and local slots.

    Let ``B, P = table.shape``, ``R = B * query_width`` and
    ``N = R // query_group_size``. All buffers live on ``table.device``.

    Args:
        table: INT32 kernel-page IDs, shape [B, P], with contiguous columns.
        lengths: Contiguous INT32 global sequence ends, shape [B].
        page_size: Tokens in one kernel page.
        block_granularity: Tokens in one ownership block; a positive multiple
            of page_size. Block zero is reserved and has no owner.
        virtual_block_count: Ownership-block capacity, including block zero.
        degree: Number of ranks sharing the cache.
        rank: This rank's group-relative index, in [0, degree).
        initial_tokens: Global prefix extent whose owned tokens are counted.
        window_tokens: Trailing global extent whose owned tokens are counted.
        query_width: Positive queries per request. For width W > 1, candidate i
            sees max(1, lengths[b] - W + 1 + i); width one preserves lengths.
        query_group_size: Positive contiguous query rows per Indexer group;
            must divide R. Each group takes its last row's page/length metadata.
        slots: Optional contiguous INT32/INT64 global token-slot addresses.
            Its shape is independent of the query metadata row count.
        out: INT32 [R, P] compact local page tables, with unused entries zeroed.
        local_lengths: INT32 [R] visible owned-token counts per query.
        initial_counts: INT32 [R] visible owned tokens in the initial prefix.
        window_counts: INT32 [R] visible owned tokens in the trailing window.
        query_ends: INT32 [R] cumulative query ends, 1 through R.
        index_table: INT32 [N, P] grouped local page tables.
        index_lengths: INT32 [N] grouped visible owned-token counts.
        index_initial_counts: INT32 [N] grouped initial-prefix counts.
        index_query_ends: INT32 [N] cumulative ends G, 2G, ..., R, where
            G = query_group_size.
        local_slots: Optional contiguous INT64 buffer matching slots.shape.
            Owned addresses are translated to local storage; null, invalid or
            foreign slots become -1. Pass both slot buffers as None to omit them.

    Returns:
        None. Caller-provided output buffers are overwritten in place. They
        must be contiguous, must not overlap inputs or one another, and must
        remain alive throughout any graph that captures this launch.
    """
    import flash_ops  # noqa: F401

    if page_size <= 0 or block_granularity % page_size:
        raise ValueError("DSA ownership blocks must contain whole kernel pages")
    torch.ops.custom.npu_kvp_metadata.dsa_out(
        table,
        lengths,
        page_size,
        block_granularity // page_size,
        virtual_block_count,
        degree,
        rank,
        initial_tokens,
        window_tokens,
        query_width,
        query_group_size,
        slots,
        out=out,
        local_lengths=local_lengths,
        initial_counts=initial_counts,
        window_counts=window_counts,
        query_ends=query_ends,
        index_table=index_table,
        index_lengths=index_lengths,
        index_initial_counts=index_initial_counts,
        index_query_ends=index_query_ends,
        local_slots=local_slots,
    )
