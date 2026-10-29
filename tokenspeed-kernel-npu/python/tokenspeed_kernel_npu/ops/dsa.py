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

import ctypes as C
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


@triton.jit
def _current_page(token, seq_start, seq_end, prefix, total, degree, PAGE: tl.constexpr):
    page_start = token - (prefix + token - seq_start) % PAGE
    base, extra = total // degree, total % degree
    threshold = (base + 1) * extra
    owner = tl.where(
        page_start < threshold,
        page_start // (base + 1),
        extra + (page_start - threshold) // tl.maximum(base, 1),
    )
    owner_start = owner * base + tl.minimum(owner, extra)
    owner_end = owner_start + base + (owner < extra)
    direct = (
        (page_start >= seq_start)
        & (page_start + PAGE <= seq_end)
        & ((page_start - owner_start) % PAGE == 0)
        & (page_start + PAGE <= owner_end)
    )
    return owner, page_start - owner_start, direct


@triton.jit
def _prefill_metadata(
    Table,
    Cum,
    Prefix,
    OutTable,
    QCu,
    KCu,
    QuerySeq,
    Scan,
    Direct,
    batch,
    columns,
    table_stride,
    first,
    rows,
    total,
    per_block,
    degree,
    local_pages,
    send_rows,
    table_tiles,
    query_tiles,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    if pid < table_tiles:
        # A CTA owns one compact request row and a contiguous column tile.
        # No wrapped, masked scatter writes into another live request row.
        tiles_per_row = tl.cdiv(columns, BLOCK)
        page_compact = pid // tiles_per_row
        page_col = (pid % tiles_per_row) * BLOCK + off
        page_req = tl.full((), 0, tl.int32)
        page_seen = tl.full((), 0, tl.int32)
        for candidate in range(batch):
            candidate_start = tl.load(Cum + candidate).to(tl.int64)
            candidate_end = tl.load(Cum + candidate + 1).to(tl.int64)
            candidate_active = (
                (candidate_start < first + rows)
                & (candidate_end > first)
                & (candidate_end > candidate_start)
            )
            page_req = tl.where(
                candidate_active & (page_seen == page_compact), candidate, page_req
            )
            page_seen += candidate_active.to(tl.int32)
        page_id = tl.load(
            Table + page_req * table_stride + page_col, page_col < columns, other=0
        ).to(tl.int64)
        page_block = page_id // per_block
        page_positive = tl.maximum(page_block - 1, 0)
        page_owner = page_positive % degree
        page_local = (page_positive // degree + 1) * per_block + page_id % per_block
        page_mapped = tl.where(page_block > 0, page_owner * local_pages + page_local, 0)
        page_begin = tl.load(Cum + page_req).to(tl.int64)
        page_end = tl.load(Cum + page_req + 1).to(tl.int64)
        page_prefix = tl.load(Prefix + page_req).to(tl.int64)
        page_token = page_begin + page_col * PAGE - page_prefix
        tail_owner, tail_row, page_direct = _current_page(
            page_token, page_begin, page_end, page_prefix, total, degree, PAGE
        )
        tail_page = degree * local_pages + (tail_owner * send_rows + tail_row) // PAGE
        page_mapped = tl.where(page_direct, tail_page, page_mapped)
        tl.store(
            OutTable + page_compact * columns + page_col,
            page_mapped,
            page_col < columns,
        )
    elif pid == table_tiles:
        cu_destination = tl.full((), 1, tl.int32)
        cu_key_total = tl.full((), 0, tl.int64)
        for cu_req in range(batch):
            cu_start = tl.load(Cum + cu_req).to(tl.int64)
            cu_end = tl.load(Cum + cu_req + 1).to(tl.int64)
            cu_prefix = tl.load(Prefix + cu_req).to(tl.int64)
            cu_active = (
                (cu_start < first + rows) & (cu_end > first) & (cu_end > cu_start)
            )
            if cu_active:
                cu_local_end = tl.minimum(
                    tl.maximum(first + rows - cu_start, 0), cu_end - cu_start
                )
                cu_key_total += cu_prefix + cu_local_end
                tl.store(
                    QCu + cu_destination,
                    tl.minimum(tl.maximum(cu_end - first, 0), rows),
                )
                tl.store(KCu + cu_destination, cu_key_total)
                cu_destination += 1
        tl.store(QCu, 0)
        tl.store(KCu, 0)
    elif pid <= table_tiles + query_tiles:
        query_local = (pid - table_tiles - 1) * BLOCK + off
        query_token = first + query_local
        query_request = tl.full((BLOCK,), 0, tl.int32)
        query_length = tl.full((BLOCK,), 0, tl.int64)
        query_seen = tl.full((), 0, tl.int32)
        for seq_req in range(batch):
            seq_start = tl.load(Cum + seq_req).to(tl.int64)
            seq_end = tl.load(Cum + seq_req + 1).to(tl.int64)
            seq_prefix = tl.load(Prefix + seq_req).to(tl.int64)
            seq_match = (query_token >= seq_start) & (query_token < seq_end)
            query_request = tl.where(seq_match, query_seen, query_request)
            query_seen += (
                (seq_start < first + rows) & (seq_end > first) & (seq_end > seq_start)
            ).to(tl.int32)
            query_length = tl.where(
                seq_match, seq_prefix + query_token - seq_start + 1, query_length
            )
        tl.store(QuerySeq + query_local, query_request, query_local < rows)
        tl.store(Scan + query_local, tl.minimum(query_length, TOPK), query_local < rows)
    else:
        current_token = (pid - table_tiles - query_tiles - 1) * BLOCK + off
        current_direct = tl.full((BLOCK,), False, tl.int1)
        for current_req in range(batch):
            current_start = tl.load(Cum + current_req).to(tl.int64)
            current_end = tl.load(Cum + current_req + 1).to(tl.int64)
            current_prefix = tl.load(Prefix + current_req).to(tl.int64)
            _, _, can_alias = _current_page(
                current_token,
                current_start,
                current_end,
                current_prefix,
                total,
                degree,
                PAGE,
            )
            current_direct = current_direct | (
                can_alias
                & (current_token >= current_start)
                & (current_token < current_end)
            )
        tl.store(Direct + current_token, current_direct, current_token < total)


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

    @staticmethod
    def prepare_prefill(
        kv_cache, index_cache, *, ranks, bootstrap_group, max_current_rows
    ):
        return AscendDSAPrefill(
            kv_cache,
            index_cache,
            ranks=ranks,
            bootstrap_group=bootstrap_group,
            max_current_rows=max_current_rows,
        )

    @staticmethod
    def prefill_metadata(
        table,
        cumulative,
        prefixes,
        *,
        first,
        rows,
        total,
        per_block,
        degree,
        local_pages,
        send_rows,
        page_size,
        topk,
        active_requests,
    ):
        """Plan final history/current addresses from existing device metadata."""
        batch, columns = table.shape
        mapped = torch.empty(
            (active_requests, columns), dtype=torch.int32, device=table.device
        )
        qcu = torch.empty(active_requests + 1, dtype=torch.int64, device=table.device)
        kcu = torch.empty_like(qcu)
        sequence = torch.empty(rows, dtype=torch.int32, device=table.device)
        scan = torch.empty_like(sequence)
        direct = torch.empty(total, dtype=torch.uint8, device=table.device)
        block = 256
        table_tiles = active_requests * triton.cdiv(columns, block)
        query_tiles = triton.cdiv(rows, block)
        _prefill_metadata[(table_tiles + 1 + query_tiles + triton.cdiv(total, block),)](
            table,
            cumulative,
            prefixes,
            mapped,
            qcu,
            kcu,
            sequence,
            scan,
            direct,
            batch,
            columns,
            table.stride(0),
            first,
            rows,
            total,
            per_block,
            degree,
            local_pages,
            send_rows,
            table_tiles,
            query_tiles,
            TOPK=topk,
            PAGE=page_size,
            BLOCK=block,
        )
        return mapped, qcu, kcu, sequence, scan, direct

    def prefill_index(
        self,
        query,
        key,
        weights,
        query_cu,
        key_cu,
        table,
        topk,
        initial_tokens,
        window_tokens,
    ):
        """Index one contiguous query span per request against gathered KV pages."""
        indices, _ = torch.ops.custom.mlp_lightning_indexer(
            query.contiguous(),
            key,
            weights.float().contiguous(),
            cur_seq_lengths_query=query_cu,
            cur_seq_lengths_key=key_cu,
            block_table=table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=topk,
            kv_block_len=1,
            q_block_len=1,
            init_num=initial_tokens,
            local_num=window_tokens,
            sparse_mode=3,
            return_value=False,
        )
        return indices.reshape(query.shape[0], topk)

    def prefill_attention(
        self, query, pages, indices, scan_count, table, query_seq, scale
    ):
        """Consume packed BF16 pages and write contiguous [heads,tokens,512]."""
        return torch.ops.custom.npu_sparse_mla_attention_prefill(
            query.contiguous(),
            pages,
            indices,
            scan_count,
            table,
            query_seq,
            scale,
            True,
        )

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


class AscendDSAPrefill:
    """One fixed receive workspace and AICPU communicator for successive layers.

    Construction is collective on the supplied CPU bootstrap group, outside
    forward. Cache rows stay owner-sharded; both receive planes remain at fixed
    addresses and are consumed directly by the Indexer and sparse attention.
    """

    def __init__(
        self, kv_cache, index_cache, *, ranks, bootstrap_group, max_current_rows
    ):
        import torch.distributed as dist

        self.degree = len(ranks)
        self.rank = ranks.index(dist.get_rank())
        self.device = kv_cache.device
        self.local_rows = kv_cache.numel() // 576
        if self.degree < 1 or self.local_rows < 1:
            raise ValueError("DSA Prefill needs nonempty cache geometry")
        self._has_index = index_cache is not None
        planes = [(kv_cache, 576)]
        if self._has_index:
            planes.append((index_cache, 128))
        for value, width in planes:
            if (
                value.device != self.device
                or value.dtype != torch.bfloat16
                or not value.is_contiguous()
                or value.numel() != self.local_rows * width
            ):
                raise ValueError(
                    "DSA Prefill needs contiguous matching BF16 cache planes"
                )
        path = os.environ.get("TOKENSPEED_DSA_CP_LIBRARY")
        if not path:
            raise RuntimeError("Set TOKENSPEED_DSA_CP_LIBRARY to libdsa_cp_prefill.so")
        self.lib = C.CDLL(path)
        self.lib.DsaCpPrefillAbiVersion.restype = C.c_uint32
        if self.lib.DsaCpPrefillAbiVersion() != 2:
            raise RuntimeError("DSA CP native ABI does not match this runtime")
        self.lib.DsaCpRootInfoBytes.restype = C.c_uint64
        signatures = {
            "DsaCpGetRootInfo": [C.c_void_p],
            "DsaCpCreateComm": [
                C.c_void_p,
                C.c_uint32,
                C.c_uint32,
                C.POINTER(C.c_void_p),
            ],
            "DsaCpDestroyComm": [C.c_void_p],
            "DsaCpAllGather": [
                C.c_void_p,
                C.c_void_p,
                C.c_void_p,
                C.c_void_p,
                C.c_uint64,
            ],
            "DsaCpPatch": [C.c_void_p] * 6 + [C.c_uint64] * 4 + [C.c_uint32] * 4,
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = arguments
            function.restype = C.c_int
        self.comm = C.c_void_p()
        if self.degree > 1:
            root = C.create_string_buffer(self.lib.DsaCpRootInfoBytes())
            if self.rank == 0:
                self._check(self.lib.DsaCpGetRootInfo(root), "root info")
            objects = [root.raw if self.rank == 0 else None]
            dist.broadcast_object_list(objects, src=ranks[0], group=bootstrap_group)
            root = C.create_string_buffer(objects[0], len(objects[0]))
            self._check(
                self.lib.DsaCpCreateComm(
                    root, self.degree, self.rank, C.byref(self.comm)
                ),
                "create AICPU communicator",
            )
        if max_current_rows < 1:
            raise ValueError("DSA Prefill needs a positive current-token capacity")
        self.max_current_rows = max_current_rows
        self.send_capacity = (
            (max_current_rows + self.degree * 128 - 1) // (self.degree * 128)
        ) * 128
        self.history_rows = self.local_rows * self.degree
        rows = self.history_rows + self.send_capacity * self.degree
        index_width = 128 if self._has_index else 0
        self.storage = torch.empty(
            rows * (576 + index_width), dtype=torch.bfloat16, device=self.device
        )
        self.index = self.storage[: rows * index_width].view(rows, index_width)
        self.kv = self.storage[rows * index_width :].view(rows, 576)
        self.send_storage = torch.empty(
            self.send_capacity * (576 + index_width),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.send_index = self.send_storage[: self.send_capacity * index_width].view(
            self.send_capacity, index_width
        )
        self.send_kv = self.send_storage[self.send_capacity * index_width :].view(
            -1, 576
        )
        self.stream = torch.npu.Stream(device=self.device)
        self.index_ready = torch.npu.Event()
        self.kv_ready = torch.npu.Event()
        self.consumed = torch.npu.Event()
        self.source_ready = torch.npu.Event()
        self.index_produced = torch.npu.Event()
        self.kv_produced = torch.npu.Event()
        self.index_current_ready = torch.npu.Event()
        self.kv_current_ready = torch.npu.Event()
        self._current_rows = 0
        self._send_rows = 0
        self._started = {128: False, 576: False}
        self.layer_id = None
        self._consumed_recorded = False
        self._sources = None
        self._closed = False

    @staticmethod
    def _check(status, operation):
        if status:
            raise RuntimeError(f"DSA CP {operation} failed: {status}")

    @property
    def workspace_bytes(self):
        return (
            self.storage.numel() + self.send_storage.numel()
        ) * self.storage.element_size()

    def prefetch(self, layer_id, kv_cache, index_cache, *, has_history):
        """Enqueue both gathers directly into the prepared receive planes."""
        if self._closed or self.layer_id is not None:
            raise RuntimeError(
                "DSA Prefill workspace is closed or still being consumed"
            )
        if (index_cache is not None) != self._has_index:
            raise ValueError("Prefill index-cache presence changed after preparation")
        planes = [(kv_cache, 576)]
        if self._has_index:
            planes.append((index_cache, 128))
        for value, width in planes:
            if (
                value.dtype != torch.bfloat16
                or value.device != self.device
                or not value.is_contiguous()
                or value.numel() != self.local_rows * width
            ):
                raise ValueError("DSA Prefill cache geometry changed after preparation")
        self.layer_id = layer_id
        self._sources = (kv_cache, index_cache)
        self._started = {128: False, 576: False}
        self.source_ready.record(torch.npu.current_stream(self.device))
        with torch.npu.stream(self.stream):
            self.stream.wait_event(self.source_ready)
            if self._consumed_recorded:
                self.stream.wait_event(self.consumed)
            gathers = [(kv_cache, self.kv[: self.history_rows], self.kv_ready)]
            if self._has_index:
                gathers.insert(
                    0, (index_cache, self.index[: self.history_rows], self.index_ready)
                )
            for source, target, ready in gathers:
                source.record_stream(self.stream)
                target.record_stream(self.stream)
                if not has_history:
                    pass
                elif self.degree == 1:
                    target.copy_(source.view_as(target))
                else:
                    self._check(
                        self.lib.DsaCpAllGather(
                            self.comm,
                            self.stream.npu_stream,
                            source.data_ptr(),
                            target.data_ptr(),
                            source.numel(),
                        ),
                        "AICPU AllGather",
                    )
                ready.record(self.stream)

    def gather_current(self, layer_id, value, *, width, total_rows):
        """Start an incremental gather immediately after this plane's producer.

        Both phases use the same communication stream. Complete aligned pages
        are read directly from this receive region; only boundary pages need
        stitching into the historical page layout.
        """
        if (
            layer_id != self.layer_id
            or self._sources is None
            or width not in (128, 576)
            or (width == 128 and not self._has_index)
        ):
            raise RuntimeError("DSA Prefill current gather has no matching layer")
        if self._started[width] or not 0 <= total_rows <= self.max_current_rows:
            raise ValueError(
                "DSA Prefill current gather is repeated or exceeds capacity"
            )
        local_count = total_rows // self.degree + (self.rank < total_rows % self.degree)
        if (
            value.device != self.device
            or value.dtype != torch.bfloat16
            or not value.is_contiguous()
            or value.numel() != local_count * width
        ):
            raise ValueError(
                "Current DSA keys must contain only this rank's token rows"
            )
        send_rows = ((total_rows + self.degree * 128 - 1) // (self.degree * 128)) * 128
        if any(self._started.values()) and total_rows != self._current_rows:
            raise ValueError("DSA Prefill key planes disagree on current-token count")
        self._current_rows, self._send_rows = total_rows, send_rows
        target = self.index if width == 128 else self.kv
        receive = target[
            self.history_rows : self.history_rows + self.degree * send_rows
        ]
        producer = self.index_produced if width == 128 else self.kv_produced
        ready = self.index_current_ready if width == 128 else self.kv_current_ready
        source = value.view(-1, width)
        compute = torch.npu.current_stream(self.device)
        if local_count != send_rows:
            padded = (self.send_index if width == 128 else self.send_kv)[:send_rows]
            padded.zero_()
            padded[:local_count].copy_(source)
            source = padded
        producer.record(compute)
        with torch.npu.stream(self.stream):
            self.stream.wait_event(producer)
            for tensor in (value, source, target):
                tensor.record_stream(self.stream)
            if send_rows:
                if self.degree == 1:
                    receive.copy_(source)
                else:
                    self._check(
                        self.lib.DsaCpAllGather(
                            self.comm,
                            self.stream.npu_stream,
                            source.data_ptr(),
                            receive.data_ptr(),
                            source.numel(),
                        ),
                        "current AICPU AllGather",
                    )
            ready.record(self.stream)
        self._started[width] = True

    def patch(self, layer_id, slots, direct, *, width, block_rows, virtual_blocks):
        """Commit owner cache rows and stitch pages that cannot alias receives."""
        if (
            layer_id != self.layer_id
            or self._sources is None
            or not self._started[width]
        ):
            raise RuntimeError("DSA Prefill update has no matching current gather")
        if (
            slots.dtype not in (torch.int32, torch.int64)
            or not slots.is_contiguous()
            or slots.device != self.device
            or slots.numel() != self._current_rows
            or direct.dtype != torch.uint8
            or not direct.is_contiguous()
            or direct.device != self.device
            or direct.numel() != slots.numel()
            or block_rows <= 0
            or self.local_rows % block_rows
        ):
            raise ValueError("DSA Prefill update has incompatible slots or layout")
        # The router publishes int32 slots; the native transport ABI reads int64.
        slots = slots.to(torch.int64)
        stream = torch.npu.current_stream(self.device)
        target, cache, ready = (
            (self.index, self._sources[1], self.index_current_ready)
            if width == 128
            else (self.kv, self._sources[0], self.kv_current_ready)
        )
        # The communication stream already orders history before current.
        stream.wait_event(ready)
        for tensor in (slots, direct, target, cache):
            tensor.record_stream(stream)
        self._check(
            self.lib.DsaCpPatch(
                stream.npu_stream,
                target[self.history_rows :].data_ptr(),
                slots.data_ptr(),
                cache.data_ptr(),
                target.data_ptr(),
                direct.data_ptr(),
                slots.numel(),
                self.local_rows,
                virtual_blocks,
                self._send_rows,
                block_rows,
                self.degree,
                self.rank,
                width,
            ),
            "current-row commit",
        )
        return target

    def finish(self, layer_id):
        """Release the workspace after SFA, ordered by a device event."""
        if self.layer_id != layer_id:
            raise RuntimeError("DSA Prefill completion differs from its active layer")
        self.consumed.record(torch.npu.current_stream(self.device))
        self._consumed_recorded = True
        self.layer_id = None
        self._sources = None

    def close(self):
        """Drain owned work and release the communicator at shutdown."""
        if self._closed:
            return
        torch.npu.current_stream(self.device).synchronize()
        self.stream.synchronize()
        if self.comm.value:
            self._check(self.lib.DsaCpDestroyComm(self.comm), "destroy communicator")
            self.comm = C.c_void_p()
        self._closed = True
