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
import torch.distributed as dist
import torch_npu
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.prefill import AscendDSAPrefill


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
    HistoryTable,
    PatchSlots,
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
    history_pages,
    table_tiles,
    query_tiles,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
    PACKED_HISTORY: tl.constexpr,
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
        if PACKED_HISTORY:
            page_mapped = tl.load(
                HistoryTable + page_req * columns + page_col,
                page_col < columns,
                other=0,
            ).to(tl.int64)
        page_begin = tl.load(Cum + page_req).to(tl.int64)
        page_end = tl.load(Cum + page_req + 1).to(tl.int64)
        page_prefix = tl.load(Prefix + page_req).to(tl.int64)
        page_token = page_begin + page_col * PAGE - page_prefix
        tail_owner, tail_row, page_direct = _current_page(
            page_token, page_begin, page_end, page_prefix, total, degree, PAGE
        )
        tail_page = history_pages + (tail_owner * send_rows + tail_row) // PAGE
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
        patch_slot = tl.full((BLOCK,), -1, tl.int64)
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
            if PACKED_HISTORY:
                matches = (
                    (current_token >= current_start)
                    & (current_token < current_end)
                    & (current_token < total)
                )
                position = current_prefix + current_token - current_start
                page = tl.load(
                    HistoryTable + current_req * columns + position // PAGE,
                    matches,
                    other=0,
                ).to(tl.int64)
                patch_slot = tl.where(
                    matches & (page > 0), page * PAGE + position % PAGE, patch_slot
                )
        tl.store(Direct + current_token, current_direct, current_token < total)
        if PACKED_HISTORY:
            tl.store(PatchSlots + current_token, patch_slot, current_token < total)


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
        if self._index is None or self._attention is None:
            raise RuntimeError(
                "LongCatDSA needs LightningIndexer and "
                "SparseFlashAttentionDecode operators"
            )
        self._full_scan_chunks = {}

    @staticmethod
    def prepare_prefill(
        kv_cache,
        index_cache,
        *,
        ranks,
        history_group,
        max_current_rows,
        history_null_rows,
    ):
        return AscendDSAPrefill(
            kv_cache,
            index_cache,
            ranks=ranks,
            history_group=history_group,
            max_current_rows=max_current_rows,
            history_null_rows=history_null_rows,
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
        history_table=None,
        patch_slots=None,
        history_pages=None,
    ):
        """Plan final history/current addresses from existing device metadata."""
        batch, columns = table.shape
        if history_table is not None and (
            history_table.shape != table.shape
            or history_table.dtype != torch.int32
            or history_table.device != table.device
            or not history_table.is_contiguous()
            or patch_slots is None
            or patch_slots.shape != (total,)
            or patch_slots.dtype != torch.int64
            or patch_slots.device != table.device
            or not patch_slots.is_contiguous()
            or history_pages is None
            or history_pages < 1
        ):
            raise ValueError("Invalid compact Prefill page table or patch slots")
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
            history_table if history_table is not None else table,
            patch_slots if patch_slots is not None else direct,
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
            degree * local_pages if history_pages is None else history_pages,
            table_tiles,
            query_tiles,
            TOPK=topk,
            PAGE=page_size,
            BLOCK=block,
            PACKED_HISTORY=history_table is not None,
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

    def require_fused_prolog(self, *, index_heads):
        """Reject an older native ABI before preparing weights or graph buffers."""
        if index_heads not in (16, 32, 64):
            raise ValueError("Fused DSA requires Indexer16/32/64")
        try:
            scaled_prolog = torch.ops.custom.npu_dsa_prolog_fused.scaled_scores_out
        except AttributeError as error:
            raise RuntimeError(
                "Upgrade flash_ops: fused DSA requires explicit query/score scales"
            ) from error

        def meta(shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device="meta")

        hidden, heads, query_rank = (
            (2048, 32, 768) if index_heads == 16 else (4096, 64, 1536)
        )
        local_heads = heads // 8
        qb_columns = 393216 if index_heads == 16 else 1966080
        down, qb = meta((hidden, query_rank + 704)), meta((16, qb_columns))
        gq, gkv, gik = meta((query_rank,)), meta((512,)), meta((128,))
        weight, uk = meta((index_heads, hidden), torch.float32), meta((heads, 128, 512))
        window = meta((2 * 1024 * 1024,), torch.uint8)
        kv, ik = meta((1024, 576)), meta((1024, 128))
        for rows in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            _, _, score = torch.ops.custom.npu_dsa_prolog_fused(
                meta((rows, hidden)),
                down,
                gq,
                gkv,
                gik,
                weight,
                qb,
                uk,
                window,
                kv,
                ik,
                meta((rows,), torch.int64),
                1e-5,
                1e-5,
                1e-5,
                None,
                None,
                0,
            )
            score_dtype = torch.bfloat16 if index_heads == 64 else torch.float32
            if score.dtype != score_dtype:
                raise ValueError(
                    "Upgrade flash_ops: Indexer16/32 fused score must retain FP32"
                )
            scaled_prolog(
                meta((rows, hidden)),
                down,
                gq,
                gkv,
                gik,
                weight,
                qb,
                uk,
                window,
                kv,
                ik,
                meta((rows,), torch.int64),
                1e-5,
                1e-5,
                1e-5,
                1.0,
                1.0,
                None,
                None,
                0,
                query=meta((8, rows, local_heads, 576)),
                index_query=meta((rows, index_heads, 128)),
                score=meta((rows, index_heads), score_dtype),
            )
            torch.ops.custom.npu_dsa_output_projection_partial(
                meta((local_heads, rows, 512)),
                meta((local_heads, 512, 128)),
                meta((hidden, local_heads * 128)),
            )
            self.merge_topk_fused(
                meta((rows, 1, 2048), torch.int32),
                meta((rows, 1, 2048)),
                group="tokenspeed-dsa-contract",
                rank_size=8,
            )
            self.merge_lse_direct(
                meta((heads, rows, 512), torch.float32),
                meta((1, rows, heads), torch.float32),
                meta((1, rows, heads), torch.float32),
                group="tokenspeed-dsa-contract",
                rank_size=8,
            )

    def fused_prolog(
        self,
        x,
        down,
        gamma_q,
        gamma_kv,
        gamma_index,
        score_weight,
        qb,
        uk,
        window,
        kv_cache,
        index_cache,
        slots,
        *,
        eps_q,
        eps_kv,
        eps_index,
        query_scale,
        score_scale,
        positions,
        cos_sin,
        rotary_mode,
        out,
    ):
        """Fused projections and cache writes with operator-owned workspace."""
        return torch.ops.custom.npu_dsa_prolog_fused.scaled_scores_out(
            x,
            down,
            gamma_q,
            gamma_kv,
            gamma_index,
            score_weight,
            qb,
            uk,
            window,
            kv_cache,
            index_cache,
            slots,
            eps_q,
            eps_kv,
            eps_index,
            query_scale,
            score_scale,
            positions,
            cos_sin,
            rotary_mode,
            query=out[0],
            index_query=out[1],
            score=out[2],
        )

    def select_local(self, indices, positions, rank):
        from tokenspeed_kernel_npu.ops.dcp_tensors import select_local_topk

        return select_local_topk(indices, positions, rank=rank)

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

    def merge_lse_direct(self, output, maximum, total, *, group, rank_size):
        return torch.ops.custom.npu_dsa_merge_lse_direct(
            output, maximum, total, group, rank_size
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
        try:
            self.merge_topk_fused(indices, scores, group=group, rank_size=rank_size)
            self.merge_lse_direct(
                output,
                statistics,
                statistics,
                group=group,
                rank_size=rank_size,
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
            weights.float().contiguous(),
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
            weights.float().contiguous(),
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
    torch.ops.custom.npu_dsa_cp_metadata.out(
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
