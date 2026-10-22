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

"""Ascend V4.1 KV codecs, metadata, index selection, and attention dispatch.

Registers NPU implementations behind the shared attention API. Index query
collectives and shared validation stay at this boundary; device attention
execution and arena constants live in tokenspeed-kernel-npu.
"""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def create_paged_context(arena):
    """Create pool-owned constants for direct paged attention."""
    from tokenspeed_kernel_npu.ops.dsv41_attention import create_paged_context as impl

    return impl(arena)


if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops import dsv41_cache as cache_ops
    from tokenspeed_kernel_npu.ops.dsv41_attention import (
        rope_inplace,
        selected_attention,
    )
    from tokenspeed_kernel_npu.ops.dsv41_index_qdq import index_qdq

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))

    @register_kernel(
        "attention",
        "dsv41_decode_rows",
        name="ascend_dsv41_decode_rows",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.int32, torch.int64}),
        priority=Priority.PERFORMANT,
    )
    def decode_rows(
        seq_lens,
        request_pool_indices,
        positions,
        requests,
        lengths,
        pool_rows,
        actual_bs,
        num_extends,
        width,
    ):
        """Refresh the eager request/token layout, including idle warmup rows."""
        bs = lengths.numel()
        lengths.zero_()
        lengths[:actual_bs].copy_(seq_lens[:actual_bs].clamp_min(width))
        positions.view(bs, width).copy_(
            lengths.long()[:, None]
            - width
            + torch.arange(width, dtype=torch.int64, device=positions.device)
        )
        requests.view(bs, width).copy_(
            torch.arange(bs, dtype=torch.int64, device=positions.device)[:, None]
        )
        for tensor in (positions, requests):
            tensor[: num_extends * width].fill_(-1)
            tensor[actual_bs * width :].fill_(-1)
        pool_rows.fill_(-1)
        pool_rows[:actual_bs].copy_(request_pool_indices[:actual_bs])

    @register_kernel(
        "attention",
        "dsv41_compressor_metadata",
        name="ascend_dsv41_compressor_metadata",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.int32, torch.int64}),
        priority=Priority.PERFORMANT,
    )
    def compressor_metadata(
        positions,
        requests,
        table,
        pages,
        active,
        pair_positions,
        pair_requests,
        previous,
        read_slots,
        write_slots,
    ):
        """Build integer pair metadata over the existing LCM tables."""
        outputs = (
            active,
            pair_positions,
            pair_requests,
            previous,
            read_slots,
            write_slots,
        )
        if (
            0 <= pages < 2**25
            and table.shape[1] * 64 < 2**31
            and table.shape[0] * table.stride(0) < 2**31
            and all(t.is_contiguous() and t.data_ptr() % 32 == 0 for t in outputs)
        ):
            from tokenspeed_kernel_npu.ops.dsv41_window_metadata_native import (
                compressor_metadata as fused_metadata,
            )

            if fused_metadata(positions, requests, table, pages, *outputs):
                return
        from tokenspeed_kernel.ops.attention.mla._triton.page_table import (
            bounded_group_slots,
        )

        active.copy_((positions >= 0) & (positions % 2 == 1))
        pair_positions.copy_((positions - 1).masked_fill(~active, -1))
        pair_requests.copy_(requests.masked_fill(~active, -1))
        previous.fill_(-1)
        in_batch = (
            active[1:]
            & (requests[1:] >= 0)
            & (requests[:-1] == requests[1:])
            & (positions[:-1] == pair_positions[1:])
        )
        previous[1:].copy_(
            torch.arange(
                max(positions.numel() - 1, 0),
                dtype=torch.int64,
                device=positions.device,
            ).masked_fill(~in_batch, -1)
        )
        read_slots.copy_(
            bounded_group_slots(pair_positions, pair_requests, table, 2, 1, 1, pages)
        )
        write_slots.copy_(
            bounded_group_slots(positions, requests, table, 2, 1, 1, pages)
        )

    @register_kernel(
        "attention",
        "dsv41_decode_window",
        name="ascend_dsv41_decode_window",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.int32, torch.int64}),
        priority=Priority.PERFORMANT,
    )
    def decode_window(
        positions, requests, write_slots, read_slots, read_lens, swa_table, swa_pages
    ):
        """Refresh the caller's LCM window buffers with bounded device gathers."""
        outputs = (write_slots, read_slots, read_lens)
        if (
            0 <= swa_pages < 2**25
            and swa_table.shape[1] * 64 < 2**31
            and swa_table.shape[0] * swa_table.stride(0) < 2**31
            and all(t.is_contiguous() and t.data_ptr() % 32 == 0 for t in outputs)
        ):
            from tokenspeed_kernel_npu.ops.dsv41_window_metadata_native import (
                decode_window as fused_window,
            )

            if fused_window(positions, requests, *outputs, swa_table, swa_pages):
                return
        from tokenspeed_kernel.ops.attention.mla._triton.page_table import (
            bounded_group_slots,
        )

        wanted = (positions.long() - 127).clamp_min(0)[:, None] + torch.arange(
            128, device=positions.device, dtype=torch.int64
        )
        wanted.masked_fill_(wanted > positions[:, None], -1)
        read_slots.copy_(
            bounded_group_slots(
                wanted,
                requests[:, None].expand_as(wanted),
                swa_table,
                64,
                1,
                1,
                swa_pages,
            )
        )
        write_slots.copy_(
            bounded_group_slots(positions, requests, swa_table, 64, 1, 1, swa_pages)
        )
        read_lens.copy_((positions + 1).clamp(0, 128))

    register_kernel(
        "attention",
        "dsv41_rope_inplace",
        name="ascend_dsv41_rope_inplace",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(
            "values", "dense", {torch.bfloat16, torch.float16}
        ),
        priority=Priority.PORTABLE,
    )(rope_inplace)

    @register_kernel(
        "attention",
        "dsv41_rope_pad_query",
        name="ascend_dsv41_rope_pad_query",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(
            "values", "dense", {torch.bfloat16, torch.float16}
        ),
        priority=Priority.PORTABLE,
    )
    def rope_pad_query(values, positions, table):
        n, heads, dim = values.shape
        out = torch.zeros(
            (n, 64 if heads <= 64 else 128, dim),
            dtype=values.dtype,
            device=values.device,
        )
        out[:, :heads].copy_(values)
        rope_inplace(out[:, :heads], positions, table)
        return out

    for _name, _dtypes in (
        ("cache_pack", {torch.bfloat16, torch.float32}),
        ("cache_unpack", {torch.uint8, torch.bfloat16}),
        ("cache_scatter", {torch.bfloat16, torch.float32}),
        ("cache_gather", {torch.uint8, torch.bfloat16}),
    ):
        register_kernel(
            "attention",
            "dsv41_" + _name,
            name="ascend_dsv41_" + _name,
            solution="torch_npu",
            capability=_CAPABILITY,
            signatures=format_signatures("x", "dense", _dtypes),
            priority=Priority.PORTABLE,
        )(getattr(cache_ops, _name))

    @register_kernel(
        "attention",
        "dsv41_index_q_quantize",
        name="ascend_dsv41_index_q_quantize",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        priority=Priority.PORTABLE,
    )
    def index_q_quantize(q, cache_format, out):
        if cache_format != "index":
            raise ValueError(
                "Ascend V4.1 index quantization requires the original index format"
            )
        if q.ndim != 3 or q.shape[-1] != 128:
            raise ValueError("Index query must be [tokens,heads,128]")
        if q.numel() == 0:
            return torch.empty_like(q) if out is None else out
        result = index_qdq(q.reshape(-1, 128), None).reshape_as(q)
        if out is None:
            return result
        out.copy_(result)
        return out

    @register_kernel(
        "attention",
        "dsv41_swa_rope_scatter",
        name="ascend_dsv41_swa_rope_scatter",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        priority=Priority.PORTABLE,
    )
    def swa_rope_scatter(
        values,
        positions,
        cos_sin_cache,
        cache,
        slots,
        cache_format,
        out,
        norm_weight,
        norm_eps,
    ):
        from tokenspeed_kernel.ops.attention.dsv41 import rope_inplace

        if cache_format not in ("swa_bf16_qdq", "swa_bf16"):
            raise ValueError("Ascend SWA requires BF16 KV storage")
        if norm_weight is not None:
            from tokenspeed_kernel.ops.layernorm import reference_rmsnorm

            values = reference_rmsnorm(values, norm_weight, norm_eps, None)
        rotated = rope_inplace(values.clone(), positions, cos_sin_cache, None)
        if values.shape[0] == 0:
            return out
        if cache_format == "swa_bf16":
            from tokenspeed_kernel_npu.ops.dsv41_bf16_cache import cache_scatter_bf16

            cache_scatter_bf16(rotated, cache, slots)
            if out is not None:
                out.copy_(rotated)
            return out
        if cache_format == "swa_bf16_qdq":
            from tokenspeed_kernel_npu.ops.dsv41_bf16_cache import (
                cache_scatter_swa_bf16,
            )

            cache_scatter_swa_bf16(rotated, cache, slots, out=out)
            return out

    register_kernel(
        "attention",
        "dsv41_selected_attention",
        name="ascend_dsv41_selected_attention",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        priority=Priority.PERFORMANT,
        traits={"flashmla_eligible": frozenset({False})},
    )(selected_attention)

    @register_kernel(
        "attention",
        "dsv41_store_index_bf16",
        name="ascend_dsv41_store_index_bf16",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float32}),
        priority=Priority.PERFORMANT,
    )
    def store_index_bf16(rows, cache, slots, quantize):
        from tokenspeed_kernel_npu.ops.dsv41_index_bf16 import store_index

        return store_index(rows, cache, slots, quantize=quantize)

    @register_kernel(
        "attention",
        "dsv41_index_topk_bf16",
        name="ascend_dsv41_index_topk_bf16",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        priority=Priority.PERFORMANT,
    )
    def index_topk_bf16(
        index_q,
        weights,
        index_cache,
        page_table,
        visible_lens,
        candidate_blocks,
        topk,
        candidate_topk,
        candidate_block_size,
        query_chunk_size,
        score_chunk_size,
        process_group,
        out,
        quantize_query,
        prepared_reindex=None,
    ):
        from tokenspeed_kernel.ops.attention.dsv41.triton import (
            _index_gather_heads,
            _index_topk_outputs,
        )
        from tokenspeed_kernel_npu.ops.dsv41_index_device import (
            topk_tile as device_topk,
        )

        if index_cache.dtype != torch.bfloat16:
            raise ValueError("Ascend index selection requires BF16 cache storage")
        out = _index_topk_outputs(
            index_q,
            weights,
            index_cache,
            page_table,
            visible_lens,
            candidate_blocks,
            topk,
            candidate_topk,
            candidate_block_size,
            query_chunk_size,
            score_chunk_size,
            process_group,
            out,
        )
        # One registered implementation for every query count. Query tiling
        # bounds scratch only; it must not select a different scoring rule.
        tile = min(query_chunk_size, 192)
        if prepared_reindex is not None and (
            index_q.shape[0] > tile or candidate_blocks is None or candidate_topk
        ):
            raise ValueError("Prepared reindex requires one native query tile")
        for start in range(0, index_q.shape[0], tile):
            end = min(start + tile, index_q.shape[0])
            q = index_q[start:end]
            if quantize_query:
                q = index_q_quantize(q, "index", None)
            q, w, shards = _index_gather_heads(q, weights[start:end], process_group)
            device_topk(
                q,
                w,
                index_cache,
                page_table[start:end],
                visible_lens[start:end],
                None if candidate_blocks is None else candidate_blocks[start:end],
                topk,
                candidate_topk,
                score_chunk_size,
                shards,
                tuple(x[start:end] for x in out),
                **(
                    {"prepared_reindex": prepared_reindex}
                    if prepared_reindex is not None
                    else {}
                ),
            )
        return out

    @register_kernel(
        "attention",
        "dsv41_selection_table",
        name="ascend_dsv41_selection_table",
        solution="triton_ascend",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.int32, torch.int64}),
        priority=Priority.PERFORMANT,
    )
    def selection_table(positions, requests, table, ratio):
        from tokenspeed_kernel_npu.ops.dsv41_metadata import selection_table as impl

        return impl(positions, requests, table, ratio)

    @register_kernel(
        "attention",
        "dsv41_global_slots",
        name="ascend_dsv41_global_slots",
        solution="triton_ascend",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.int32, torch.int64}),
        priority=Priority.PERFORMANT,
    )
    def global_slots(rows, positions, requests, table, ratio, pages):
        from tokenspeed_kernel_npu.ops.dsv41_metadata import global_slots as impl

        return impl(rows, positions, requests, table, ratio, pages)

    @register_kernel(
        "attention",
        "dsv41_index_select_all",
        name="ascend_dsv41_index_select_all",
        solution="ascend",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.int32, torch.int64}),
        priority=Priority.PERFORMANT,
    )
    def index_select_all(page_table, visible_lens, pages, candidate_block_size, out):
        from tokenspeed_kernel_npu.ops.dsv41_index_device import select_all

        return select_all(page_table, visible_lens, pages, candidate_block_size, out)
