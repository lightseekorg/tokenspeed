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

"""Hybrid gfx950 KPool prefill selection with fused Gluon scoring.

Short single-window rows use the portable deterministic sort so equal scores
have stable pool-ID ordering. Long rows and their running merges use gfx950
Gluon radix top-k. Pool-ID payload gathering and pool-to-FlatKV expansion
remain portable Triton while those smaller stages are ported independently.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.triton.kpool_expand import (
    expand_kpool_to_flat_kv,
)
from tokenspeed_kernel.ops.attention.triton.kpool_score import (
    _kpool_sort_topk_kernel,
)
from tokenspeed_kernel.ops.attention.triton.kpool_select import (
    _DEFAULT_CHUNK_POOLS,
    _empty_result,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
    gluon_dsa_kpool_prefill_logits_gfx950,
    gluon_dsa_kpool_prefill_plan_logits_gfx950,
    gluon_dsa_logical_topk_gfx950,
)

_HEAD_DIM = 128
_POOL_SIZE = 4
_PAGE_SIZE = 16
_TOPK_POOLS = 512


def _validate_specialization(
    q: torch.Tensor,
    weights: torch.Tensor,
    *,
    pool_size: int,
    page_size: int,
    topk_pools: int,
    apply_relu: bool,
) -> None:
    if q.dim() != 3 or q.dtype != torch.bfloat16:
        raise ValueError("KPool queries must be [tokens, heads, dim] in bfloat16")
    if weights.dim() != 2 or weights.shape != q.shape[:2]:
        raise ValueError("KPool weights must match the query token and head axes")
    specialization = (
        int(q.shape[1]),
        int(q.shape[2]),
        int(pool_size),
        int(page_size),
        int(topk_pools),
        bool(apply_relu),
    )
    expected = (32, _HEAD_DIM, _POOL_SIZE, _PAGE_SIZE, _TOPK_POOLS, True)
    if specialization != expected:
        raise ValueError(
            "gfx950 Gluon KPool prefill requires "
            "index_heads=32, head_dim=128, pool_size=4, page_size=16, "
            "topk_pools=512, and apply_relu=True; "
            f"got {specialization}"
        )


def _normalized_window_width(
    max_num_pools: int,
    chunk_pools: int,
    topk_pools: int,
) -> int:
    chunk_pools = max(int(chunk_pools), int(topk_pools))
    chunk_pools = min(triton.next_power_of_2(chunk_pools), max(int(max_num_pools), 1))
    return triton.next_power_of_2(chunk_pools)


def _uses_deterministic_single_window_sort(
    *,
    max_num_pools: int,
    window_width: int,
) -> bool:
    return max_num_pools <= window_width <= 2048


@triton.jit
def _gather_gluon_topk_kernel(
    values,
    selected_positions,
    selected_lens,
    payloads,
    selected_values,
    selected_payloads,
    values_stride: tl.constexpr,
    selected_stride: tl.constexpr,
    payloads_stride: tl.constexpr,
    output_stride: tl.constexpr,
    payload_offset: tl.constexpr,
    HAS_PAYLOADS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    ranks = tl.arange(0, BLOCK)
    selected_len = tl.load(selected_lens + row).to(tl.int32)
    rank_valid = ranks < selected_len
    positions = tl.load(
        selected_positions + row * selected_stride + ranks,
        mask=rank_valid & (ranks < TOPK),
        other=0,
    ).to(tl.int32)
    position_valid = rank_valid & (ranks < TOPK) & (positions >= 0)
    selected = tl.load(
        values + row * values_stride + positions,
        mask=position_valid,
        other=-float("inf"),
    ).to(tl.float32)
    valid = position_valid
    if HAS_PAYLOADS:
        payload = tl.load(
            payloads + row * payloads_stride + positions,
            mask=valid,
            other=-1,
        ).to(tl.int32)
    else:
        payload = positions + payload_offset
    valid &= payload >= 0
    tl.store(
        selected_values + row * output_stride + ranks,
        tl.where(valid, selected, -float("inf")),
        mask=ranks < TOPK,
    )
    tl.store(
        selected_payloads + row * output_stride + ranks,
        tl.where(valid, payload, -1),
        mask=ranks < TOPK,
    )


def _gather_gluon_topk(
    values: torch.Tensor,
    positions: torch.Tensor,
    lens: torch.Tensor,
    *,
    topk: int,
    payload_offset: int = 0,
    payloads: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_values = torch.empty(
        (values.shape[0], topk), dtype=torch.float32, device=values.device
    )
    selected_payloads = torch.empty(
        (values.shape[0], topk), dtype=torch.int32, device=values.device
    )
    payload_arg = positions if payloads is None else payloads
    _gather_gluon_topk_kernel[(values.shape[0],)](
        values,
        positions,
        lens,
        payload_arg,
        selected_values,
        selected_payloads,
        values_stride=values.stride(0),
        selected_stride=positions.stride(0),
        payloads_stride=payload_arg.stride(0),
        output_stride=selected_values.stride(0),
        payload_offset=int(payload_offset),
        HAS_PAYLOADS=payloads is not None,
        TOPK=int(topk),
        BLOCK=triton.next_power_of_2(topk),
        num_warps=8,
        num_stages=1,
    )
    return selected_values, selected_payloads


def _select_pools_chunked_gluon(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    index_block_table: torch.Tensor,
    *,
    pool_workspace_slots: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    pool_size: int,
    page_size: int,
    topk_pools: int,
    softmax_scale: float,
    max_num_pools: int,
    chunk_pools: int,
    max_logits_bytes: int | None,
) -> torch.Tensor:
    """Score bounded windows and select candidates with stable short rows.

    Short selections that fit in one normalized window use logical head
    accumulation and deterministic sorting. Longer or split selections retain
    the faster balanced reduction and radix selection. Query rows are tiled so
    the fused path honors the runtime's temporary-workspace cap.
    """
    plan_parts = (pool_workspace_slots, row_starts, row_ends)
    has_prefill_plan = all(part is not None for part in plan_parts)
    if any(part is not None for part in plan_parts) and not has_prefill_plan:
        raise ValueError(
            "Gluon KPool scoring requires pool_workspace_slots, row_starts, "
            "and row_ends together"
        )
    num_tokens = q.shape[0]
    window_width = _normalized_window_width(
        max_num_pools,
        chunk_pools,
        topk_pools,
    )
    use_deterministic_sort = _uses_deterministic_single_window_sort(
        max_num_pools=max_num_pools,
        window_width=window_width,
    )
    result = torch.empty((num_tokens, topk_pools), dtype=torch.int32, device=q.device)
    if num_tokens == 0:
        return result

    # Include the persistent sort/radix intermediates, not only logits, when
    # deriving the row tile. One row remains legal when its workspace alone
    # exceeds the configured cap, matching the portable path.
    if use_deterministic_sort:
        workspace_columns = 3 * window_width
    else:
        workspace_columns = window_width + 10 * topk_pools
    workspace_row_bytes = workspace_columns * torch.float32.itemsize + 4
    if max_logits_bytes is None:
        rows_per_tile = num_tokens
    else:
        max_logits_bytes = int(max_logits_bytes)
        if max_logits_bytes <= 0:
            raise ValueError(
                f"max_logits_bytes must be positive, got {max_logits_bytes}"
            )
        rows_per_tile = min(
            num_tokens,
            max(1, max_logits_bytes // workspace_row_bytes),
        )

    logits_workspace = torch.empty(
        (rows_per_tile, window_width), dtype=torch.float32, device=q.device
    )
    local_row_ends_workspace = torch.empty(
        rows_per_tile, dtype=torch.int32, device=q.device
    )
    if use_deterministic_sort:
        sorted_vals_workspace = torch.empty_like(logits_workspace)
        sorted_pools_workspace = torch.empty(
            logits_workspace.shape, dtype=torch.int32, device=q.device
        )
        radix_row_starts_workspace = None
        selected_positions_workspace = None
        selected_lens_workspace = None
    else:
        sorted_vals_workspace = None
        sorted_pools_workspace = None
        radix_row_starts_workspace = torch.zeros(
            rows_per_tile, dtype=torch.int32, device=q.device
        )
        selected_positions_workspace = torch.empty(
            (rows_per_tile, topk_pools), dtype=torch.int32, device=q.device
        )
        selected_lens_workspace = torch.empty(
            rows_per_tile, dtype=torch.int32, device=q.device
        )

    for row_start in range(0, num_tokens, rows_per_tile):
        row_end = min(row_start + rows_per_tile, num_tokens)
        tile_rows = row_end - row_start
        tile_logits = logits_workspace[:tile_rows]
        tile_local_ends = local_row_ends_workspace[:tile_rows]
        best_vals: torch.Tensor | None = None
        best_pools: torch.Tensor | None = None

        for offset in range(0, max_num_pools, window_width):
            if has_prefill_plan:
                assert pool_workspace_slots is not None
                assert row_starts is not None
                assert row_ends is not None
                gluon_dsa_kpool_prefill_plan_logits_gfx950(
                    q[row_start:row_end],
                    pooled_k_cache,
                    weights[row_start:row_end],
                    pool_workspace_slots,
                    row_starts[row_start:row_end],
                    row_ends[row_start:row_end],
                    pool_size=pool_size,
                    page_size=page_size,
                    pool_offset=offset,
                    window_cols=window_width,
                    softmax_scale=softmax_scale,
                    ordered_head_fold=use_deterministic_sort,
                    out=tile_logits,
                    row_ends_out=tile_local_ends,
                )
            else:
                gluon_dsa_kpool_prefill_logits_gfx950(
                    q[row_start:row_end],
                    pooled_k_cache,
                    weights[row_start:row_end],
                    causal_lens[row_start:row_end],
                    req_ids[row_start:row_end],
                    index_block_table,
                    pool_size=pool_size,
                    page_size=page_size,
                    pool_offset=offset,
                    window_cols=window_width,
                    softmax_scale=softmax_scale,
                    ordered_head_fold=use_deterministic_sort,
                    out=tile_logits,
                    row_ends_out=tile_local_ends,
                )
            if use_deterministic_sort:
                assert sorted_vals_workspace is not None
                assert sorted_pools_workspace is not None
                tile_sorted_vals = sorted_vals_workspace[:tile_rows]
                tile_sorted_pools = sorted_pools_workspace[:tile_rows]
                _kpool_sort_topk_kernel[(tile_rows,)](
                    tile_logits,
                    tile_local_ends,
                    tile_sorted_vals,
                    tile_sorted_pools,
                    offset,
                    logits_stride=tile_logits.stride(0),
                    pool_size=pool_size,
                    chunk_pools=window_width,
                    LENS_ARE_POOL_COUNTS=True,
                    num_warps=8,
                )
                best_vals = tile_sorted_vals[:, :topk_pools]
                best_pools = tile_sorted_pools[:, :topk_pools]
                continue

            assert radix_row_starts_workspace is not None
            assert selected_positions_workspace is not None
            assert selected_lens_workspace is not None
            tile_radix_starts = radix_row_starts_workspace[:tile_rows]
            tile_selected_positions = selected_positions_workspace[:tile_rows]
            tile_selected_lens = selected_lens_workspace[:tile_rows]
            gluon_dsa_logical_topk_gfx950(
                tile_logits,
                tile_radix_starts,
                tile_local_ends,
                topk=topk_pools,
                out=tile_selected_positions,
                lens_out=tile_selected_lens,
            )
            vals, pools = _gather_gluon_topk(
                tile_logits,
                tile_selected_positions,
                tile_selected_lens,
                topk=topk_pools,
                payload_offset=offset,
            )
            if best_vals is None:
                best_vals, best_pools = vals, pools
                continue

            merged_vals = torch.cat((best_vals, vals), dim=1)
            merged_pools = torch.cat((best_pools, pools), dim=1)
            tile_local_ends.fill_(merged_vals.shape[1])
            gluon_dsa_logical_topk_gfx950(
                merged_vals,
                tile_radix_starts,
                tile_local_ends,
                topk=topk_pools,
                out=tile_selected_positions,
                lens_out=tile_selected_lens,
            )
            best_vals, best_pools = _gather_gluon_topk(
                merged_vals,
                tile_selected_positions,
                tile_selected_lens,
                topk=topk_pools,
                payloads=merged_pools,
            )

        if best_pools is None or best_vals is None:
            result[row_start:row_end].fill_(-1)
            continue
        best_pools = torch.where(
            torch.isfinite(best_vals), best_pools, torch.full_like(best_pools, -1)
        )
        if best_pools.shape[1] < topk_pools:
            best_pools = torch.nn.functional.pad(
                best_pools,
                (0, topk_pools - best_pools.shape[1]),
                value=-1,
            )
        result[row_start:row_end].copy_(best_pools)

    return result.contiguous()


def _kpool_prefill_topk_fp8_gfx950(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    index_block_table: torch.Tensor,
    kv_block_table: torch.Tensor,
    *,
    pool_size: int,
    page_size: int,
    kv_page_size: int,
    topk_pools: int,
    softmax_scale: float,
    apply_relu: bool = True,
    append_tail: bool = True,
    chunk_pools: int = _DEFAULT_CHUNK_POOLS,
    req_ids: torch.Tensor | None = None,
    causal_lens: torch.Tensor | None = None,
    pool_workspace_slots: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    max_num_pools: int | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select GLM KPool candidates through the fused GFX950 scorers.

    Complete runtime plans use physical-slot addressing. With no plan, request
    ids and causal lengths are reconstructed for the page-table scorer. Partial
    plans are rejected so the two addressing modes cannot be mixed.
    """
    _validate_specialization(
        q,
        weights,
        pool_size=pool_size,
        page_size=page_size,
        topk_pools=topk_pools,
        apply_relu=apply_relu,
    )

    num_tokens = q.shape[0]
    if query_start_loc.dim() != 1 or query_start_loc.numel() < 2:
        raise ValueError(
            f"query_start_loc must be [reqs + 1], got {tuple(query_start_loc.shape)}"
        )
    if positions.numel() != num_tokens:
        raise ValueError(
            f"positions must have {num_tokens} entries, got {positions.numel()}"
        )
    plan_parts = (
        req_ids,
        causal_lens,
        pool_workspace_slots,
        row_starts,
        row_ends,
        max_num_pools,
    )
    has_prefill_plan = all(part is not None for part in plan_parts)
    if any(part is not None for part in plan_parts) and not has_prefill_plan:
        raise ValueError(
            "gfx950 Gluon KPool prefill requires req_ids, causal_lens, "
            "pool_workspace_slots, row_starts, row_ends, and max_num_pools together"
        )
    if num_tokens == 0:
        return _empty_result(q, topk_pools, pool_size, append_tail)
    if not q.is_cuda:
        raise RuntimeError("KPool prefill top-k requires CUDA tensors.")

    device = q.device
    if has_prefill_plan:
        assert req_ids is not None
        assert causal_lens is not None
        assert pool_workspace_slots is not None
        assert row_starts is not None
        assert row_ends is not None
        assert max_num_pools is not None
        metadata = (
            ("req_ids", req_ids, torch.int32),
            ("causal_lens", causal_lens, torch.int32),
            ("row_starts", row_starts, torch.int32),
            ("row_ends", row_ends, torch.int32),
        )
        for name, value, dtype in metadata:
            if value.shape != (num_tokens,):
                raise ValueError(
                    f"{name} must have shape {(num_tokens,)}, got {tuple(value.shape)}"
                )
            if (
                value.dtype != dtype
                or value.device != device
                or not value.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous {dtype} on q.device")
        if (
            pool_workspace_slots.dim() != 1
            or pool_workspace_slots.dtype != torch.int64
            or pool_workspace_slots.device != device
            or not pool_workspace_slots.is_contiguous()
        ):
            raise ValueError(
                "pool_workspace_slots must be contiguous int64 on q.device"
            )
        max_num_pools = int(max_num_pools)
        if max_num_pools < 0:
            raise ValueError(f"max_num_pools must be nonnegative, got {max_num_pools}")
    else:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CUDA graph capture requires a complete KPool prefill plan; "
                "no-plan metadata reconstruction is eager-only"
            )
        query_start_loc = query_start_loc.to(
            device=device, dtype=torch.int32
        ).contiguous()
        query_lens = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.long)
        if int(query_lens.sum().item()) != num_tokens:
            raise ValueError("query_start_loc does not cover every query token")
        req_ids = torch.repeat_interleave(
            torch.arange(query_lens.numel(), device=device, dtype=torch.int32),
            query_lens,
        )
        causal_lens = (
            positions.to(device=device, dtype=torch.int32).add(1).clamp_min_(0)
        )
        max_num_pools = max(int(causal_lens.max().item()) // pool_size, 1)
        index_block_table = index_block_table.to(
            device=device, dtype=torch.int32
        ).contiguous()
    assert req_ids is not None
    assert causal_lens is not None
    assert max_num_pools is not None
    scoring_weights = weights
    if scoring_weights.dtype not in (torch.bfloat16, torch.float32):
        scoring_weights = scoring_weights.to(torch.float32)
    pool_indices = torch.empty(
        (num_tokens, topk_pools), dtype=torch.int32, device=device
    )
    if max_num_pools > topk_pools:
        pool_indices = _select_pools_chunked_gluon(
            q.contiguous(),
            pooled_k_cache,
            scoring_weights.contiguous(),
            causal_lens,
            req_ids,
            index_block_table,
            pool_workspace_slots=pool_workspace_slots,
            row_starts=row_starts,
            row_ends=row_ends,
            pool_size=pool_size,
            page_size=page_size,
            topk_pools=topk_pools,
            softmax_scale=softmax_scale,
            max_num_pools=max_num_pools,
            chunk_pools=chunk_pools,
            max_logits_bytes=max_logits_bytes,
        )
    return expand_kpool_to_flat_kv(
        pool_indices,
        causal_lens,
        req_ids,
        kv_block_table.to(device=device, dtype=torch.int32).contiguous(),
        pool_size=pool_size,
        kv_page_size=int(kv_page_size),
        append_tail=append_tail,
        out=out,
        lens_out=lens_out,
    )


def gluon_kpool_prefill_topk_fp8_gfx950(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    index_block_table: torch.Tensor,
    kv_block_table: torch.Tensor,
    *,
    pool_size: int,
    page_size: int,
    kv_page_size: int,
    topk_pools: int,
    softmax_scale: float,
    apply_relu: bool = True,
    append_tail: bool = True,
    chunk_pools: int = _DEFAULT_CHUNK_POOLS,
    req_ids: torch.Tensor | None = None,
    causal_lens: torch.Tensor | None = None,
    pool_workspace_slots: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    max_num_pools: int | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select KPool candidates with the production hybrid Gluon path.

    Short selections of at most 2048 pools use ordered head accumulation when
    they fit in one normalized scoring window. Longer or split selections keep
    the balanced reduction.
    """
    return _kpool_prefill_topk_fp8_gfx950(
        q,
        pooled_k_cache,
        weights,
        positions,
        query_start_loc,
        index_block_table,
        kv_block_table,
        pool_size=pool_size,
        page_size=page_size,
        kv_page_size=kv_page_size,
        topk_pools=topk_pools,
        softmax_scale=softmax_scale,
        apply_relu=apply_relu,
        append_tail=append_tail,
        chunk_pools=chunk_pools,
        req_ids=req_ids,
        causal_lens=causal_lens,
        pool_workspace_slots=pool_workspace_slots,
        row_starts=row_starts,
        row_ends=row_ends,
        max_num_pools=max_num_pools,
        max_logits_bytes=max_logits_bytes,
        out=out,
        lens_out=lens_out,
    )


__all__ = ["gluon_kpool_prefill_topk_fp8_gfx950"]
