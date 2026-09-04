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

"""Portable KPool scoring kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _kpool_score_prefill_chunk_kernel(
    q,
    pooled_k_fp8,
    pooled_k_scale,
    causal_lens,
    req_ids,
    block_table,
    head_logits,
    pool_offset,
    cache_page_stride_bytes: tl.constexpr,
    block_table_stride: tl.constexpr,
    head_logits_token_stride: tl.constexpr,
    head_logits_head_stride: tl.constexpr,
    chunk_pools: tl.constexpr,
    page_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_groups: tl.constexpr,
    pool_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)

    req = tl.load(req_ids + token).to(tl.int32)
    causal = tl.load(causal_lens + token).to(tl.int32)
    num_pools = causal // pool_size

    local = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    pools = pool_offset + local
    valid = (local < chunk_pools) & (pools < num_pools)

    page_idx = pools // page_size
    page_offset = pools - page_idx * page_size
    page = tl.load(
        block_table + req * block_table_stride + page_idx, mask=valid, other=0
    ).to(tl.int64)
    fp8_base = page * cache_page_stride_bytes + page_offset * head_dim
    scale_base = (
        page * (cache_page_stride_bytes // 4)
        + (page_size * head_dim) // 4
        + page_offset * num_groups
    )

    scores = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dim_offsets = tl.arange(0, BLOCK_D)
    for dim_start in tl.static_range(0, head_dim, BLOCK_D):
        dims = dim_start + dim_offsets
        q_vals = tl.load(
            q + (token * num_heads + head) * head_dim + dims,
            mask=dims < head_dim,
            other=0.0,
        ).to(tl.float32)
        k_vals = tl.load(
            pooled_k_fp8 + fp8_base[:, None] + dims[None, :],
            mask=valid[:, None] & (dims[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        k_scale = tl.load(
            pooled_k_scale + scale_base + dim_start // 128,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        scores += tl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)

    tl.store(
        head_logits
        + token * head_logits_token_stride
        + head * head_logits_head_stride
        + local,
        scores,
        mask=local < chunk_pools,
    )


@triton.jit
def _kpool_reduce_heads_kernel(
    head_logits,
    weights,
    causal_lens,
    logits,
    pool_offset,
    head_logits_token_stride: tl.constexpr,
    head_logits_head_stride: tl.constexpr,
    logits_stride: tl.constexpr,
    chunk_pools: tl.constexpr,
    num_heads: tl.constexpr,
    pool_size: tl.constexpr,
    softmax_scale: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    block_id = tl.program_id(1)
    local = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    pools = pool_offset + local
    num_pools = tl.load(causal_lens + token).to(tl.int32) // pool_size
    valid = (local < chunk_pools) & (pools < num_pools)

    scores = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for head in tl.static_range(0, num_heads):
        head_score = tl.load(
            head_logits
            + token * head_logits_token_stride
            + head * head_logits_head_stride
            + local,
            mask=local < chunk_pools,
            other=0.0,
        )
        head_score *= softmax_scale
        if APPLY_RELU:
            head_score = tl.maximum(head_score, 0.0)
        weight = tl.load(weights + token * num_heads + head).to(tl.float32)
        scores += head_score * weight

    scores = tl.where(valid, scores, -float("inf"))
    tl.store(logits + token * logits_stride + local, scores, mask=local < chunk_pools)


@triton.jit
def _kpool_score_dense_mma_kernel(
    q,
    pooled_k_fp8,
    pooled_k_scale,
    weights,
    causal_lens,
    req_ids,
    block_table,
    logits,
    max_num_pools,
    cache_page_stride_bytes: tl.constexpr,
    block_table_stride: tl.constexpr,
    logits_stride: tl.constexpr,
    page_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_groups: tl.constexpr,
    pool_size: tl.constexpr,
    softmax_scale: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    NUM_WORKERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Score visible pool tiles with a fixed set of persistent workers."""
    token = tl.program_id(0)
    req = tl.load(req_ids + token).to(tl.int32)
    causal = tl.load(causal_lens + token).to(tl.int32)
    num_pools = tl.maximum(0, tl.minimum(causal // pool_size, max_num_pools))
    num_pool_blocks = tl.cdiv(num_pools, BLOCK_M)
    heads = tl.arange(0, BLOCK_H)

    for pool_block in tl.range(
        tl.program_id(1), num_pool_blocks, NUM_WORKERS, num_stages=1
    ):
        pools = pool_block * BLOCK_M + tl.arange(0, BLOCK_M)
        valid_pool = pools < num_pools
        page_idx = pools // page_size
        page_offset = pools - page_idx * page_size
        page = tl.load(
            block_table + req * block_table_stride + page_idx,
            mask=valid_pool,
            other=0,
        ).to(tl.int64)
        fp8_base = page * cache_page_stride_bytes + page_offset * head_dim
        scale_base = (
            page * (cache_page_stride_bytes // 4)
            + (page_size * head_dim) // 4
            + page_offset * num_groups
        )

        head_scores = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
        dim_offsets = tl.arange(0, BLOCK_D)
        for group in tl.static_range(0, num_groups):
            dims = group * BLOCK_D + dim_offsets
            k_tile = tl.load(
                pooled_k_fp8 + fp8_base[:, None] + dims[None, :],
                mask=valid_pool[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            q_tile = tl.load(
                q + (token * num_heads + heads[:, None]) * head_dim + dims[None, :],
                mask=heads[:, None] < num_heads,
                other=0.0,
            )
            group_scale = tl.load(
                pooled_k_scale + scale_base + group,
                mask=valid_pool,
                other=0.0,
            ).to(tl.float32)
            head_scores += (
                tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.float32)
                * group_scale[:, None]
            )

        head_scores *= softmax_scale
        if APPLY_RELU:
            head_scores = tl.maximum(head_scores, 0.0)
        head_weights = tl.load(
            weights + token * num_heads + heads,
            mask=heads < num_heads,
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(head_scores * head_weights[None, :], axis=1)
        scores = tl.where(valid_pool, scores, -float("inf"))
        tl.store(logits + token * logits_stride + pools, scores, mask=valid_pool)


@triton.jit
def _kpool_sort_topk_kernel(
    logits,
    candidate_lens,
    values_out,
    indices_out,
    pool_offset,
    logits_stride: tl.constexpr,
    pool_size: tl.constexpr,
    chunk_pools: tl.constexpr,
    LENS_ARE_POOL_COUNTS: tl.constexpr = False,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, chunk_pools)
    candidate_len = tl.maximum(tl.load(candidate_lens + token), 0)
    if LENS_ARE_POOL_COUNTS:
        num_pools = tl.minimum(candidate_len, chunk_pools)
    else:
        num_pools = tl.minimum(
            candidate_len // pool_size - pool_offset,
            chunk_pools,
        )
    values = tl.load(logits + token * logits_stride + offsets)
    valid = offsets < num_pools
    values = tl.where(valid, values, -float("inf"))

    bits = values.to(tl.uint32, bitcast=True)
    sign = bits & 0x80000000
    value_keys = bits ^ tl.where(sign != 0, 0xFFFFFFFF, 0x80000000)
    index_keys = (chunk_pools - offsets).to(tl.uint64)
    packed = (value_keys.to(tl.uint64) << 32) | index_keys
    packed = tl.sort(packed[None, :], dim=1, descending=True)
    packed = tl.reshape(packed, (chunk_pools,))

    selected_indices = chunk_pools - (packed & 0xFFFFFFFF).to(tl.int32)
    selected_keys = (packed >> 32).to(tl.uint32)
    selected_sign = selected_keys & 0x80000000
    selected_bits = selected_keys ^ tl.where(selected_sign != 0, 0x80000000, 0xFFFFFFFF)
    selected_values = selected_bits.to(tl.float32, bitcast=True)
    selected_valid = offsets < num_pools
    tl.store(
        values_out + token * chunk_pools + offsets,
        tl.where(selected_valid, selected_values, -float("inf")),
    )
    tl.store(
        indices_out + token * chunk_pools + offsets,
        tl.where(selected_valid, selected_indices, -1),
    )


def select_kpool_chunked(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    index_block_table: torch.Tensor,
    *,
    pool_size: int,
    page_size: int,
    topk_pools: int,
    softmax_scale: float,
    apply_relu: bool,
    max_num_pools: int,
    chunk_pools: int,
    max_logits_bytes: int | None = None,
) -> torch.Tensor:
    """Select ragged-prefill pools with bounded score windows.

    Args:
        q: Packed index queries.
        pooled_k_cache: Packed paged FP8 KPool cache.
        weights: Per-query head weights.
        causal_lens: Visible raw-token lengths.
        req_ids: Request id for each query.
        index_block_table: Pooled-cache page table.
        pool_size: Raw tokens represented by one pool.
        page_size: Rows in each pooled-cache page.
        topk_pools: Pools selected per query.
        softmax_scale: Per-head score scale.
        apply_relu: Whether to apply the indexer ReLU.
        max_num_pools: Host-known pool bound.
        chunk_pools: Maximum pools scored per window.
        max_logits_bytes: Optional cap for score and sort workspaces. At least
            one query row is processed when a single row exceeds the cap.

    Returns:
        Request-local selected pool ids.
    """
    num_tokens, num_heads, head_dim = q.shape
    num_groups = head_dim // 128

    chunk_pools = max(int(chunk_pools), int(topk_pools))
    chunk_pools = min(triton.next_power_of_2(chunk_pools), max(max_num_pools, 1))
    chunk_pools = triton.next_power_of_2(chunk_pools)
    block_n = min(chunk_pools, 256)
    cache_fp8 = pooled_k_cache.view(torch.float8_e4m3fn)
    cache_fp32 = pooled_k_cache.view(torch.float32)
    result = torch.empty((num_tokens, topk_pools), dtype=torch.int32, device=q.device)
    if num_tokens == 0:
        return result

    use_device_sort = chunk_pools <= 2048 and max_num_pools >= chunk_pools
    workspace_columns = (num_heads + 1) * chunk_pools
    if use_device_sort:
        workspace_columns += 2 * chunk_pools
    workspace_row_bytes = workspace_columns * torch.float32.itemsize
    if max_logits_bytes is None:
        rows_per_tile = num_tokens
    else:
        max_logits_bytes = int(max_logits_bytes)
        if max_logits_bytes <= 0:
            raise ValueError(
                f"max_logits_bytes must be positive, got {max_logits_bytes}"
            )
        rows_per_tile = min(num_tokens, max(1, max_logits_bytes // workspace_row_bytes))

    logits_workspace = torch.empty(
        (rows_per_tile, chunk_pools), dtype=torch.float32, device=q.device
    )
    head_logits_workspace = torch.empty(
        (rows_per_tile, num_heads, chunk_pools),
        dtype=torch.float32,
        device=q.device,
    )
    sorted_vals_workspace = sorted_local_workspace = None
    if use_device_sort:
        sorted_vals_workspace = torch.empty_like(logits_workspace)
        sorted_local_workspace = torch.empty(
            logits_workspace.shape, dtype=torch.int32, device=q.device
        )

    for row_start in range(0, num_tokens, rows_per_tile):
        row_end = min(row_start + rows_per_tile, num_tokens)
        tile_rows = row_end - row_start
        tile_logits = logits_workspace[:tile_rows]
        tile_head_logits = head_logits_workspace[:tile_rows]
        tile_causal_lens = causal_lens[row_start:row_end]
        best_vals = best_pools = None

        for offset in range(0, max_num_pools, chunk_pools):
            span = min(chunk_pools, max_num_pools - offset)
            score_grid = (tile_rows, num_heads, triton.cdiv(span, block_n))
            _kpool_score_prefill_chunk_kernel[score_grid](
                q[row_start:row_end],
                cache_fp8,
                cache_fp32,
                tile_causal_lens,
                req_ids[row_start:row_end],
                index_block_table,
                tile_head_logits,
                offset,
                cache_page_stride_bytes=pooled_k_cache.stride(0),
                block_table_stride=index_block_table.stride(0),
                head_logits_token_stride=tile_head_logits.stride(0),
                head_logits_head_stride=tile_head_logits.stride(1),
                chunk_pools=chunk_pools,
                page_size=page_size,
                num_heads=num_heads,
                head_dim=head_dim,
                num_groups=num_groups,
                pool_size=pool_size,
                BLOCK_N=block_n,
                BLOCK_D=min(head_dim, 128),
                num_warps=4,
                num_stages=1,
            )
            _kpool_reduce_heads_kernel[(tile_rows, triton.cdiv(span, block_n))](
                tile_head_logits,
                weights[row_start:row_end],
                tile_causal_lens,
                tile_logits,
                offset,
                head_logits_token_stride=tile_head_logits.stride(0),
                head_logits_head_stride=tile_head_logits.stride(1),
                logits_stride=tile_logits.stride(0),
                chunk_pools=chunk_pools,
                num_heads=num_heads,
                pool_size=pool_size,
                softmax_scale=float(softmax_scale),
                APPLY_RELU=bool(apply_relu),
                BLOCK_N=block_n,
                num_warps=4,
                num_stages=1,
            )
            take = min(topk_pools, span)
            if span == chunk_pools and use_device_sort:
                tile_sorted_vals = sorted_vals_workspace[:tile_rows]
                tile_sorted_local = sorted_local_workspace[:tile_rows]
                _kpool_sort_topk_kernel[(tile_rows,)](
                    tile_logits,
                    tile_causal_lens,
                    tile_sorted_vals,
                    tile_sorted_local,
                    offset,
                    logits_stride=tile_logits.stride(0),
                    pool_size=pool_size,
                    chunk_pools=chunk_pools,
                    num_warps=8,
                )
                vals = tile_sorted_vals[:, :take]
                local = tile_sorted_local[:, :take]
            else:
                vals, local = torch.topk(
                    tile_logits[:, :span], k=take, dim=-1, sorted=False
                )
            pools = local.to(torch.int32) + offset
            if best_vals is None:
                # The device-sort output is a reusable workspace; retain only
                # the running candidates before the next window overwrites it.
                best_vals, best_pools = vals.clone(), pools
                continue
            merged_vals = torch.cat((best_vals, vals), dim=1)
            merged_pools = torch.cat((best_pools, pools), dim=1)
            best_vals, winners = torch.topk(
                merged_vals,
                k=min(topk_pools, merged_vals.shape[1]),
                dim=-1,
                sorted=False,
            )
            best_pools = torch.gather(merged_pools, 1, winners)

        if best_pools is None:
            result[row_start:row_end].fill_(-1)
            continue
        best_pools = torch.where(
            torch.isfinite(best_vals), best_pools, torch.full_like(best_pools, -1)
        )
        if best_pools.shape[1] < topk_pools:
            best_pools = torch.nn.functional.pad(
                best_pools, (0, topk_pools - best_pools.shape[1]), value=-1
            )
        result[row_start:row_end].copy_(best_pools)

    return result.contiguous()


def score_kpool_dense(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    index_block_table: torch.Tensor,
    *,
    pool_size: int,
    page_size: int,
    softmax_scale: float,
    apply_relu: bool,
    max_num_pools: int,
    out: torch.Tensor | None = None,
    length_masked_consumer: bool = False,
) -> torch.Tensor:
    """Materialize length-masked KPool scores with tensor-core MMA.

    Args:
        q: Packed index queries.
        pooled_k_cache: Packed paged FP8 KPool cache.
        weights: Per-query head weights.
        causal_lens: Visible raw-token lengths.
        req_ids: Request id for each query.
        index_block_table: Pooled-cache page table.
        pool_size: Raw tokens represented by one pool.
        page_size: Rows in each pooled-cache page.
        softmax_scale: Per-head score scale.
        apply_relu: Whether to apply the indexer ReLU.
        max_num_pools: Width of the score matrix.
        out: Optional caller-owned score matrix.
        length_masked_consumer: Skip initializing masked columns when safe.

    Returns:
        FP32 scores shaped ``[tokens, max_num_pools]``.
    """
    num_tokens, num_heads, head_dim = q.shape
    num_groups = head_dim // 128
    max_num_pools = int(max_num_pools)
    if max_num_pools <= 0:
        raise ValueError(f"max_num_pools must be positive, got {max_num_pools}")

    expected_shape = (num_tokens, max_num_pools)
    if out is None:
        logits = torch.empty(expected_shape, dtype=torch.float32, device=q.device)
    else:
        if (
            tuple(out.shape) != expected_shape
            or out.dtype != torch.float32
            or out.device != q.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                f"out must be contiguous float32 {expected_shape} on {q.device}"
            )
        logits = out

    block_h = max(16, triton.next_power_of_2(num_heads))
    block_m = max(16, min(128, 4096 // block_h))
    num_pool_blocks = triton.cdiv(max_num_pools, block_m)
    target_programs = (
        16 * torch.cuda.get_device_properties(q.device).multi_processor_count
    )
    num_workers = min(
        num_pool_blocks,
        1024,
        max(1, triton.cdiv(target_programs, num_tokens)),
    )
    if not length_masked_consumer:
        logits.fill_(-float("inf"))
    _kpool_score_dense_mma_kernel[(num_tokens, num_workers)](
        q,
        pooled_k_cache.view(torch.float8_e4m3fn),
        pooled_k_cache.view(torch.float32),
        weights,
        causal_lens,
        req_ids,
        index_block_table,
        logits,
        max_num_pools,
        cache_page_stride_bytes=pooled_k_cache.stride(0),
        block_table_stride=index_block_table.stride(0),
        logits_stride=logits.stride(0),
        page_size=page_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_groups=num_groups,
        pool_size=pool_size,
        softmax_scale=float(softmax_scale),
        APPLY_RELU=bool(apply_relu),
        NUM_WORKERS=num_workers,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        BLOCK_D=128,
        num_warps=8,
        num_stages=2,
    )
    return logits


__all__ = ["score_kpool_dense", "select_kpool_chunked"]
