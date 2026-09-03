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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@triton.jit
def _kpool_hadamard_stage(x, groups: tl.constexpr, stride: tl.constexpr):
    x = tl.reshape(x, (groups, 2, stride))
    x = tl.trans(x, 0, 2, 1)
    a, b = tl.split(x)
    x = tl.join(a + b, a - b)
    x = tl.trans(x, 0, 2, 1)
    return tl.reshape(x, (128,))


@triton.jit
def _kpool_hadamard_128(x):
    x = _kpool_hadamard_stage(x, 64, 1)
    x = _kpool_hadamard_stage(x, 32, 2)
    x = _kpool_hadamard_stage(x, 16, 4)
    x = _kpool_hadamard_stage(x, 8, 8)
    x = _kpool_hadamard_stage(x, 4, 16)
    x = _kpool_hadamard_stage(x, 2, 32)
    x = _kpool_hadamard_stage(x, 1, 64)
    return x * 0.08838834764831845


@triton.jit
def _kpool_quantize(x):
    x = x.to(tl.bfloat16).to(tl.float32)
    x = _kpool_hadamard_128(x).to(tl.bfloat16).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
    scale = absmax / 448.0
    return tl.minimum(tl.maximum(x / scale, -448.0), 448.0), scale


@triton.jit
def _kpool_prefill_write_kernel(
    slot_k_ptr,
    slot_k_stride_row,
    slot_k_stride_pool,
    slot_score_ptr,
    slot_score_stride_row,
    slot_score_stride_pool,
    ape_ptr,
    ape_stride_pool,
    write_slots_ptr,
    index_values_ptr,
    index_values_stride_page,
    index_values_stride_row,
    index_scales_ptr,
    index_scales_stride_page,
    index_scales_stride_row,
    INDEX_ROWS_PER_PAGE: tl.constexpr,
    INDEX_NUM_PAGES: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM

    max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_row
            + slot * slot_score_stride_pool
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(
            ape_ptr + slot * ape_stride_pool + offs, mask=mask, other=0.0
        ).to(tl.float32)
        max_score = tl.maximum(max_score, score)

    acc = tl.zeros((BLOCK_D,), tl.float32)
    denom = tl.zeros((BLOCK_D,), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_row
            + slot * slot_score_stride_pool
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(
            ape_ptr + slot * ape_stride_pool + offs, mask=mask, other=0.0
        ).to(tl.float32)
        prob = tl.exp(score - max_score)
        denom += prob
        k = tl.load(
            slot_k_ptr + row * slot_k_stride_row + slot * slot_k_stride_pool + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += k * prob

    quantized, scale = _kpool_quantize(acc / denom)

    write_slot = tl.load(write_slots_ptr + row).to(tl.int64)
    page = write_slot // INDEX_ROWS_PER_PAGE
    index_row = write_slot % INDEX_ROWS_PER_PAGE
    valid = (write_slot >= 0) & (page < INDEX_NUM_PAGES)
    value_base = page * index_values_stride_page + index_row * index_values_stride_row
    scale_base = page * index_scales_stride_page + index_row * index_scales_stride_row
    tl.store(
        index_values_ptr + value_base + offs,
        quantized.to(index_values_ptr.dtype.element_ty),
        mask=mask & valid,
    )
    tl.store(index_scales_ptr + scale_base, scale, mask=valid)


@register_kernel(
    "attention",
    "kpool_prefill_write",
    name="triton_kpool_prefill_write",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {format_signature(slot_k=dense_tensor_format(torch.bfloat16))}
    ),
    traits={
        "head_dim": frozenset({128}),
        "pool_size": frozenset({2, 4, 8, 16}),
        "index_k_format": frozenset({"fp8_scaled"}),
        "rotate": frozenset({True}),
    },
    priority=Priority.PORTABLE,
)
def triton_kpool_prefill_write(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    write_slots: torch.Tensor,
    index_values: torch.Tensor,
    index_scales: torch.Tensor,
    ape: torch.Tensor,
) -> None:
    """Compress completed prefill pools directly into their index-cache slots.

    Args:
        slot_k: BF16 keys shaped ``[rows, pool_size, 128]``.
        slot_score: Matching BF16 or FP32 per-channel scores.
        write_slots: Flattened physical index-cache slots.
        index_values: Paged FP8 pool values, updated in place.
        index_scales: Paged FP32 pool scales, updated in place.
        ape: FP32 intra-pool bias shaped ``[pool_size, 128]``.
    Returns:
        None. Completed pools are written directly to the cache.
    """
    rows, pool_size, head_dim = slot_k.shape
    if (
        slot_score.shape != slot_k.shape
        or write_slots.numel() != rows
        or ape.shape != (pool_size, head_dim)
        or index_values.dim() != 3
        or index_values.shape[-1] != head_dim
        or index_scales.shape[:2] != index_values.shape[:2]
    ):
        raise ValueError("invalid KPool prefill cache geometry")
    if not slot_k.is_cuda:
        raise RuntimeError("KPool prefill writes require CUDA tensors")

    if rows == 0:
        return

    slot_k = slot_k.contiguous()
    slot_score = slot_score.contiguous()
    write_slots = write_slots.to(device=slot_k.device, dtype=torch.int64).contiguous()
    ape = ape.contiguous()

    _kpool_prefill_write_kernel[(rows,)](
        slot_k,
        slot_k.stride(0),
        slot_k.stride(1),
        slot_score,
        slot_score.stride(0),
        slot_score.stride(1),
        ape,
        ape.stride(0),
        write_slots,
        index_values,
        index_values.stride(0),
        index_values.stride(1),
        index_scales,
        index_scales.stride(0),
        index_scales.stride(1),
        INDEX_ROWS_PER_PAGE=index_values.shape[1],
        INDEX_NUM_PAGES=index_values.shape[0],
        POOL_SIZE=pool_size,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _kpool_prefill_tail_write_kernel(
    k_ptr,
    gate_ptr,
    tail_k_ptr,
    tail_gate_ptr,
    source_starts_ptr,
    destination_slots_ptr,
    destination_positions_ptr,
    valid_counts_ptr,
    k_stride_row: tl.constexpr,
    gate_stride_row: tl.constexpr,
    tail_k_stride_req: tl.constexpr,
    tail_k_stride_pool: tl.constexpr,
    tail_gate_stride_req: tl.constexpr,
    tail_gate_stride_pool: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    TAIL_NUM_REQUESTS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    source_start = tl.load(source_starts_ptr + row).to(tl.int64)
    destination_slot = tl.load(destination_slots_ptr + row).to(tl.int64)
    destination_position = tl.load(destination_positions_ptr + row).to(tl.int64)
    valid_count = tl.load(valid_counts_ptr + row).to(tl.int64)
    valid_destination = (
        (destination_slot >= 0)
        & (destination_slot < TAIL_NUM_REQUESTS)
        & (destination_position >= 0)
    )
    safe_destination_slot = tl.where(valid_destination, destination_slot, 0)

    for pool_offset in tl.static_range(0, POOL_SIZE):
        source_row = source_start + pool_offset
        active = (
            valid_destination
            & (pool_offset < valid_count)
            & (source_row >= 0)
            & (source_row < NUM_TOKENS)
        )
        safe_source_row = tl.where(active, source_row, 0)
        destination_row = (destination_position + pool_offset) % TAIL_SIZE
        k_destination_base = (
            safe_destination_slot * tail_k_stride_req
            + destination_row * tail_k_stride_pool
        )
        gate_destination_base = (
            safe_destination_slot * tail_gate_stride_req
            + destination_row * tail_gate_stride_pool
        )
        k_source_base = safe_source_row * k_stride_row
        gate_source_base = safe_source_row * gate_stride_row
        key = tl.load(k_ptr + k_source_base + offs, mask=dim_mask & active, other=0.0)
        score = tl.load(
            gate_ptr + gate_source_base + offs, mask=dim_mask & active, other=0.0
        )
        tl.store(tail_k_ptr + k_destination_base + offs, key, mask=dim_mask & active)
        tl.store(
            tail_gate_ptr + gate_destination_base + offs,
            score,
            mask=dim_mask & active,
        )


@register_kernel(
    "attention",
    "kpool_prefill_tail_write",
    name="triton_kpool_prefill_tail_write",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({format_signature(k=dense_tensor_format(torch.bfloat16))}),
    traits={
        "head_dim": frozenset({128}),
        "pool_size": frozenset({2, 4, 8, 16}),
    },
    priority=Priority.PORTABLE,
)
def triton_kpool_prefill_tail_write(
    k: torch.Tensor,
    gate: torch.Tensor,
    tail_k: torch.Tensor,
    tail_gate: torch.Tensor,
    source_starts: torch.Tensor,
    destination_slots: torch.Tensor,
    destination_positions: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    pool_size: int,
) -> None:
    """Copy incomplete prefill pools into fixed request-local tail buffers.

    Args:
        k: Full BF16 prefill keys shaped ``[tokens, 128]``.
        gate: Matching BF16 per-channel pool scores.
        tail_k: Request-local key ring, updated in place.
        tail_gate: Request-local score ring, updated in place.
        source_starts: First source token for each fixed metadata row.
        destination_slots: Stable request-tail slot for each metadata row.
            Negative slots are ignored.
        destination_positions: First logical destination position per row.
        valid_counts: Number of live tokens in each row. Zero makes the row
            inactive without changing the captured launch geometry.
        pool_size: Maximum number of tokens represented by one metadata row.

    Returns:
        None. Only destinations selected by live metadata are updated.
    """
    if k.dim() != 2 or k.shape[1] != 128 or gate.shape != k.shape:
        raise ValueError("invalid KPool prefill tail input geometry")
    if (
        tail_k.dim() != 3
        or tail_gate.shape != tail_k.shape
        or tail_k.shape[2] != k.shape[1]
        or tail_k.shape[1] < pool_size
        or pool_size not in (2, 4, 8, 16)
    ):
        raise ValueError("invalid KPool prefill tail cache geometry")
    metadata = (
        source_starts,
        destination_slots,
        destination_positions,
        valid_counts,
    )
    rows = source_starts.numel()
    if any(item.dim() != 1 or item.numel() != rows for item in metadata):
        raise ValueError("KPool prefill tail metadata must have matching row counts")
    integer_dtypes = (torch.int32, torch.int64)
    if (
        k.dtype != torch.bfloat16
        or gate.dtype != torch.bfloat16
        or tail_k.dtype != torch.bfloat16
        or tail_gate.dtype != torch.bfloat16
        or any(item.dtype not in integer_dtypes for item in metadata)
    ):
        raise TypeError("invalid KPool prefill tail dtype")
    if not k.is_cuda or any(
        item.device != k.device for item in (gate, tail_k, tail_gate, *metadata)
    ):
        raise RuntimeError("KPool prefill tail writes require one CUDA device")
    if (
        k.stride(1) != 1
        or gate.stride(1) != 1
        or tail_k.stride(2) != 1
        or tail_gate.stride(2) != 1
        or any(not item.is_contiguous() for item in metadata)
    ):
        raise ValueError("KPool prefill tail inputs must use contiguous rows")
    if rows == 0:
        return

    _kpool_prefill_tail_write_kernel[(rows,)](
        k,
        gate,
        tail_k,
        tail_gate,
        source_starts,
        destination_slots,
        destination_positions,
        valid_counts,
        k.stride(0),
        gate.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        tail_gate.stride(0),
        tail_gate.stride(1),
        NUM_TOKENS=k.shape[0],
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_k.shape[1],
        TAIL_NUM_REQUESTS=tail_k.shape[0],
        HEAD_DIM=k.shape[1],
        BLOCK_D=triton.next_power_of_2(k.shape[1]),
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _kpool_decode_append_kernel(
    k_ptr,
    gate_ptr,
    tail_k_ptr,
    tail_gate_ptr,
    seq_lens_ptr,
    request_slots_ptr,
    index_table_ptr,
    index_values_ptr,
    index_scales_ptr,
    ape_ptr,
    k_stride_req: tl.constexpr,
    k_stride_step: tl.constexpr,
    tail_stride_req: tl.constexpr,
    tail_stride_pool: tl.constexpr,
    index_table_stride_req: tl.constexpr,
    index_table_stride_col: tl.constexpr,
    index_values_stride_page: tl.constexpr,
    index_values_stride_row: tl.constexpr,
    index_scales_stride_page: tl.constexpr,
    index_scales_stride_row: tl.constexpr,
    ape_stride_pool: tl.constexpr,
    NUM_STEPS: tl.constexpr,
    INDEX_ROWS_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INDEX_TABLE_COLS: tl.constexpr,
    TAIL_NUM_REQUESTS: tl.constexpr,
    INDEX_NUM_PAGES: tl.constexpr,
):
    req = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM

    final_seq_len = tl.load(seq_lens_ptr + req).to(tl.int32)
    first_before = final_seq_len - NUM_STEPS
    request_slot = tl.load(request_slots_ptr + req).to(tl.int64)
    valid_request = (request_slot > 0) & (request_slot < TAIL_NUM_REQUESTS)
    request_slot = tl.where(valid_request, request_slot, 0)
    active = (first_before >= 0) & valid_request
    tail_base = request_slot * tail_stride_req

    for step in tl.static_range(0, NUM_STEPS):
        before = first_before + step
        safe_before = tl.maximum(before, 0)
        logical_slot = safe_before % POOL_SIZE
        physical_slot = safe_before % TAIL_SIZE

        token_base = req * k_stride_req + step * k_stride_step
        k_new = tl.load(k_ptr + token_base + offs, mask=mask & active, other=0.0)
        g_new = tl.load(gate_ptr + token_base + offs, mask=mask & active, other=0.0)
        tl.store(
            tail_k_ptr + tail_base + physical_slot * tail_stride_pool + offs,
            k_new,
            mask=mask & active,
        )
        tl.store(
            tail_gate_ptr + tail_base + physical_slot * tail_stride_pool + offs,
            g_new,
            mask=mask & active,
        )
        tl.debug_barrier()

        is_full = active & (logical_slot == POOL_SIZE - 1)
        if is_full:
            pool_logical_start = safe_before - (POOL_SIZE - 1)
            max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                pool_physical_slot = (pool_logical_start + pool_slot) % TAIL_SIZE
                stored_g = tl.load(
                    tail_gate_ptr
                    + tail_base
                    + pool_physical_slot * tail_stride_pool
                    + offs,
                    mask=mask,
                    other=0.0,
                )
                score = tl.where(pool_slot == logical_slot, g_new, stored_g).to(
                    tl.float32
                )
                score += tl.load(
                    ape_ptr + pool_slot * ape_stride_pool + offs,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                max_score = tl.maximum(max_score, score)

            acc = tl.zeros((BLOCK_D,), tl.float32)
            denom = tl.zeros((BLOCK_D,), tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                pool_physical_slot = (pool_logical_start + pool_slot) % TAIL_SIZE
                stored_g = tl.load(
                    tail_gate_ptr
                    + tail_base
                    + pool_physical_slot * tail_stride_pool
                    + offs,
                    mask=mask,
                    other=0.0,
                )
                score = tl.where(pool_slot == logical_slot, g_new, stored_g).to(
                    tl.float32
                )
                score += tl.load(
                    ape_ptr + pool_slot * ape_stride_pool + offs,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                prob = tl.exp(score - max_score)
                denom += prob
                stored_k = tl.load(
                    tail_k_ptr
                    + tail_base
                    + pool_physical_slot * tail_stride_pool
                    + offs,
                    mask=mask,
                    other=0.0,
                )
                k_value = tl.where(pool_slot == logical_slot, k_new, stored_k).to(
                    tl.float32
                )
                acc += k_value * prob

            quantized, scale = _kpool_quantize(acc / denom)

            pool_id = safe_before // POOL_SIZE
            table_col = pool_id // INDEX_ROWS_PER_PAGE
            index_page = tl.load(
                index_table_ptr
                + req * index_table_stride_req
                + table_col * index_table_stride_col,
                mask=is_full & (table_col < INDEX_TABLE_COLS),
                other=0,
            ).to(tl.int64)
            index_page_valid = (
                is_full & (index_page > 0) & (index_page < INDEX_NUM_PAGES)
            )
            safe_index_page = tl.where(index_page_valid, index_page, 0)
            index_row = pool_id % INDEX_ROWS_PER_PAGE
            value_base = (
                safe_index_page * index_values_stride_page
                + index_row * index_values_stride_row
            )
            tl.store(
                index_values_ptr + value_base + offs,
                quantized.to(index_values_ptr.dtype.element_ty),
                mask=mask & index_page_valid,
            )
            scale_base = (
                safe_index_page * index_scales_stride_page
                + index_row * index_scales_stride_row
            )
            tl.store(index_scales_ptr + scale_base, scale, mask=index_page_valid)


@register_kernel(
    "attention",
    "kpool_decode_append",
    name="triton_kpool_decode_append",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({format_signature(k=dense_tensor_format(torch.bfloat16))}),
    traits={
        "head_dim": frozenset({128}),
        "pool_size": frozenset({2, 4, 8, 16}),
        "index_k_format": frozenset({"fp8_scaled"}),
        "rotate": frozenset({True}),
    },
    priority=Priority.PERFORMANT,
)
def triton_kpool_decode_append(
    k: torch.Tensor,
    gate: torch.Tensor,
    tail_k: torch.Tensor,
    tail_gate: torch.Tensor,
    seq_lens: torch.Tensor,
    request_slots: torch.Tensor,
    index_block_table: torch.Tensor,
    index_values: torch.Tensor,
    index_scales: torch.Tensor,
    ape: torch.Tensor,
) -> None:
    """Append a decode window to request-local tails and paged indices.

    Args:
        k: BF16 index keys shaped ``[requests, steps, 128]``.
        gate: Matching per-channel pool scores.
        tail_k: Request-local key rings.
        tail_gate: Request-local score rings.
        seq_lens: Final sequence length per request.
        request_slots: Stable tail row per request.
        index_block_table: Logical-to-physical index-page mapping.
        index_values: Paged FP8 pool values.
        index_scales: Paged FP32 pool scales.
        ape: FP32 intra-pool bias shaped ``[pool_size, 128]``.
    Returns:
        None. Cache tensors are updated in place.
    """
    requests, steps, head_dim = k.shape
    pool_size = ape.shape[0]
    tail_size = tail_k.shape[1]
    if (
        steps < 1
        or head_dim != 128
        or gate.shape != k.shape
        or tail_gate.shape != tail_k.shape
        or tail_k.shape[-1] != head_dim
        or tail_size < pool_size
        or ape.shape != (pool_size, head_dim)
        or index_block_table.dim() != 2
        or index_block_table.shape[0] < requests
        or index_values.dim() != 3
        or index_values.shape[-1] != head_dim
        or index_scales.dim() != 3
        or index_scales.shape[:2] != index_values.shape[:2]
        or index_scales.shape[-1] < 1
    ):
        raise ValueError("invalid KPool decode cache geometry")
    if (
        gate.dtype != torch.bfloat16
        or ape.dtype != torch.float32
        or index_values.dtype != torch.float8_e4m3fn
        or index_scales.dtype != torch.float32
    ):
        raise TypeError("invalid KPool decode cache dtype")
    if seq_lens.numel() < requests or request_slots.numel() < requests:
        raise ValueError("decode metadata does not cover every request")
    if not k.is_cuda:
        raise RuntimeError("KPool decode append requires CUDA tensors")

    if requests == 0:
        return
    seq_lens = seq_lens.to(device=k.device, dtype=torch.int32).contiguous()
    request_slots = request_slots.to(device=k.device, dtype=torch.int32).contiguous()
    index_rows_per_page = index_values.shape[1]
    _kpool_decode_append_kernel[(requests,)](
        k,
        gate,
        tail_k,
        tail_gate,
        seq_lens,
        request_slots,
        index_block_table,
        index_values,
        index_scales,
        ape,
        k.stride(0),
        k.stride(1),
        tail_k.stride(0),
        tail_k.stride(1),
        index_block_table.stride(0),
        index_block_table.stride(1),
        index_values.stride(0),
        index_values.stride(1),
        index_scales.stride(0),
        index_scales.stride(1),
        ape.stride(0),
        NUM_STEPS=steps,
        INDEX_ROWS_PER_PAGE=index_rows_per_page,
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_size,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        INDEX_TABLE_COLS=index_block_table.shape[1],
        TAIL_NUM_REQUESTS=tail_k.shape[0],
        INDEX_NUM_PAGES=index_values.shape[0],
        num_warps=4,
        num_stages=1,
    )


__all__ = [
    "triton_kpool_decode_append",
    "triton_kpool_prefill_tail_write",
    "triton_kpool_prefill_write",
]
