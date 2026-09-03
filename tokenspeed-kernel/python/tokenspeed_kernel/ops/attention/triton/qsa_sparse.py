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

"""QSA sparse-attention kernels and FlashInfer metadata preparation."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.triton.qwen4_exp_qsa import (
    qwen4_exp_qsa_sparse_attention,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_QSA_BLACKWELL_NUM_HEADS = 6
_QSA_BLACKWELL_HEAD_DIM = 256
_QSA_BLACKWELL_SELECTED_WIDTH = 2051
_QSA_BLACKWELL_NUM_SPLITS = 8
_QSA_BLACKWELL_BLOCK_H = 16
_QSA_BLACKWELL_BLOCK_N = 32
_QSA_BLACKWELL_SPLIT_SIZE = triton.cdiv(
    _QSA_BLACKWELL_SELECTED_WIDTH, _QSA_BLACKWELL_NUM_SPLITS
)


@triton.jit
def _qsa_sparse_split_kernel(
    query,
    key_cache,
    value_cache,
    selected_slots,
    partial_output,
    partial_stats,
    scale,
    k_descale,
    v_descale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_v_n,
    stride_v_d,
    stride_s_n,
    stride_s_k,
    stride_po_n,
    stride_po_s,
    stride_po_h,
    stride_po_d,
    stride_ps_n,
    stride_ps_s,
    stride_ps_h,
    stride_ps_v,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SELECTED_WIDTH: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, HEAD_DIM)
    token_offsets = tl.arange(0, BLOCK_N)
    head_mask = head_offsets < NUM_HEADS

    query_values = tl.load(
        query
        + row * stride_q_n
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=head_mask[:, None],
        other=0.0,
    )
    query_values = (query_values * scale * 1.4426950408889634).to(query_values.dtype)

    maximum = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, HEAD_DIM), dtype=tl.float32)
    split_start = split * SPLIT_SIZE
    split_end = tl.minimum(split_start + SPLIT_SIZE, SELECTED_WIDTH)
    slot_row = selected_slots + row * stride_s_n

    for local_start in range(0, SPLIT_SIZE, BLOCK_N):
        selected_offsets = split_start + local_start + token_offsets
        in_split = selected_offsets < split_end
        slots = tl.load(
            slot_row + selected_offsets * stride_s_k,
            mask=in_split,
            other=0,
        ).to(tl.int64)
        valid = in_split & (slots > 0)
        safe_slots = tl.where(valid, slots, 0)
        keys = tl.load(
            key_cache
            + safe_slots[None, :] * stride_k_n
            + dim_offsets[:, None] * stride_k_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            value_cache
            + safe_slots[:, None] * stride_v_n
            + dim_offsets[None, :] * stride_v_d,
            mask=valid[:, None],
            other=0.0,
        )
        keys = (keys.to(tl.float32) * k_descale).to(query_values.dtype)
        values = (values.to(tl.float32) * v_descale).to(query_values.dtype)

        # Put the 32 selected slots on GEMM M instead of padding six heads to
        # a tensor-core M tile.
        scores = tl.trans(
            tl.dot(
                tl.trans(keys),
                tl.trans(query_values),
                out_dtype=tl.float32,
            )
        )
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, -float("inf"))
        has_valid = head_mask & (tl.sum(valid.to(tl.int32), axis=0) > 0)
        block_maximum = tl.max(scores, axis=1)
        next_maximum = tl.where(has_valid, tl.maximum(maximum, block_maximum), maximum)
        correction = tl.where(has_valid, tl.exp2(maximum - next_maximum), 1.0)
        probabilities = tl.where(
            head_mask[:, None] & valid[None, :],
            tl.exp2(scores - next_maximum[:, None]),
            0.0,
        )
        accumulator *= correction[:, None]
        # Swapping P/V makes head_dim=256 the value GEMM's M instead of the
        # padded query-head dimension.
        accumulator += tl.trans(
            tl.dot(
                tl.trans(values),
                tl.trans(probabilities.to(values.dtype)),
            )
        )
        normalizer = normalizer * correction + tl.sum(probabilities, axis=1)
        maximum = next_maximum

    tl.store(
        partial_output
        + row * stride_po_n
        + split * stride_po_s
        + head_offsets[:, None] * stride_po_h
        + dim_offsets[None, :] * stride_po_d,
        accumulator,
        mask=head_mask[:, None],
    )
    stats_base = (
        partial_stats
        + row * stride_ps_n
        + split * stride_ps_s
        + head_offsets * stride_ps_h
    )
    tl.store(stats_base, maximum, mask=head_mask)
    tl.store(stats_base + stride_ps_v, normalizer, mask=head_mask)
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _qsa_sparse_merge_kernel(
    partial_output,
    partial_stats,
    output,
    stride_po_n,
    stride_po_s,
    stride_po_h,
    stride_po_d,
    stride_ps_n,
    stride_ps_s,
    stride_ps_h,
    stride_ps_v,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, HEAD_DIM)
    head_mask = head_offsets < NUM_HEADS

    tl.extra.cuda.gdc_wait()
    maximum = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, HEAD_DIM), dtype=tl.float32)

    for split in tl.static_range(NUM_SPLITS):
        stats_base = (
            partial_stats
            + row * stride_ps_n
            + split * stride_ps_s
            + head_offsets * stride_ps_h
        )
        partial_maximum = tl.load(stats_base, mask=head_mask, other=-float("inf"))
        partial_normalizer = tl.load(
            stats_base + stride_ps_v, mask=head_mask, other=0.0
        )
        partial_accumulator = tl.load(
            partial_output
            + row * stride_po_n
            + split * stride_po_s
            + head_offsets[:, None] * stride_po_h
            + dim_offsets[None, :] * stride_po_d,
            mask=head_mask[:, None],
            other=0.0,
        )

        partial_valid = head_mask & (partial_normalizer > 0.0)
        next_maximum = tl.where(
            partial_valid,
            tl.maximum(maximum, partial_maximum),
            maximum,
        )
        old_valid = head_mask & (normalizer > 0.0)
        safe_maximum = tl.where(old_valid, maximum, next_maximum)
        safe_partial_maximum = tl.where(partial_valid, partial_maximum, next_maximum)
        old_correction = tl.where(
            old_valid,
            tl.exp2(safe_maximum - next_maximum),
            0.0,
        )
        partial_correction = tl.where(
            partial_valid,
            tl.exp2(safe_partial_maximum - next_maximum),
            0.0,
        )
        accumulator = (
            accumulator * old_correction[:, None]
            + partial_accumulator * partial_correction[:, None]
        )
        normalizer = (
            normalizer * old_correction + partial_normalizer * partial_correction
        )
        maximum = next_maximum

    result = tl.where(
        normalizer[:, None] > 0.0,
        accumulator / normalizer[:, None],
        0.0,
    )
    tl.store(
        output
        + row * stride_o_n
        + head_offsets[:, None] * stride_o_h
        + dim_offsets[None, :] * stride_o_d,
        result,
        mask=head_mask[:, None],
    )


def _blackwell_qsa_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    k_scale: float | torch.Tensor,
    v_scale: float | torch.Tensor,
) -> torch.Tensor:
    rows = query.shape[0]
    output = torch.empty_like(query)
    partial_output = torch.empty(
        (
            rows,
            _QSA_BLACKWELL_NUM_SPLITS,
            _QSA_BLACKWELL_NUM_HEADS,
            _QSA_BLACKWELL_HEAD_DIM,
        ),
        dtype=torch.float32,
        device=query.device,
    )
    partial_stats = torch.empty(
        (
            rows,
            _QSA_BLACKWELL_NUM_SPLITS,
            _QSA_BLACKWELL_NUM_HEADS,
            2,
        ),
        dtype=torch.float32,
        device=query.device,
    )

    _qsa_sparse_split_kernel[(rows, _QSA_BLACKWELL_NUM_SPLITS)](
        query,
        key_cache,
        value_cache,
        selected_slots,
        partial_output,
        partial_stats,
        float(scale),
        float(k_scale),
        float(v_scale),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(2),
        selected_slots.stride(0),
        selected_slots.stride(1),
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_stats.stride(0),
        partial_stats.stride(1),
        partial_stats.stride(2),
        partial_stats.stride(3),
        NUM_HEADS=_QSA_BLACKWELL_NUM_HEADS,
        HEAD_DIM=_QSA_BLACKWELL_HEAD_DIM,
        SELECTED_WIDTH=_QSA_BLACKWELL_SELECTED_WIDTH,
        NUM_SPLITS=_QSA_BLACKWELL_NUM_SPLITS,
        SPLIT_SIZE=_QSA_BLACKWELL_SPLIT_SIZE,
        BLOCK_H=_QSA_BLACKWELL_BLOCK_H,
        BLOCK_N=_QSA_BLACKWELL_BLOCK_N,
        num_warps=4,
        num_stages=2,
        launch_pdl=True,
    )
    _qsa_sparse_merge_kernel[(rows,)](
        partial_output,
        partial_stats,
        output,
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_stats.stride(0),
        partial_stats.stride(1),
        partial_stats.stride(2),
        partial_stats.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_HEADS=_QSA_BLACKWELL_NUM_HEADS,
        HEAD_DIM=_QSA_BLACKWELL_HEAD_DIM,
        NUM_SPLITS=_QSA_BLACKWELL_NUM_SPLITS,
        BLOCK_H=_QSA_BLACKWELL_BLOCK_H,
        num_warps=4,
        num_stages=1,
        launch_pdl=True,
    )
    return output


@triton.jit
def _prepare_qsa_sparse_indices_kernel(
    selected_slots,
    indices,
    packed_mask,
    stride_s_n,
    stride_s_k,
    stride_i_n,
    packed_stride,
    WIDTH: tl.constexpr,
    PACKED_WIDTH: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    row = tl.program_id(0)
    byte_offsets = tl.program_id(1) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    bit_offsets = tl.arange(0, 8)
    columns = byte_offsets[:, None] * 8 + bit_offsets[None, :]
    column_mask = columns < WIDTH
    slots = tl.load(
        selected_slots + row * stride_s_n + columns * stride_s_k,
        mask=column_mask,
        other=-1,
    ).to(tl.int32)
    valid = column_mask & (slots > 0)
    tl.store(
        indices + row * stride_i_n + columns,
        tl.where(valid, slots, 0),
        mask=column_mask,
    )
    packed = tl.sum(valid.to(tl.int32) << bit_offsets[None, :], axis=1)
    tl.store(
        packed_mask + row * packed_stride + byte_offsets,
        packed,
        mask=byte_offsets < PACKED_WIDTH,
    )


def prepare_qsa_sparse_indices(
    selected_slots: torch.Tensor,
    indices: torch.Tensor,
    packed_mask: torch.Tensor,
    *,
    enable_pdl: bool = False,
) -> None:
    """Sanitize QSA slots and pack their validity into static FlashInfer buffers.

    Args:
        selected_slots: Physical slots ``[rows, width]``; non-positive values
            are invalid.
        indices: Contiguous int32 output buffer with the same shape. Invalid
            entries are replaced by the safe sentinel slot zero.
        packed_mask: Contiguous uint8 buffer with
            ``rows * ceil(width / 8)`` elements, packed little-endian per row.
        enable_pdl: Mark the producer launch for a PDL-capable consumer.

    Returns:
        None. ``indices`` and ``packed_mask`` are written in place.
    """

    if selected_slots.ndim != 2 or indices.shape != selected_slots.shape:
        raise ValueError("QSA slots and FlashInfer indices must share [rows, width]")
    if indices.dtype != torch.int32 or not indices.is_contiguous():
        raise TypeError("FlashInfer QSA indices must be contiguous int32")
    if packed_mask.dtype != torch.uint8 or not packed_mask.is_contiguous():
        raise TypeError("FlashInfer QSA packed mask must be contiguous uint8")
    if selected_slots.device != indices.device or indices.device != packed_mask.device:
        raise ValueError("QSA slots, indices, and packed mask must share one device")
    rows, width = selected_slots.shape
    packed_width = (width + 7) // 8
    if packed_mask.numel() != rows * packed_width:
        raise ValueError(
            "FlashInfer QSA packed mask has the wrong size: "
            f"expected {rows * packed_width}, got {packed_mask.numel()}"
        )
    if rows == 0 or width == 0:
        return
    block_bytes = 128
    pdl_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _prepare_qsa_sparse_indices_kernel[(rows, triton.cdiv(packed_width, block_bytes))](
        selected_slots,
        indices,
        packed_mask,
        selected_slots.stride(0),
        selected_slots.stride(1),
        indices.stride(0),
        packed_width,
        WIDTH=width,
        PACKED_WIDTH=packed_width,
        BLOCK_BYTES=block_bytes,
        num_warps=4,
        num_stages=1,
        **pdl_kwargs,
    )


_QSA_SIGNATURES = {
    format_signature(
        q=dense_tensor_format(dtype),
        k_cache=dense_tensor_format(dtype),
        v_cache=dense_tensor_format(dtype),
    )
    for dtype in (torch.bfloat16, torch.float16)
}
_QSA_SIGNATURES.update(
    format_signature(
        q=dense_tensor_format(q_dtype),
        k_cache=dense_tensor_format(kv_dtype),
        v_cache=dense_tensor_format(kv_dtype),
    )
    for q_dtype in (torch.bfloat16, torch.float16)
    for kv_dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    )
)
_QSA_SIGNATURES = frozenset(_QSA_SIGNATURES)

_QSA_BLACKWELL_SIGNATURES = frozenset(
    {
        format_signature(
            q=dense_tensor_format(torch.bfloat16),
            k_cache=dense_tensor_format(torch.float8_e4m3fn),
            v_cache=dense_tensor_format(torch.float8_e4m3fn),
        )
    }
)


@register_kernel(
    "attention",
    "qsa_sparse_attention",
    name="triton_blackwell_qsa_sparse_attention",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=_QSA_BLACKWELL_SIGNATURES,
    traits={
        "batch_size": frozenset(range(1, 9)),
        "head_dim": frozenset({_QSA_BLACKWELL_HEAD_DIM}),
        "value_head_dim": frozenset({_QSA_BLACKWELL_HEAD_DIM}),
        "num_q_heads": frozenset({_QSA_BLACKWELL_NUM_HEADS}),
        "num_kv_heads": frozenset({1}),
        "selected_width": frozenset({_QSA_BLACKWELL_SELECTED_WIDTH}),
    },
    priority=Priority.SPECIALIZED + 1,
    tags={"latency", "blackwell", "sparse"},
)
def triton_blackwell_qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    max_seqlen_q: int = 1,
    k_scale: float | torch.Tensor | None = None,
    v_scale: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the SM100 TP4 QSA sparse-attention specialization.

    Args:
        q: BF16 query tensor shaped ``[tokens, 6, 256]``.
        k_cache: FP8 E4M3 key cache shaped ``[cache_slots, 1, 256]``.
        v_cache: FP8 E4M3 value cache shaped ``[cache_slots, 1, 256]``.
        selected_slots: Physical cache slots shaped ``[tokens, 2051]``;
            non-positive values are ignored.
        scale: Softmax scale applied to query-key scores.
        max_seqlen_q: Uniform query-token count per request. The current
            fallback computes the packed rows independently.
        k_scale: Scalar key-cache descale.
        v_scale: Scalar value-cache descale.

    Returns:
        BF16 attention output shaped ``[tokens, 6, 256]``.
    """

    if q.shape[1:] != (_QSA_BLACKWELL_NUM_HEADS, _QSA_BLACKWELL_HEAD_DIM):
        raise ValueError("SM100 QSA requires query shape [tokens, 6, 256]")
    if k_cache.shape[1:] != (1, _QSA_BLACKWELL_HEAD_DIM):
        raise ValueError("SM100 QSA requires key cache shape [slots, 1, 256]")
    if v_cache.shape != k_cache.shape:
        raise ValueError("SM100 QSA requires matching key/value cache shapes")
    if selected_slots.shape != (q.shape[0], _QSA_BLACKWELL_SELECTED_WIDTH):
        raise ValueError("SM100 QSA requires selected slots shape [tokens, 2051]")
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.float8_e4m3fn:
        raise TypeError("SM100 QSA requires BF16 queries and FP8 E4M3 caches")
    if v_cache.dtype != k_cache.dtype:
        raise TypeError("SM100 QSA key/value cache dtypes must match")
    if k_scale is None or v_scale is None:
        raise ValueError("SM100 QSA FP8 cache requires both K and V descales")
    if q.shape[0] == 0:
        return q.new_empty((0, _QSA_BLACKWELL_NUM_HEADS, _QSA_BLACKWELL_HEAD_DIM))
    del max_seqlen_q
    return _blackwell_qsa_sparse_attention(
        q,
        k_cache,
        v_cache,
        selected_slots,
        scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )


@register_kernel(
    "attention",
    "qsa_sparse_attention",
    name="triton_qsa_sparse_attention",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=_QSA_SIGNATURES,
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    max_seqlen_q: int = 1,
    k_scale: float | torch.Tensor | None = None,
    v_scale: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Triton QSA sparse attention, using the SM100 split specialization.

    Args:
        q: Query tensor shaped ``[tokens, query_heads, head_dim]``.
        k_cache: Flattened key cache shaped
            ``[cache_slots, kv_heads, head_dim]``.
        v_cache: Flattened value cache shaped
            ``[cache_slots, kv_heads, value_head_dim]``.
        selected_slots: Physical cache slots shaped ``[tokens, budget]``;
            non-positive values are ignored.
        scale: Softmax scale applied to query-key scores.
        max_seqlen_q: Uniform query-token count per request. Portable Triton
            currently computes the packed rows independently.
        k_scale: Optional scalar FP8 key descale.
        v_scale: Optional scalar FP8 value descale.

    Returns:
        Attention output shaped ``[tokens, query_heads, value_head_dim]``.
    """

    platform = current_platform()
    if (
        platform.is_nvidia
        and platform.arch_version == ArchVersion(10, 0)
        and q.ndim == 3
        and k_cache.ndim == 3
        and v_cache.ndim == 3
        and selected_slots.ndim == 2
        and 0 < q.shape[0] <= 8
        and q.shape[1:] == (_QSA_BLACKWELL_NUM_HEADS, _QSA_BLACKWELL_HEAD_DIM)
        and k_cache.shape[1:] == (1, _QSA_BLACKWELL_HEAD_DIM)
        and v_cache.shape == k_cache.shape
        and selected_slots.shape == (q.shape[0], _QSA_BLACKWELL_SELECTED_WIDTH)
        and q.dtype == torch.bfloat16
        and k_cache.dtype == torch.float8_e4m3fn
        and v_cache.dtype == k_cache.dtype
    ):
        return triton_blackwell_qsa_sparse_attention(
            q,
            k_cache,
            v_cache,
            selected_slots,
            scale=scale,
            max_seqlen_q=max_seqlen_q,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    del max_seqlen_q
    return qwen4_exp_qsa_sparse_attention(
        q,
        k_cache,
        v_cache,
        selected_slots,
        scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )


__all__ = [
    "prepare_qsa_sparse_indices",
    "triton_blackwell_qsa_sparse_attention",
    "triton_qsa_sparse_attention",
]
