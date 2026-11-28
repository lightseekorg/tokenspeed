# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the LICENSE file
# of https://github.com/fla-org/flash-linear-attention (v0.5.1); vendored here
# with dual-index in-place state-pool addressing added for the flat KV cache.

"""Packed convolution producer for retained KDA verification.

Read each request's convolution history from its pre-convolution ring and
retain its candidate QKV rows there without publishing state. Packed query
boundaries cover live rows and graph padding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel._triton import tl, triton

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.attention.kda import KdaPreconvRing


# The input row, output row and ring row strides are compile-time values:
# they are fixed for a model, and folding them into the addresses lets each
# row's loads and stores vectorize. The ring's slot stride is a runtime value;
# RING_SLOT_ALIGN states the power of two that divides it, so the slot base
# keeps the alignment the row vectors need.
@triton.jit(
    do_not_specialize=[
        "query_stride",
        "read_stride",
        "live_stride",
        "slot_stride",
        "head_stride",
        "total_tokens",
        "pages",
        "slots",
        "ring_slot_stride",
    ]
)
def fused_kda_verify_conv_update_kernel(
    qkv_raw,
    conv_w,
    conv_bias,
    ring,
    read_indices,
    live_indices,
    record_slots,
    ring_heads,
    query_start_loc,
    query_stride,
    read_stride,
    live_stride,
    slot_stride,
    head_stride,
    total_tokens,
    pages,
    slots,
    ring_slot_stride,
    out,
    stride_raw_tok: tl.constexpr,
    stride_out_tok: tl.constexpr,
    stride_ring_row: tl.constexpr,
    RING_SLOT_ALIGN: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # The ring keeps the 3 inputs before each request's window: R = T + 3.
    R: tl.constexpr = T + 3
    T_TILES: tl.constexpr = (T + BLOCK_T - 1) // BLOCK_T
    # One program covers up to BLOCK_T tokens of a request for one channel
    # block (the whole window for typical decode T <= 16), so metadata loads
    # once and every tap load issues before the first store (GB300, Kimi-K3
    # TP8 rank, T 4, batch 64: 165 us a round faster than a program per
    # token). Larger windows fold token tiles into grid dim 0 so BLOCK_T *
    # BLOCK_C stays within Triton's element limit without hitting CUDA's
    # grid.z limit; history and candidate ring rows are disjoint, so
    # concurrent token tiles do not alias.
    if T_TILES == 1:
        i_n = tl.program_id(0).to(tl.int64)
        o_t = tl.arange(0, BLOCK_T)
    else:
        i_nt = tl.program_id(0).to(tl.int64)
        i_n = i_nt // T_TILES
        o_t = ((i_nt % T_TILES) * BLOCK_T).to(tl.int32) + tl.arange(0, BLOCK_T)
    i_c = tl.program_id(1).to(tl.int64)
    begin = tl.load(query_start_loc + i_n * query_stride).to(tl.int64)
    end = tl.load(query_start_loc + (i_n + 1) * query_stride).to(tl.int64)
    token_valid = (
        (begin >= 0) & (end <= total_tokens) & (end - begin <= T) & (o_t < end - begin)
    )
    o_c = i_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = token_valid[:, None] & (o_c < C)[None, :]
    b_read = tl.load(read_indices + i_n * read_stride).to(tl.int64)
    b_live = tl.load(live_indices + i_n * live_stride).to(tl.int64)
    slot = tl.load(record_slots + i_n * slot_stride).to(tl.int64)
    head = tl.load(ring_heads + i_n * head_stride).to(tl.int32)
    # A row takes part as in the retained prepare: its source and live pages
    # lie in the pool and its record slot is resident. Other rows are padding:
    # they read a zero history and keep the ring as it is.
    has_ring = (
        (b_read > 0)
        & (b_read < pages)
        & (b_live > 0)
        & (b_live < pages)
        & (slot >= 0)
        & (slot < slots)
    )
    # The slot's rows: one scalar base, so each row adds a scalar offset to
    # the channel vector and its loads and stores vectorize.
    slot_rows = ring + slot * (ring_slot_stride // RING_SLOT_ALIGN) * RING_SLOT_ALIGN
    rows = qkv_raw + begin * stride_raw_tok + o_c[None, :]
    acc = tl.zeros([BLOCK_T, BLOCK_C], dtype=tl.float32)
    for i_w in tl.static_range(4):
        i_window = o_t + i_w
        from_raw = (i_window >= 3)[:, None]
        i_raw = i_window - 3
        # Tap i_window < 3 is ring row (head - 3 + i_window) mod R. The row
        # index stays below R, so its INT32 math cannot wrap; it is widened
        # before the row stride multiplies it, since the product grows with
        # the ring's row pitch. (INT64 row math measured 40 us a round slower
        # at a batch of 64 with T = 4.)
        i_past = head + i_raw
        i_past = tl.where(i_past < 0, i_past + R, i_past)
        b_state = tl.load(
            slot_rows + i_past.to(tl.int64)[:, None] * stride_ring_row + o_c[None, :],
            mask=mask_c & ~from_raw & has_ring,
            other=0.0,
        ).to(tl.float32)
        b_raw = tl.load(
            rows + i_raw.to(tl.int64)[:, None] * stride_raw_tok,
            mask=mask_c & from_raw,
            other=0.0,
        )
        if i_w == 3:
            # The last tap reads each row before convolution and SiLU.
            kept = b_raw
        b_w = tl.load(conv_w + o_c * 4 + i_w, mask=o_c < C, other=0.0).to(tl.float32)
        acc += tl.where(from_raw, b_raw.to(tl.float32), b_state) * b_w[None, :]
    # Candidate t is ring row (head + t) mod R. Stores follow every load.
    i_kept = head + o_t
    i_kept = tl.where(i_kept >= R, i_kept - R, i_kept)
    tl.store(
        slot_rows + i_kept.to(tl.int64)[:, None] * stride_ring_row + o_c[None, :],
        kept.to(ring.dtype.element_ty),
        mask=mask_c & has_ring,
    )
    if HAS_BIAS:
        acc += tl.load(conv_bias + o_c, mask=o_c < C, other=0.0).to(tl.float32)[None, :]
    acc = acc * tl.sigmoid(acc)
    tl.store(
        out + (begin + o_t.to(tl.int64))[:, None] * stride_out_tok + o_c[None, :],
        acc.to(out.dtype.element_ty),
        mask=mask_c,
    )


def fused_kda_verify_conv_update(
    qkv_raw: torch.Tensor,
    conv_w: torch.Tensor,
    read_indices: torch.Tensor,
    *,
    live_indices: torch.Tensor,
    preconv: KdaPreconvRing,
    record_slots: torch.Tensor,
    page_count: int,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    query_start_loc: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_c: int = 512,
    num_warps: int = 4,
    _validated: bool = False,
) -> torch.Tensor:
    """Compute verify conv1d(+silu) rows from the pre-convolution ring.

    Args:
        qkv_raw: ``[N*T, 3*P]`` FP16/BF16 pre-conv packed q|k|v, request-major.
        conv_w: ``[3*P, 4]`` contiguous convolution taps in the input dtype.
        read_indices: ``[N]`` committed page per request.
        live_indices: ``[N]`` private live page per request. A request takes
            part when its committed and live pages lie in ``(0,
            page_count)`` and its record slot in ``[0, slots)``, as in the
            retained prepare; other rows are padding.
        preconv: The layer's ``[slots, T + 3, 3*P]`` ring, in the input dtype
            or BF16 for FP16 inputs, and each request's ring head. A request
            reads its 3 previous inputs from rows ``(head - 3 + i) mod
            (T + 3)`` and writes input row t to row ``(head + t) mod
            (T + 3)``. History and current inputs are converted
            independently to FP32 for convolution. Padding rows read a zero
            history and write no ring row; their output rows still hold the
            convolution of their own inputs.
        record_slots: ``[N]`` record slot of each request in the ring.
        page_count: Pages in the State pool ``read_indices`` address.
        num_heads/head_dim: Per-rank head geometry (P = num_heads*head_dim).
        draft_token_num: Maximum draft positions per request (T).
        query_start_loc: Packed request boundaries, with one boundary per request.
        bias: Optional ``[3*P]`` convolution bias added before SiLU.
        out: Optional ``[N*T, 3*P]`` output tensor in the input dtype.
        block_c: Number of feature channels handled by each program.
        num_warps: Number of warps used by each program.
        _validated: Internal. Set by a split verify that already ran
            ``check_kda_split_verify_extents`` with this producer's operands
            and ``KdaPreconvRing.check_verify_storage`` on them (whose
            device, bias and storage checks cover the ones below); a
            supplied ``out`` is still checked here. Other callers leave it
            False.

    Returns:
        out: ``[N*T, 3*P]`` convolved and SiLU-activated q|k|v rows.
    """
    T = draft_token_num
    P = num_heads * head_dim
    C = 3 * P
    total = qkv_raw.shape[0]
    if T <= 0:
        raise ValueError(f"draft_token_num={T} must be positive")
    N = read_indices.numel()
    if (
        query_start_loc.shape != (N + 1,)
        or query_start_loc.dtype not in (torch.int32, torch.int64)
        or query_start_loc.device != qkv_raw.device
    ):
        raise ValueError(
            "query_start_loc must be an integer [requests + 1] boundary vector"
        )
    if qkv_raw.shape != (total, C) or qkv_raw.stride(-1) != 1:
        raise ValueError(f"qkv_raw must have shape [{total}, {C}] and last stride 1")
    if qkv_raw.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"qkv_raw must have dtype FP16 or BF16, got {qkv_raw.dtype}")
    if conv_w.shape != (C, 4) or not conv_w.is_contiguous():
        raise ValueError(f"conv_w must be contiguous with shape [{C}, 4]")
    if conv_w.dtype != qkv_raw.dtype:
        raise ValueError(f"conv_w must have dtype {qkv_raw.dtype}, got {conv_w.dtype}")
    ring = preconv.rows
    if ring.shape[1:] != (T + 3, C) or ring.device != qkv_raw.device:
        raise ValueError(
            f"the pre-convolution ring must have shape [slots, {T + 3}, {C}], got "
            f"{tuple(ring.shape)}"
        )
    if ring.dtype not in (qkv_raw.dtype, torch.bfloat16):
        raise ValueError(
            f"the pre-convolution ring must have dtype {qkv_raw.dtype} or "
            f"torch.bfloat16, got {ring.dtype}"
        )
    if read_indices.ndim != 1 or read_indices.numel() != N:
        raise ValueError(f"read_indices must have shape [{N}]")
    if any(
        x.shape != (N,) or x.dtype not in (torch.int32, torch.int64)
        for x in (read_indices, live_indices, record_slots, preconv.heads)
    ):
        raise ValueError(
            "read_indices, live_indices, record_slots and ring heads must be "
            "integer [requests] vectors"
        )
    if (
        isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or page_count < 1
    ):
        raise ValueError("page_count must be a positive page-pool size")
    if block_c <= 0 or block_c & (block_c - 1):
        raise ValueError(f"block_c={block_c} must be a positive power of two")
    supplied_out = out is not None
    if out is None:
        out = torch.empty_like(qkv_raw, memory_format=torch.contiguous_format)
    elif out.shape != (total, C) or out.stride(-1) != 1:
        raise ValueError(f"out must have shape [{total}, {C}] and last stride 1")
    elif out.dtype != qkv_raw.dtype:
        raise ValueError(f"out must have dtype {qkv_raw.dtype}, got {out.dtype}")
    if _validated:
        # The caller's checks covered every operand but an output it did not
        # see; one allocated above is fresh storage on the input's device.
        if supplied_out:
            if out.device != qkv_raw.device:
                raise ValueError(
                    "the verify convolution's operands must share one device"
                )
            preconv.check_separate_storage(out)
    else:
        # A supplied bias is added to all C channels (HAS_BIAS), so it must
        # hold them, even when an empty view would otherwise pass as present.
        if bias is not None and (
            bias.shape != (C,)
            or bias.stride(0) != 1
            or not bias.is_floating_point()
            or bias.device != qkv_raw.device
        ):
            raise ValueError(f"bias must be a unit-stride floating [{C}] vector")
        # The launch passes bare pointers: every operand must be on the
        # input's device (the ring's heads share the ring's, checked above).
        device = qkv_raw.device
        for operand in (conv_w, out, read_indices, live_indices, record_slots):
            if operand.device != device:
                raise ValueError(
                    "the verify convolution's operands must share one device"
                )
        # The producer writes ring rows while it reads every other operand.
        # The split verify after it also reads the checkpoint and records;
        # callers that run both check those too (see kda_verify_conv_update).
        preconv.check_verify_storage(
            qkv_raw,
            conv_w,
            bias,
            out,
            read_indices,
            live_indices,
            record_slots,
            query_start_loc,
        )
    # 16-byte row vectors need the slot base 16-byte aligned.
    vector = 16 // ring.element_size()
    # Cap BLOCK_T so BLOCK_T * block_c stays well below Triton's per-program
    # element limit even for large T; decode windows (T <= 16) stay in one tile.
    max_block_t = min(16, max(1, (1 << 16) // block_c))
    block_t = triton.next_power_of_2(min(T, max_block_t))
    grid = (N * triton.cdiv(T, block_t), triton.cdiv(C, block_c))
    fused_kda_verify_conv_update_kernel[grid](
        qkv_raw=qkv_raw,
        conv_w=conv_w,
        conv_bias=bias,
        ring=ring,
        read_indices=read_indices,
        live_indices=live_indices,
        record_slots=record_slots,
        ring_heads=preconv.heads,
        query_start_loc=query_start_loc,
        query_stride=query_start_loc.stride(0),
        read_stride=read_indices.stride(0),
        live_stride=live_indices.stride(0),
        slot_stride=record_slots.stride(0),
        head_stride=preconv.heads.stride(0),
        total_tokens=total,
        pages=page_count,
        slots=ring.shape[0],
        ring_slot_stride=ring.stride(0),
        out=out,
        stride_raw_tok=qkv_raw.stride(0),
        stride_out_tok=out.stride(0),
        stride_ring_row=ring.stride(1),
        RING_SLOT_ALIGN=vector if ring.stride(0) % vector == 0 else 1,
        T=T,
        BLOCK_T=block_t,
        C=C,
        BLOCK_C=block_c,
        HAS_BIAS=bias is not None,
        num_warps=num_warps,
    )
    return out
