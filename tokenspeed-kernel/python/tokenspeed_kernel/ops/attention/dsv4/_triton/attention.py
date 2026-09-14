# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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

"""Selected DSV4 attention over dense and page-planar KV caches."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv4._triton.common import (
    DEEPSEEK_V4_FP8_QUANT_BLOCK,
    DEEPSEEK_V4_HEAD_DIM,
    DEEPSEEK_V4_NOPE_DIM,
    DEEPSEEK_V4_ROPE_DIM,
    DEEPSEEK_V4_SWA_SCALE_DIM,
    DEEPSEEK_V4_SWA_TOKEN_STRIDE,
)
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@triton.jit
def _dsv4_sparse_attention_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    lens_ptr,
    sink_ptr,
    out_ptr,
    q_stride_token,
    q_stride_head,
    kv_stride_row,
    indices_stride_token,
    out_stride_token,
    out_stride_head,
    softmax_scale,
    num_kv_rows,
    TOPK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim = tl.arange(0, BLOCK_DIM)
    dim_mask = dim < HEAD_DIM
    q = tl.load(
        q_ptr + token_idx * q_stride_token + head_idx * q_stride_head + dim,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    max_logit = tl.load(sink_ptr + head_idx).to(tl.float32)
    denominator = tl.full((), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_DIM,), tl.float32)
    valid_len = tl.minimum(tl.maximum(tl.load(lens_ptr + token_idx), 0), TOPK)
    topk_offsets = tl.arange(0, BLOCK_TOPK)

    for start in range(0, TOPK, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        rows = tl.load(
            indices_ptr + token_idx * indices_stride_token + cols,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid = valid & (rows >= 0) & (rows < num_kv_rows)
        rows = tl.where(valid, rows, 0)
        kv = tl.load(
            kv_ptr + rows[:, None] * kv_stride_row + dim[None, :],
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(kv * q[None, :], axis=1) * softmax_scale
        logits = tl.where(valid, logits, -float("inf"))
        block_max = tl.max(logits, axis=0)
        next_max = tl.maximum(max_logit, block_max)
        previous_scale = tl.exp(max_logit - next_max)
        probabilities = tl.exp(logits - next_max)
        probabilities = tl.where(valid, probabilities, 0.0)
        accumulator = accumulator * previous_scale + tl.sum(
            probabilities[:, None] * kv,
            axis=0,
        )
        denominator = denominator * previous_scale + tl.sum(probabilities, axis=0)
        max_logit = next_max

    output = tl.where(denominator > 0.0, accumulator / denominator, 0.0)
    tl.store(
        out_ptr + token_idx * out_stride_token + head_idx * out_stride_head + dim,
        output,
        mask=dim_mask,
    )


@register_kernel(
    "attention",
    "dsv4_prefill",
    name="triton_dsv4_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                q=dense_tensor_format(torch.bfloat16),
                kv=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    traits={
        "head_dim": frozenset({DEEPSEEK_V4_HEAD_DIM}),
        "cache_layout": frozenset({"dense_workspace"}),
        "support_sink": frozenset({True}),
        "metadata_dtypes": frozenset({torch.int32, torch.int64}),
    },
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_dsv4_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run selected shared-KV attention for DeepSeek V4 geometry."""

    if q.dim() != 3 or q.shape[-1] != DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(f"expected q [tokens, heads, 512], got {tuple(q.shape)}")
    kv_2d = kv.reshape(-1, kv.shape[-1])
    if kv_2d.shape[-1] != DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(f"expected kv rows of width 512, got {tuple(kv.shape)}")
    indices_2d = indices.reshape(indices.shape[0], -1).contiguous()
    lens = lens.reshape(-1).contiguous()
    if indices_2d.shape[0] != q.shape[0] or lens.shape[0] != q.shape[0]:
        raise ValueError("selected-attention metadata must have one row per query")
    if attn_sink.numel() < q.shape[1]:
        raise ValueError("attention sink must provide one value per query head")

    output = out if out is not None else torch.empty_like(q)
    _dsv4_sparse_attention_kernel[(q.shape[0], q.shape[1])](
        q,
        kv_2d,
        indices_2d,
        lens,
        attn_sink,
        output,
        q.stride(0),
        q.stride(1),
        kv_2d.stride(0),
        indices_2d.stride(0),
        output.stride(0),
        output.stride(1),
        softmax_scale,
        kv_2d.shape[0],
        TOPK=indices_2d.shape[1],
        HEAD_DIM=DEEPSEEK_V4_HEAD_DIM,
        BLOCK_TOPK=16,
        BLOCK_DIM=triton.next_power_of_2(DEEPSEEK_V4_HEAD_DIM),
        num_warps=4,
        num_stages=1,
    )
    return output


@triton.jit
def _dsv4_dequantize_selected_cache_rows_kernel(
    cache_ptr,
    slots_ptr,
    lens_ptr,
    out_ptr,
    indices_ptr,
    selected_lens_ptr,
    cache_block_stride,
    slots_stride_token,
    indices_stride_token,
    out_stride_token,
    out_stride_row,
    block_size,
    cache_capacity,
    OUTPUT_ROW_OFFSET: tl.constexpr,
    INDEX_OFFSET: tl.constexpr,
    WORKSPACE_WIDTH: tl.constexpr,
    HAS_METADATA: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    slot = tl.load(slots_ptr + token_idx * slots_stride_token + row_idx).to(tl.int64)
    valid = (slot >= 0) & (slot < cache_capacity)
    if HAS_METADATA:
        valid &= row_idx < tl.load(lens_ptr + token_idx)
    safe_slot = tl.where(valid, slot, 0)
    cache_block = safe_slot // block_size
    cache_position = safe_slot % block_size
    block_base = cache_ptr + cache_block * cache_block_stride
    token_base = block_base + cache_position * TOKEN_STRIDE
    scale_base = block_base + block_size * TOKEN_STRIDE + cache_position * SCALE_DIM
    out_base = (
        out_ptr
        + token_idx * out_stride_token
        + (OUTPUT_ROW_OFFSET + row_idx) * out_stride_row
    )

    dim = tl.arange(0, BLOCK_DIM)
    nope_mask = dim < NOPE_DIM
    values_u8 = tl.load(token_base + dim, mask=valid & nope_mask, other=0)
    values_fp8 = values_u8.to(tl.float8e4nv, bitcast=True)
    scale_idx = dim // QUANT_BLOCK
    exponent = (
        tl.load(
            scale_base + scale_idx,
            mask=valid & nope_mask,
            other=127,
        ).to(tl.float32)
        - 127.0
    )
    nope = values_fp8.to(tl.float32) * tl.exp2(exponent)
    tl.store(out_base + dim, nope, mask=nope_mask)

    rope_offsets = tl.arange(0, ROPE_DIM)
    rope_ptr = (token_base + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
    rope = tl.load(rope_ptr + rope_offsets, mask=valid, other=0.0)
    tl.store(out_base + NOPE_DIM + rope_offsets, rope)
    if HAS_METADATA:
        flat_index = token_idx * WORKSPACE_WIDTH + INDEX_OFFSET + row_idx
        tl.store(
            indices_ptr + token_idx * indices_stride_token + INDEX_OFFSET + row_idx,
            tl.where(valid, flat_index, -1),
        )
        tl.store(
            selected_lens_ptr + token_idx,
            WORKSPACE_WIDTH,
            mask=row_idx == 0,
        )


def _dsv4_dequantize_selected_cache_segment(
    cache_2d: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    block_size: int,
    output: torch.Tensor,
    indices: torch.Tensor,
    selected_lens: torch.Tensor,
    output_row_offset: int,
    workspace_width: int,
) -> None:
    """Dequantize one cache segment and produce flattened attention metadata."""
    slots_2d = slots.reshape(slots.shape[0], -1).contiguous()
    lens_1d = lens.reshape(-1).contiguous()
    if slots_2d.shape[0] != output.shape[0] or lens_1d.shape[0] != output.shape[0]:
        raise ValueError("selected cache segment must have one row per query")
    if output_row_offset + slots_2d.shape[1] > output.shape[1]:
        raise ValueError("selected cache segment exceeds the output workspace")
    if slots_2d.numel() == 0:
        return
    _dsv4_dequantize_selected_cache_rows_kernel[(slots_2d.shape[0], slots_2d.shape[1])](
        cache_2d,
        slots_2d,
        lens_1d,
        output,
        indices,
        selected_lens,
        cache_2d.stride(0),
        slots_2d.stride(0),
        indices.stride(0),
        output.stride(0),
        output.stride(1),
        block_size,
        cache_2d.shape[0] * block_size,
        OUTPUT_ROW_OFFSET=output_row_offset,
        INDEX_OFFSET=output_row_offset,
        WORKSPACE_WIDTH=workspace_width,
        HAS_METADATA=True,
        HEAD_DIM=DEEPSEEK_V4_HEAD_DIM,
        NOPE_DIM=DEEPSEEK_V4_NOPE_DIM,
        ROPE_DIM=DEEPSEEK_V4_ROPE_DIM,
        QUANT_BLOCK=DEEPSEEK_V4_FP8_QUANT_BLOCK,
        TOKEN_STRIDE=DEEPSEEK_V4_SWA_TOKEN_STRIDE,
        SCALE_DIM=DEEPSEEK_V4_SWA_SCALE_DIM,
        BLOCK_DIM=triton.next_power_of_2(DEEPSEEK_V4_HEAD_DIM),
        num_warps=4,
    )


@register_kernel(
    "attention",
    "dsv4_decode",
    name="triton_dsv4_decode",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                q=dense_tensor_format(torch.bfloat16),
                swa_kv_cache=dense_tensor_format(torch.uint8),
            )
        }
    ),
    traits={
        "head_dim": frozenset({DEEPSEEK_V4_HEAD_DIM}),
        "cache_layout": frozenset({"fp8_swa_page_planar"}),
        "topk_layout": frozenset({"global_slots"}),
        "support_sink": frozenset({True}),
        "has_extra_segment": frozenset({False, True}),
        "metadata_dtypes": frozenset({torch.int32, torch.int64}),
    },
    priority=Priority.PORTABLE,
    tags={"portability", "paged_cache", "selected_attention"},
)
def triton_dsv4_decode(
    q: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    swa_slots: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_page_size: int,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    extra_kv_cache: torch.Tensor | None = None,
    extra_slots: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
    extra_page_size: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose page-planar dequantization with registered dense attention."""
    from tokenspeed_kernel.ops.attention.dsv4 import dsv4_prefill

    tokens = q.shape[0]
    swa_width = swa_slots.numel() // tokens
    extra_width = 0 if extra_slots is None else extra_slots.numel() // tokens
    workspace_width = swa_width + extra_width
    kv_workspace = torch.empty(
        (tokens, workspace_width, DEEPSEEK_V4_HEAD_DIM),
        dtype=torch.bfloat16,
        device=q.device,
    )
    selected_indices = torch.empty(
        (tokens, workspace_width), dtype=torch.int32, device=q.device
    )
    selected_lens = torch.empty((tokens,), dtype=torch.int32, device=q.device)
    _dsv4_dequantize_selected_cache_segment(
        swa_kv_cache,
        swa_slots,
        swa_lens,
        swa_page_size,
        kv_workspace,
        selected_indices,
        selected_lens,
        0,
        workspace_width,
    )
    if extra_kv_cache is not None:
        assert extra_slots is not None
        assert extra_lens is not None
        assert extra_page_size is not None
        _dsv4_dequantize_selected_cache_segment(
            extra_kv_cache,
            extra_slots,
            extra_lens,
            extra_page_size,
            kv_workspace,
            selected_indices,
            selected_lens,
            swa_width,
            workspace_width,
        )
    return dsv4_prefill(
        q=q,
        kv=kv_workspace,
        indices=selected_indices,
        lens=selected_lens,
        attn_sink=attn_sink,
        softmax_scale=softmax_scale,
        out=out,
    )
