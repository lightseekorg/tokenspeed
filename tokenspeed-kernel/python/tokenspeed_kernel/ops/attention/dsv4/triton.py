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

"""Triton DeepSeek V4 selected attention over a dense K/V workspace."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

DEEPSEEK_V4_HEAD_DIM = 512

__all__ = [
    "triton_dsv4_prefill",
]


@triton.jit
def _dsv4_sparse_attention_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    lens_ptr,
    sink_ptr,
    out_ptr,
    lse_ptr,
    num_heads,
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

    if sink_ptr is not None:
        max_logit = tl.load(sink_ptr + head_idx).to(tl.float32)
        denominator = tl.full((), 1.0, tl.float32)
    else:
        max_logit = tl.full((), -float("inf"), tl.float32)
        denominator = tl.full((), 0.0, tl.float32)
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
        # Empty shards/tiles must not introduce exp(-inf - -inf).
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        previous_scale = tl.exp(max_logit - safe_max)
        probabilities = tl.exp(logits - safe_max)
        probabilities = tl.where(valid, probabilities, 0.0)
        accumulator = accumulator * previous_scale + tl.sum(
            probabilities[:, None] * kv,
            axis=0,
        )
        denominator = denominator * previous_scale + tl.sum(probabilities, axis=0)
        max_logit = next_max

    if lse_ptr is not None:
        logsumexp = tl.where(
            denominator > 0.0, max_logit + tl.log(denominator), -float("inf")
        )
        tl.store(lse_ptr + token_idx * num_heads + head_idx, logsumexp)
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
        "metadata_dtypes": frozenset({torch.int32, torch.int64}),
        "sinks": frozenset({True, False}),
        "return_lse": frozenset({True, False}),
    },
    priority=Priority.PORTABLE,
)
def triton_dsv4_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
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
    if attn_sink is not None and attn_sink.numel() < q.shape[1]:
        raise ValueError("attention sink must provide one value per query head")

    output = out if out is not None else torch.empty_like(q)
    _dsv4_sparse_attention_kernel[(q.shape[0], q.shape[1])](
        q,
        kv_2d,
        indices_2d,
        lens,
        attn_sink,
        output,
        lse,
        q.shape[1],
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
