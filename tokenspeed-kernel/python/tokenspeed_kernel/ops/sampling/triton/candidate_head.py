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

"""EAGLE-4 low-rank candidate head: the exact logits of each row's candidates,
``out[b, k] = weight[indices[b, k] - id_offset] . hidden[b]`` in fp32."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["gather_dot"]


@triton.jit
def _gather_dot_kernel(
    weight_ptr,  # [V, H] LM head weight
    indices_ptr,  # [B, K] candidate vocab ids, id_offset at weight row 0
    hidden_ptr,  # [B, H]
    output_ptr,  # [B, K]
    stride_w_v,
    stride_w_h,
    stride_i_b,
    stride_i_k,
    stride_h_b,
    stride_h_h,
    stride_o_b,
    stride_o_k,
    K,
    id_offset,
    H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # One program per BLOCK_K candidates of a row loads the row's hidden once per tile.
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    # int64: vocab_id * stride_w_v can exceed int32 for a large vocab x hidden.
    vocab_ids = tl.load(
        indices_ptr + pid_b * stride_i_b + k_offs * stride_i_k, mask=k_mask, other=0
    ).to(tl.int64)
    vocab_ids -= id_offset

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    EVEN_H: tl.constexpr = H % BLOCK_H == 0
    for off_h in range(0, H, BLOCK_H):
        cols = off_h + tl.arange(0, BLOCK_H)
        if EVEN_H:
            h_val = tl.load(hidden_ptr + pid_b * stride_h_b + cols * stride_h_h)
            w_val = tl.load(
                weight_ptr
                + vocab_ids[:, None] * stride_w_v
                + cols[None, :] * stride_w_h,
                mask=k_mask[:, None],
                other=0.0,
            )
        else:
            h_mask = cols < H
            h_val = tl.load(
                hidden_ptr + pid_b * stride_h_b + cols * stride_h_h,
                mask=h_mask,
                other=0.0,
            )
            w_val = tl.load(
                weight_ptr
                + vocab_ids[:, None] * stride_w_v
                + cols[None, :] * stride_w_h,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
        # Products and sums in fp32: the one rounding is the store's.
        acc += tl.sum(w_val.to(tl.float32) * h_val.to(tl.float32)[None, :], axis=1)

    tl.store(
        output_ptr + pid_b * stride_o_b + k_offs * stride_o_k,
        acc.to(output_ptr.dtype.element_ty),
        mask=k_mask,
    )


# (candidates, hidden) where BLOCK_K 1 was validated against BLOCK_K 4 (same H reduction layout).
_ONE_CANDIDATE_SHAPES = frozenset({(128, 5120)})


def _tile_config(num_rows: int, num_candidates: int, hidden: int, dtype: torch.dtype):
    """Static (BLOCK_K, BLOCK_H, num_warps, num_stages) per row bucket: few rows
    are latency bound and want deep H tiles, many rows bandwidth bound and
    want wide K tiles. Every bucket up to 32 rows reduces along H in the same
    layout, so their results agree bit for bit."""
    if num_rows <= 2:
        return 2, 2048, 8, 3
    if num_rows <= 32:
        shape = (num_candidates, hidden)
        one = shape in _ONE_CANDIDATE_SHAPES and dtype == torch.bfloat16
        return (1 if one else 4), 2048, 8, 3
    return 32, 256, 4, 3


def gather_dot(
    weight: torch.Tensor,
    indices: torch.Tensor,
    hidden_states: torch.Tensor,
    id_offset: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score gathered LM head rows:
    ``out[b, k] = weight[indices[b, k] - id_offset] . hidden[b]``.

    Args:
        weight: ``[V, H]`` LM head weight (this rank's vocab shard).
        indices: ``[B, K]`` vocab ids of each row's candidates, any strides.
        hidden_states: ``[B, H]`` in the weight's dtype.
        id_offset: The vocab id of ``weight`` row 0 (the shard's first id).
        out: ``[B, K]`` destination in the hidden dtype, any strides (e.g. a
            block of a packed all-gather buffer); a new tensor when None.

    Returns:
        The ``[B, K]`` scores, computed in fp32 and stored in the hidden dtype.
    """
    num_rows, num_candidates = indices.shape
    hidden = weight.shape[1]
    if out is None:
        out = torch.empty(
            (num_rows, num_candidates), device=indices.device, dtype=hidden_states.dtype
        )
    if num_rows == 0:
        return out
    block_k, block_h, num_warps, num_stages = _tile_config(
        num_rows, num_candidates, hidden, hidden_states.dtype
    )
    block_h = min(block_h, triton.next_power_of_2(hidden))
    _gather_dot_kernel[(num_rows, triton.cdiv(num_candidates, block_k))](
        weight,
        indices,
        hidden_states,
        out,
        weight.stride(0),
        weight.stride(1),
        indices.stride(0),
        indices.stride(1),
        hidden_states.stride(0),
        hidden_states.stride(1),
        out.stride(0),
        out.stride(1),
        num_candidates,
        id_offset,
        H=hidden,
        BLOCK_K=block_k,
        BLOCK_H=block_h,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
