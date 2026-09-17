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

"""Dense-workspace DeepSeek V4 selected-attention prefill for GFX1250.

Wave32 WMMA-v3 port of the generic H=16 GFX950 ``dsv4_prefill`` path.
Prefill gathers SWA/global rows into a BF16 workspace first; this kernel
only runs selected attention over that dense buffer.
"""

from __future__ import annotations

import math
import os

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = ["gluon_dsv4_prefill_gfx1250"]


@gluon.jit
def _dsv4_prefill_kernel(
    q,
    kv,
    indices,
    lens,
    attn_sink,
    out,
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_row: tl.int64,
    stride_indices_t: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    softmax_scale: tl.float32,
    num_heads: tl.int32,
    num_kv_rows: tl.int64,
    SELECTED_WIDTH: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    HEAD_DIM: gl.constexpr,
):
    WARP_SIZE: gl.constexpr = 32
    NUM_WARPS: gl.constexpr = gl.num_warps()
    K_WIDTH: gl.constexpr = 8
    qk_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[1, 0], [2, 0]],
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    pv_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2]],
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 16],
        [1, WARP_SIZE],
        [NUM_WARPS, 1],
        [1, 0],
    )
    kv_load_layout: gl.constexpr = gl.BlockedLayout(
        [16, 1],
        [WARP_SIZE, 1],
        [1, NUM_WARPS],
        [0, 1],
    )
    q_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HEAD_DIM, K_WIDTH]],
        [BLOCK_H, HEAD_DIM],
        [1, 0],
    )
    kv_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HEAD_DIM, K_WIDTH]],
        [HEAD_DIM, TILE_K],
        [0, 1],
    )
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(0, qk_layout, K_WIDTH)
    k_dot_layout: gl.constexpr = gl.DotOperandLayout(1, qk_layout, K_WIDTH)
    p_dot_layout: gl.constexpr = gl.DotOperandLayout(0, pv_layout, K_WIDTH)
    v_dot_layout: gl.constexpr = gl.DotOperandLayout(1, pv_layout, K_WIDTH)

    token_idx = gl.program_id(axis=0)
    head_group_idx = gl.program_id(axis=1)
    head_offset = head_group_idx * BLOCK_H
    valid_len = gl.load(lens + token_idx).to(tl.int32)
    effective_len = gl.minimum(gl.maximum(valid_len, 0), SELECTED_WIDTH)
    num_tiles = gl.maximum(gl.cdiv(effective_len, TILE_K), 1)

    q_heads = head_offset + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, q_load_layout)
    )
    q_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_load_layout))
    q_offsets = (
        token_idx.to(tl.int64) * stride_q_t
        + q_heads[:, None].to(tl.int64) * stride_q_h
        + q_dims[None, :].to(tl.int64)
    )
    q_shared = gl.allocate_shared_memory(
        q.dtype.element_ty,
        [BLOCK_H, HEAD_DIM],
        layout=q_shared_layout,
    )
    q_shared.store(
        gl.load(q + q_offsets, mask=(q_heads < num_heads)[:, None], other=0.0)
    )
    kv_shared = gl.allocate_shared_memory(
        q.dtype.element_ty,
        [HEAD_DIM, TILE_K],
        layout=kv_shared_layout,
    )
    gl.barrier()
    q_dot = q_shared.load(q_dot_layout)

    score_heads = head_offset + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, qk_layout)
    )
    valid_heads = score_heads < num_heads
    max_value = gl.load(attn_sink + score_heads, mask=valid_heads, other=0.0).to(
        gl.float32
    )
    denominator = gl.full(
        [BLOCK_H],
        1.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(1, qk_layout),
    )
    accumulator = gl.zeros([BLOCK_H, HEAD_DIM], dtype=gl.float32, layout=pv_layout)

    local_k_load = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, kv_load_layout))
    kv_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(1, kv_load_layout))
    indices_base = token_idx.to(tl.int64) * stride_indices_t

    for tile_idx in range(num_tiles):
        positions = tile_idx * TILE_K + local_k_load
        rows_load = gl.load(
            indices + indices_base + positions.to(tl.int64),
            mask=positions < SELECTED_WIDTH,
            other=-1,
        )
        valid_load = (
            (positions < effective_len)
            & (rows_load >= 0)
            & (rows_load.to(tl.int64) < num_kv_rows)
        )
        safe_rows = gl.where(valid_load, rows_load, 0).to(tl.int64)
        kv_values = gl.load(
            kv + safe_rows[None, :] * stride_kv_row + kv_dims[:, None].to(tl.int64),
            mask=valid_load[None, :],
            other=0.0,
        )
        valid_col = gl.convert_layout(valid_load, gl.SliceLayout(0, qk_layout))
        kv_shared.store(kv_values)
        gl.barrier()

        k_dot = kv_shared.load(k_dot_layout)
        v_dot = kv_shared.permute([1, 0]).load(v_dot_layout)
        scores = (
            gl.amd.cdna5.wmma(
                q_dot,
                k_dot,
                gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=qk_layout),
            )
            * softmax_scale
        )
        scores = gl.where(
            valid_heads[:, None] & valid_col[None, :],
            scores,
            -float("inf"),
        )

        tile_max = gl.max(scores, axis=1)
        next_max = gl.maximum(max_value, tile_max)
        safe_next_max = gl.where(next_max > -float("inf"), next_max, 0.0)
        previous_scale = gl.exp(max_value - safe_next_max)
        probabilities = gl.exp(scores - safe_next_max[:, None])
        denominator = previous_scale * denominator + gl.sum(probabilities, axis=1)
        accumulator *= gl.convert_layout(previous_scale[:, None], pv_layout)
        p_dot = gl.convert_layout(probabilities.to(q.dtype.element_ty), p_dot_layout)
        accumulator = gl.amd.cdna5.wmma(p_dot, v_dot, accumulator)
        max_value = next_max
        gl.barrier()

    denominator_value = gl.convert_layout(denominator, gl.SliceLayout(1, pv_layout))
    safe_denominator = gl.where(denominator_value > 0.0, denominator_value, 1.0)
    accumulator /= safe_denominator[:, None]
    accumulator = gl.where(denominator_value[:, None] > 0.0, accumulator, 0.0)

    out_heads = head_offset + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, q_load_layout)
    )
    out_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_load_layout))
    out_offsets = (
        token_idx.to(tl.int64) * stride_o_t
        + out_heads[:, None].to(tl.int64) * stride_o_h
        + out_dims[None, :].to(tl.int64)
    )
    output = gl.convert_layout(accumulator.to(out.dtype.element_ty), q_load_layout)
    gl.store(out + out_offsets, output, mask=(out_heads < num_heads)[:, None])


def _tile_k() -> int:
    value = int(os.environ.get("TOKENSPEED_DSV4_PREFILL_TILE_K", "32"))
    if value not in (32, 64):
        raise ValueError("TOKENSPEED_DSV4_PREFILL_TILE_K must be 32 or 64 on gfx1250")
    return value


def _check_tensor(name: str, tensor: object) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return tensor


def _shares_storage(tensor: torch.Tensor, other: torch.Tensor) -> bool:
    if tensor.numel() == 0 or other.numel() == 0:
        return False
    return tensor.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()


def _validate_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor | None,
) -> None:
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must be BF16, got {q.dtype}")
    if q.dim() != 3 or q.shape[2] != 512:
        raise ValueError(
            f"q must have shape [tokens, heads, 512], got {tuple(q.shape)}"
        )
    if not q.is_cuda:
        raise ValueError("q must be on an AMD GPU")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")

    if kv.dtype != torch.bfloat16:
        raise TypeError(f"kv must be BF16, got {kv.dtype}")
    if kv.device != q.device:
        raise ValueError("kv must be on the same device as q")
    if not kv.is_contiguous() or kv.numel() % 512 != 0:
        raise ValueError("kv must be contiguous and reshapeable to [-1, 512]")

    if indices.dtype != torch.int32:
        raise TypeError(f"indices must be int32, got {indices.dtype}")
    if indices.dim() != 2 or indices.shape[0] != q.shape[0]:
        raise ValueError(
            "indices must have shape [tokens, selected_width], got "
            f"{tuple(indices.shape)}"
        )
    if indices.device != q.device or not indices.is_contiguous():
        raise ValueError("indices must be contiguous and on the same device as q")

    if lens.dtype != torch.int32:
        raise TypeError(f"lens must be int32, got {lens.dtype}")
    if lens.shape != (q.shape[0],):
        raise ValueError(f"lens must have shape [tokens], got {tuple(lens.shape)}")
    if lens.device != q.device or not lens.is_contiguous():
        raise ValueError("lens must be contiguous and on the same device as q")

    if attn_sink.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"attn_sink must be FP32 or BF16, got {attn_sink.dtype}")
    if attn_sink.device != q.device or not attn_sink.is_contiguous():
        raise ValueError("attn_sink must be contiguous and on the same device as q")
    if attn_sink.numel() < q.shape[1]:
        raise ValueError("attn_sink must provide at least one value per query head")

    if out is None:
        return
    if out.dtype != torch.bfloat16:
        raise TypeError(f"out must be BF16, got {out.dtype}")
    if out.shape != q.shape:
        raise ValueError(
            f"out must have exact shape {tuple(q.shape)}, got {tuple(out.shape)}"
        )
    if out.device != q.device or not out.is_contiguous():
        raise ValueError("out must be contiguous and on the same device as q")
    for name, tensor in (
        ("q", q),
        ("kv", kv),
        ("indices", indices),
        ("lens", lens),
        ("attn_sink", attn_sink),
    ):
        if _shares_storage(out, tensor):
            raise ValueError(f"out must not alias {name}")


def gluon_dsv4_prefill_gfx1250(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run dense-workspace selected attention for DeepSeek V4 on GFX1250.

    Args:
        q: Contiguous BF16 queries shaped `[tokens, heads, 512]`.
        kv: Contiguous BF16 storage reshapeable to rows of 512 channels.
        indices: Contiguous int32 selected-row indices shaped
            `[tokens, selected_width]`. Negative entries are ignored.
        lens: Contiguous int32 valid selected widths shaped `[tokens]`.
        attn_sink: Contiguous FP32 or BF16 sink logits with at least one value
            per query head.
        softmax_scale: Scale applied to query-key dot products.
        out: Optional exact contiguous BF16 output shaped like `q`.

    Returns:
        The BF16 selected-attention output shaped `[tokens, heads, 512]`.
    """

    q = _check_tensor("q", q)
    kv = _check_tensor("kv", kv)
    indices = _check_tensor("indices", indices)
    lens = _check_tensor("lens", lens)
    attn_sink = _check_tensor("attn_sink", attn_sink)
    if out is not None:
        out = _check_tensor("out", out)
    _validate_inputs(q, kv, indices, lens, attn_sink, out)

    try:
        scale = float(softmax_scale)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("softmax_scale must be a finite real scalar") from error
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")

    output = out if out is not None else torch.empty_like(q)
    if q.shape[0] == 0 or q.shape[1] == 0 or indices.shape[1] == 0:
        output.zero_()
        return output

    kv_rows = kv.reshape(-1, 512)
    sink_values = attn_sink.reshape(-1)
    grid = (q.shape[0], triton.cdiv(q.shape[1], 16))
    _dsv4_prefill_kernel[grid](
        q,
        kv_rows,
        indices,
        lens,
        sink_values,
        output,
        q.stride(0),
        q.stride(1),
        kv_rows.stride(0),
        indices.stride(0),
        output.stride(0),
        output.stride(1),
        scale,
        q.shape[1],
        kv_rows.shape[0],
        SELECTED_WIDTH=indices.shape[1],
        BLOCK_H=16,
        TILE_K=_tile_k(),
        HEAD_DIM=512,
        num_warps=4,
    )
    return output
