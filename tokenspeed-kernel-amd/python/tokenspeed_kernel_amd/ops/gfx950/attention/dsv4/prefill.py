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

"""DeepSeek V4 prefill selected-attention kernel for AMD GFX950."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = ["gluon_dsv4_prefill_gfx950"]


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
    mfma_score: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mfma_value: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )

    q_threads_d: gl.constexpr = min(64, HEAD_DIM // 8)
    q_threads_h: gl.constexpr = 64 // q_threads_d
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[q_threads_h, q_threads_d],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    kv_threads_d: gl.constexpr = min(64, HEAD_DIM // 8)
    kv_threads_k: gl.constexpr = 64 // kv_threads_d
    kv_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[kv_threads_d, kv_threads_k],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    out_threads_d: gl.constexpr = min(64, HEAD_DIM // 8)
    out_threads_h: gl.constexpr = 64 // out_threads_d
    out_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[out_threads_h, out_threads_d],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    q_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]],
        [BLOCK_H, HEAD_DIM],
        [1, 0],
    )
    kv_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]],
        [HEAD_DIM, TILE_K],
        [0, 1],
    )
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_score,
        k_width=8,
    )
    k_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_score,
        k_width=8,
    )
    p_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_value,
        k_width=4,
    )
    v_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_value,
        k_width=4,
    )

    token_idx = gl.program_id(axis=0)
    head_group_idx = gl.program_id(axis=1)
    head_offset = head_group_idx * BLOCK_H
    valid_len = gl.load(lens + token_idx).to(tl.int32)
    effective_len = gl.minimum(gl.maximum(valid_len, 0), SELECTED_WIDTH)
    num_tiles = gl.maximum(gl.cdiv(effective_len, TILE_K), 1)

    q_heads = head_offset + gl.arange(
        0,
        BLOCK_H,
        layout=gl.SliceLayout(1, q_load_layout),
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
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=q_shared,
        ptr=q,
        offsets=q_offsets.to(tl.int32),
        mask=(q_heads < num_heads)[:, None],
    )
    gl.amd.cdna4.async_copy.commit_group()

    local_k_load = gl.arange(
        0,
        TILE_K,
        layout=gl.SliceLayout(0, kv_load_layout),
    )
    local_k_mfma = gl.arange(
        0,
        TILE_K,
        layout=gl.SliceLayout(0, mfma_score),
    )
    kv_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(1, kv_load_layout))
    indices_base = token_idx.to(tl.int64) * stride_indices_t

    rows_load = gl.amd.cdna4.buffer_load(
        ptr=indices,
        offsets=indices_base.to(tl.int32) + local_k_load,
        mask=local_k_load < SELECTED_WIDTH,
        other=-1,
    )
    rows_mfma = gl.amd.cdna4.buffer_load(
        ptr=indices,
        offsets=indices_base.to(tl.int32) + local_k_mfma,
        mask=local_k_mfma < SELECTED_WIDTH,
        other=-1,
    )
    valid_load = (
        (local_k_load < effective_len)
        & (rows_load >= 0)
        & (rows_load.to(tl.int64) < num_kv_rows)
    )
    valid_mfma = (
        (local_k_mfma < effective_len)
        & (rows_mfma >= 0)
        & (rows_mfma.to(tl.int64) < num_kv_rows)
    )
    safe_rows = gl.where(valid_load, rows_load, 0)

    kv_shared = gl.allocate_shared_memory(
        kv.dtype.element_ty,
        [2, HEAD_DIM, TILE_K],
        layout=kv_shared_layout,
    )
    kv_offsets = safe_rows[None, :].to(tl.int64) * stride_kv_row + kv_dims[:, None].to(
        tl.int64
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=kv_shared.index(0),
        ptr=kv,
        offsets=kv_offsets.to(tl.int32),
        mask=valid_load[None, :],
    )
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.wait_group(1)
    q_dot = q_shared.load(q_dot_layout)

    score_heads = head_offset + gl.arange(
        0,
        BLOCK_H,
        layout=gl.SliceLayout(1, mfma_score),
    )
    valid_heads = score_heads < num_heads
    max_value = gl.load(
        attn_sink + score_heads,
        mask=valid_heads,
        other=0.0,
    ).to(gl.float32)
    denominator = gl.full(
        [BLOCK_H],
        1.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(1, mfma_score),
    )
    accumulator = gl.zeros(
        [BLOCK_H, HEAD_DIM],
        dtype=gl.float32,
        layout=mfma_value,
    )

    current_buffer = 0
    for tile_idx in range(num_tiles - 1):
        next_start = (tile_idx + 1) * TILE_K
        next_k_load = next_start + local_k_load
        next_k_mfma = next_start + local_k_mfma
        next_rows_load = gl.amd.cdna4.buffer_load(
            ptr=indices,
            offsets=indices_base.to(tl.int32) + next_k_load,
            mask=next_k_load < SELECTED_WIDTH,
            other=-1,
        )
        next_rows_mfma = gl.amd.cdna4.buffer_load(
            ptr=indices,
            offsets=indices_base.to(tl.int32) + next_k_mfma,
            mask=next_k_mfma < SELECTED_WIDTH,
            other=-1,
        )
        next_valid_load = (
            (next_k_load < effective_len)
            & (next_rows_load >= 0)
            & (next_rows_load.to(tl.int64) < num_kv_rows)
        )
        next_valid_mfma = (
            (next_k_mfma < effective_len)
            & (next_rows_mfma >= 0)
            & (next_rows_mfma.to(tl.int64) < num_kv_rows)
        )
        next_safe_rows = gl.where(next_valid_load, next_rows_load, 0)
        next_buffer = 1 - current_buffer
        next_kv_offsets = next_safe_rows[None, :].to(
            tl.int64
        ) * stride_kv_row + kv_dims[:, None].to(tl.int64)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=kv_shared.index(next_buffer),
            ptr=kv,
            offsets=next_kv_offsets.to(tl.int32),
            mask=next_valid_load[None, :],
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(1)

        current_kv = kv_shared.index(current_buffer)
        k_dot = current_kv.load(k_dot_layout)
        v_dot = current_kv.permute([1, 0]).load(v_dot_layout)
        scores = gl.zeros(
            [BLOCK_H, TILE_K],
            dtype=gl.float32,
            layout=mfma_score,
        )
        scores = gl.amd.cdna4.mfma(q_dot, k_dot, scores) * softmax_scale
        scores = gl.where(
            valid_heads[:, None] & valid_mfma[None, :],
            scores,
            -float("inf"),
        )

        tile_max = gl.max(scores, axis=1)
        next_max = gl.maximum(max_value, tile_max)
        safe_next_max = gl.where(next_max > -float("inf"), next_max, 0.0)
        previous_scale = gl.exp(max_value - safe_next_max)
        probabilities = gl.exp(scores - safe_next_max[:, None])
        denominator = previous_scale * denominator + gl.sum(probabilities, axis=1)
        accumulator_scale = gl.convert_layout(
            previous_scale,
            gl.SliceLayout(1, mfma_value),
        )
        accumulator *= accumulator_scale[:, None]
        p_dot = gl.convert_layout(probabilities.to(q.dtype.element_ty), p_dot_layout)
        accumulator = gl.amd.cdna4.mfma(p_dot, v_dot, accumulator)

        max_value = next_max
        current_buffer = next_buffer
        valid_mfma = next_valid_mfma

    gl.amd.cdna4.async_copy.wait_group(0)
    current_kv = kv_shared.index(current_buffer)
    k_dot = current_kv.load(k_dot_layout)
    v_dot = current_kv.permute([1, 0]).load(v_dot_layout)
    scores = gl.zeros(
        [BLOCK_H, TILE_K],
        dtype=gl.float32,
        layout=mfma_score,
    )
    scores = gl.amd.cdna4.mfma(q_dot, k_dot, scores) * softmax_scale
    scores = gl.where(
        valid_heads[:, None] & valid_mfma[None, :],
        scores,
        -float("inf"),
    )

    tile_max = gl.max(scores, axis=1)
    next_max = gl.maximum(max_value, tile_max)
    safe_next_max = gl.where(next_max > -float("inf"), next_max, 0.0)
    previous_scale = gl.exp(max_value - safe_next_max)
    probabilities = gl.exp(scores - safe_next_max[:, None])
    denominator = previous_scale * denominator + gl.sum(probabilities, axis=1)
    accumulator_scale = gl.convert_layout(
        previous_scale,
        gl.SliceLayout(1, mfma_value),
    )
    accumulator *= accumulator_scale[:, None]
    p_dot = gl.convert_layout(probabilities.to(q.dtype.element_ty), p_dot_layout)
    accumulator = gl.amd.cdna4.mfma(p_dot, v_dot, accumulator)

    denominator_value = gl.convert_layout(
        denominator,
        gl.SliceLayout(1, mfma_value),
    )
    safe_denominator = gl.where(denominator_value > 0.0, denominator_value, 1.0)
    accumulator /= safe_denominator[:, None]
    accumulator = gl.where(
        denominator_value[:, None] > 0.0,
        accumulator,
        0.0,
    )

    out_heads = head_offset + gl.arange(
        0,
        BLOCK_H,
        layout=gl.SliceLayout(1, out_layout),
    )
    out_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, out_layout))
    out_offsets = (
        token_idx.to(tl.int64) * stride_o_t
        + out_heads[:, None].to(tl.int64) * stride_o_h
        + out_dims[None, :].to(tl.int64)
    )
    output = gl.convert_layout(accumulator.to(out.dtype.element_ty), out_layout)
    gl.amd.cdna4.buffer_store(
        stored_value=output,
        ptr=out,
        offsets=out_offsets.to(tl.int32),
        mask=(out_heads < num_heads)[:, None],
    )


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


def gluon_dsv4_prefill_gfx950(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run dense-workspace selected attention for DeepSeek V4 on GFX950.

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
        TILE_K=32,
        HEAD_DIM=512,
        num_warps=4,
    )
    return output
