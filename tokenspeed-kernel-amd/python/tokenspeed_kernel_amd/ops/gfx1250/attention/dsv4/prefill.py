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

Prefill gathers SWA/global rows into a BF16 workspace first; this kernel
only runs selected attention over that dense buffer.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = ["launch_gluon_dsv4_prefill_gfx1250"]

_LOG2E = 1.4426950408889634
_BLOCK_H = 16
# Four waves split a tile's rows for QK, 16 rows each.
_TILE_K = 64
_NUM_WARPS = 4
_NUM_XCDS = 8


def _prefill_launch_metadata(grid, kernel, args):
    """Report selected-attention capacity without reading device-resident lengths."""
    q = args["q"]
    tokens, heads = q.shape[0], q.shape[1]
    width = args["SELECTED_WIDTH"]
    head_dim = args["HEAD_DIM"]
    return {
        "name": kernel.name,
        # QK^T and PV over every selected slot.
        "flops16": 4 * tokens * heads * width * head_dim,
        # Each workgroup gathers its token's selected rows (and indices) once
        # for 16 heads, and reads q / writes out once.
        "bytes": tokens * grid[1] * width * (head_dim * 2 + 4)
        + 2 * q.numel() * q.element_size(),
    }


@gluon.jit
def _tile_rows(indices, index_row, tile_idx, length, num_kv_rows, TILE_K, LAYOUT):
    """Return (clamped row, valid) for one tile of a token's selected rows."""
    positions = tile_idx * TILE_K + gl.arange(0, TILE_K, layout=LAYOUT)
    in_range = positions < length
    rows = gl.amd.cdna5.buffer_load(
        indices + index_row, positions, mask=in_range, other=-1
    )
    valid = in_range & (rows >= 0) & (rows < num_kv_rows)
    return gl.where(valid, rows, 0), valid


@gluon.jit(
    launch_metadata=_prefill_launch_metadata,
    do_not_specialize=("num_heads", "num_kv_rows"),
)
def gluon_dsv4_prefill_gfx1250(
    q,
    kv,
    indices,
    lens,
    attn_sink,
    out,
    qk_scale_log2: tl.float32,
    num_heads,
    num_kv_rows,
    SELECTED_WIDTH: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    # One workgroup per (token, 16-head group). The 16 heads are the WMMA M
    # dimension, so each gathered KV row feeds all heads. The four waves split
    # a tile's rows for QK and the head dims for PV.
    wmma: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2]],
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma, k_width=8
    )
    kv_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma, k_width=8
    )
    # A lane loads 16 contiguous bytes of a row.
    kv_load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    out_layout: gl.constexpr = gl.BlockedLayout([1, 16], [1, 32], [4, 1], [1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, kv_load_layout)
    score_row_layout: gl.constexpr = gl.SliceLayout(0, wmma)
    head_layout: gl.constexpr = gl.SliceLayout(1, wmma)
    kv_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [TILE_K, HEAD_DIM], [1, 0]
    )

    # Workgroups are dispatched round-robin over the XCDs; give each XCD a
    # contiguous token range so neighbouring tokens' shared KV rows (sliding
    # window, overlapping top-k) hit in that XCD's L2.
    pid = gl.program_id(axis=0)
    num_tokens = gl.num_programs(axis=0)
    xcd = pid % NUM_XCDS
    token_idx = (
        xcd * (num_tokens // NUM_XCDS)
        + gl.minimum(xcd, num_tokens % NUM_XCDS)
        + pid // NUM_XCDS
    )
    head_offset = gl.program_id(axis=1) * BLOCK_H
    valid_len = gl.load(lens + token_idx)
    effective_len = gl.minimum(gl.maximum(valid_len, 0), SELECTED_WIDTH)
    num_tiles = gl.cdiv(effective_len, TILE_K)
    index_row = token_idx.to(tl.int64) * SELECTED_WIDTH

    # q, out, and kv are contiguous with 512-element rows (checked on launch).
    dot_heads = head_offset + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, q_dot_layout)
    )
    dot_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_dot_layout))
    q_dot = gl.load(
        q
        + (token_idx.to(tl.int64) * num_heads + dot_heads[:, None]) * HEAD_DIM
        + dot_dims[None, :],
        mask=(dot_heads < num_heads)[:, None],
        other=0.0,
    )

    # Online softmax in the log2 domain. The sink seeds the running max with a
    # unit weight.
    score_heads = head_offset + gl.arange(0, BLOCK_H, layout=head_layout)
    valid_heads = score_heads < num_heads
    max_value = (
        gl.load(attn_sink + score_heads, mask=valid_heads, other=0.0).to(gl.float32)
        * 1.4426950408889634  # log2(e): the sink joins the base-2 softmax.
    )
    denominator = gl.full([BLOCK_H], 1.0, dtype=gl.float32, layout=head_layout)
    accumulator = gl.zeros([BLOCK_H, HEAD_DIM], dtype=gl.float32, layout=wmma)

    # One KV tile in LDS: a second, asynchronously filled tile costs more
    # occupancy than its overlap saves.
    kv_shared = gl.allocate_shared_memory(
        gl.bfloat16, [TILE_K, HEAD_DIM], layout=kv_shared_layout
    )
    kv_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, kv_load_layout))
    for tile_idx in range(num_tiles):
        rows, valid = _tile_rows(
            indices,
            index_row,
            tile_idx,
            effective_len,
            num_kv_rows,
            TILE_K,
            row_layout,
        )
        # The softmax needs the mask in the score layout; reloading the cached
        # indices avoids a cross-lane layout conversion.
        _, score_valid = _tile_rows(
            indices,
            index_row,
            tile_idx,
            effective_len,
            num_kv_rows,
            TILE_K,
            score_row_layout,
        )
        # Invalid rows must load zeros: a zero probability times stale NaN
        # bytes is NaN.
        kv_shared.store(
            gl.load(
                kv + rows.to(tl.int64)[:, None] * 512 + kv_dims[None, :],
                mask=valid[:, None],
                other=0.0,
            )
        )
        k_dot = kv_shared.permute([1, 0]).load(kv_dot_layout)
        scores = gl.amd.cdna5.wmma(
            q_dot, k_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=wmma)
        )
        scores = gl.where(score_valid[None, :], scores * qk_scale_log2, -float("inf"))
        next_max = gl.maximum(max_value, gl.max(scores, axis=1))
        safe_max = gl.where(next_max > -float("inf"), next_max, 0.0)
        alpha = gl.exp2(max_value - safe_max)
        max_value = next_max
        probabilities = gl.exp2(scores - safe_max[:, None])
        denominator = denominator * alpha + gl.sum(probabilities, axis=1)
        accumulator *= alpha[:, None]
        p_dot = gl.convert_layout(probabilities.to(gl.bfloat16), q_dot_layout)
        v_dot = kv_shared.load(kv_dot_layout)
        accumulator = gl.amd.cdna5.wmma(p_dot, v_dot, accumulator)

    safe_denominator = gl.where(denominator > 0.0, denominator, 1.0)
    accumulator = gl.where(
        denominator[:, None] > 0.0, accumulator / safe_denominator[:, None], 0.0
    )

    out_heads = head_offset + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, out_layout)
    )
    out_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, out_layout))
    gl.store(
        out
        + (token_idx.to(tl.int64) * num_heads + out_heads[:, None]) * HEAD_DIM
        + out_dims[None, :],
        gl.convert_layout(accumulator.to(out.dtype.element_ty), out_layout),
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


def launch_gluon_dsv4_prefill_gfx1250(
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
    grid = (q.shape[0], triton.cdiv(q.shape[1], _BLOCK_H))
    gluon_dsv4_prefill_gfx1250[grid](
        q,
        kv_rows,
        indices,
        lens,
        attn_sink.reshape(-1),
        output,
        scale * _LOG2E,
        q.shape[1],
        kv_rows.shape[0],
        SELECTED_WIDTH=indices.shape[1],
        BLOCK_H=_BLOCK_H,
        TILE_K=_TILE_K,
        HEAD_DIM=512,
        NUM_XCDS=_NUM_XCDS,
        num_warps=_NUM_WARPS,
    )
    return output
