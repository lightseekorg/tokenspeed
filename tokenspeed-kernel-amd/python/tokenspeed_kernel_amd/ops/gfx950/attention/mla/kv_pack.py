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

"""Fused MLA K/V packing and FP8 quantization for gfx950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon


@gluon.jit
def _mla_kv_pack_quantize_fp8_kernel(
    k_nope,
    k_pe,
    v,
    k_out,
    v_out,
    k_scale_inv,
    v_scale_inv,
    seq_len,
    k_nope_stride_t,
    k_nope_stride_h,
    k_pe_stride_t,
    v_stride_t,
    v_stride_h,
    k_out_stride_t,
    k_out_stride_h,
    v_out_stride_t,
    v_out_stride_h,
    QK_NOPE: gl.constexpr,
    QK_ROPE: gl.constexpr,
    V_HEAD: gl.constexpr,
    BLOCK_S: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid_s = gl.program_id(0)
    pid_h = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NUM_WARPS, 1], [1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, layout)
    rows = pid_s * BLOCK_S + gl.arange(0, BLOCK_S, layout=row_layout)
    row_mask = rows < seq_len

    nope_cols = gl.arange(0, QK_NOPE, layout=col_layout)
    nope_offsets = (
        rows[:, None] * k_nope_stride_t + pid_h * k_nope_stride_h + nope_cols[None, :]
    )
    nope = gl.load(k_nope + nope_offsets, mask=row_mask[:, None]).to(gl.float32)
    nope *= k_scale_inv
    k_out_base = rows[:, None] * k_out_stride_t + pid_h * k_out_stride_h
    gl.store(
        k_out + k_out_base + nope_cols[None, :],
        nope.to(k_out.dtype.element_ty),
        mask=row_mask[:, None],
    )

    rope_cols = gl.arange(0, QK_ROPE, layout=col_layout)
    rope_offsets = rows[:, None] * k_pe_stride_t + rope_cols[None, :]
    rope = gl.load(k_pe + rope_offsets, mask=row_mask[:, None]).to(gl.float32)
    rope *= k_scale_inv
    gl.store(
        k_out + k_out_base + QK_NOPE + rope_cols[None, :],
        rope.to(k_out.dtype.element_ty),
        mask=row_mask[:, None],
    )

    v_cols = gl.arange(0, V_HEAD, layout=col_layout)
    v_offsets = rows[:, None] * v_stride_t + pid_h * v_stride_h + v_cols[None, :]
    values = gl.load(v + v_offsets, mask=row_mask[:, None]).to(gl.float32)
    values *= v_scale_inv
    v_out_offsets = (
        rows[:, None] * v_out_stride_t + pid_h * v_out_stride_h + v_cols[None, :]
    )
    gl.store(
        v_out + v_out_offsets,
        values.to(v_out.dtype.element_ty),
        mask=row_mask[:, None],
    )


def gluon_mla_kv_pack_quantize_fp8_gfx950(
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    k_scale_inv: float = 1.0,
    v_scale_inv: float = 1.0,
    k_out: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack broadcast MLA keys and quantize K/V to FP8 in one launch.

    Args:
        k_nope: Unrotated keys with shape ``[tokens, heads, qk_nope]``.
        k_pe: Shared rotated keys with shape ``[tokens, qk_rope]`` or
            ``[tokens, 1, qk_rope]``.
        v: Values with shape ``[tokens, heads, v_head]``.
        k_scale_inv: Multiplier applied to both key components before casting.
        v_scale_inv: Multiplier applied to values before casting.
        k_out: Optional pre-allocated packed key output.
        v_out: Optional pre-allocated value output.
        fp8_dtype: FP8 output dtype.
        enable_pdl: Accepted for compatibility with the cross-platform API.

    Returns:
        Packed FP8 keys and FP8 values in the supplied or allocated outputs.
    """
    del enable_pdl

    if k_nope.ndim != 3:
        raise ValueError(f"k_nope must be 3D, got shape {tuple(k_nope.shape)}")
    if v.ndim != 3:
        raise ValueError(f"v must be 3D, got shape {tuple(v.shape)}")
    if k_pe.ndim not in (2, 3):
        raise ValueError(f"k_pe must be 2D or 3D, got shape {tuple(k_pe.shape)}")

    seq_len, num_heads, qk_nope = k_nope.shape
    if v.shape[:2] != (seq_len, num_heads):
        raise ValueError(
            f"v shape {tuple(v.shape)} mismatches k_nope {tuple(k_nope.shape)}"
        )
    if k_pe.shape[0] != seq_len:
        raise ValueError(
            f"k_pe first dim {k_pe.shape[0]} mismatches k_nope first dim {seq_len}"
        )
    if k_pe.ndim == 3:
        if k_pe.shape[1] != 1:
            raise ValueError(f"k_pe head dim must be 1, got {k_pe.shape[1]}")
        k_pe = k_pe.squeeze(1)
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"unsupported FP8 dtype: {fp8_dtype}")
    for tensor, name in ((k_nope, "k_nope"), (k_pe, "k_pe"), (v, "v")):
        if tensor.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(f"{name} must be BF16 or FP16, got {tensor.dtype}")
        if tensor.device != k_nope.device:
            raise ValueError(f"{name} must be colocated with k_nope")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have stride-1 inner dim")

    qk_rope = k_pe.shape[-1]
    v_head = v.shape[-1]
    expected_k_shape = (seq_len, num_heads, qk_nope + qk_rope)
    expected_v_shape = (seq_len, num_heads, v_head)
    if k_out is None:
        k_out = torch.empty(expected_k_shape, dtype=fp8_dtype, device=k_nope.device)
    elif k_out.shape != expected_k_shape or k_out.dtype != fp8_dtype:
        raise ValueError(f"k_out must be {expected_k_shape} with dtype {fp8_dtype}")
    if v_out is None:
        v_out = torch.empty(expected_v_shape, dtype=fp8_dtype, device=v.device)
    elif v_out.shape != expected_v_shape or v_out.dtype != fp8_dtype:
        raise ValueError(f"v_out must be {expected_v_shape} with dtype {fp8_dtype}")
    for tensor, name in ((k_out, "k_out"), (v_out, "v_out")):
        if tensor.device != k_nope.device or tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be colocated with stride-1 inner dim")

    if seq_len == 0:
        return k_out, v_out

    if seq_len < 512:
        block_s, num_warps = 1, 1
    elif seq_len < 2048:
        block_s, num_warps = 4, 2
    else:
        block_s, num_warps = 16, 4
    grid_s = (seq_len + block_s - 1) // block_s
    _mla_kv_pack_quantize_fp8_kernel[(grid_s, num_heads)](
        k_nope,
        k_pe,
        v,
        k_out,
        v_out,
        k_scale_inv,
        v_scale_inv,
        seq_len,
        k_nope.stride(0),
        k_nope.stride(1),
        k_pe.stride(0),
        v.stride(0),
        v.stride(1),
        k_out.stride(0),
        k_out.stride(1),
        v_out.stride(0),
        v_out.stride(1),
        QK_NOPE=qk_nope,
        QK_ROPE=qk_rope,
        V_HEAD=v_head,
        BLOCK_S=block_s,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=0,
    )
    return k_out, v_out


__all__ = ["gluon_mla_kv_pack_quantize_fp8_gfx950"]
