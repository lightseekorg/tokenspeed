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

"""Fused MLA chunked-prefill K/V FP8 quantize.

Replaces the four-kernel sequence in DeepSeek-V3 chunked prefill:
    k = torch.empty(..., dtype=BF16)
    k[..., :qk_nope] = k_nope          # BF16 copy
    k[..., qk_nope:] = k_pe            # BF16 copy with broadcast
    k = k.to(torch.float8_e4m3fn)      # FP8 cast
    v = v.to(torch.float8_e4m3fn)      # FP8 cast

with a single Triton kernel that reads BF16 k_nope/k_pe/v and writes FP8 k
(concatenation of k_nope and broadcast k_pe) and FP8 v in one launch.
"""

from typing import Optional, Tuple

import torch
from tokenspeed_mla._triton import tl, triton


@triton.jit(
    do_not_specialize=["total_rows"],
    do_not_specialize_on_alignment=["total_rows"],
)
def _mla_kv_pack_quantize_fp8_kernel(
    k_nope_ptr,
    k_pe_ptr,
    v_ptr,
    k_out_ptr,
    v_out_ptr,
    k_scale_inv,
    v_scale_inv,
    total_rows,
    k_nope_stride_t,
    k_nope_stride_h,
    k_pe_stride_t,
    v_stride_t,
    v_stride_h,
    k_out_stride_t,
    k_out_stride_h,
    v_out_stride_t,
    v_out_stride_h,
    HEADS: tl.constexpr,
    QK_NOPE: tl.constexpr,
    QK_ROPE: tl.constexpr,
    V_HEAD: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    ROWS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    # Flatten token/head rows; int64 addressing also covers page-strided views.
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    t_idx = (rows // HEADS)[:, None]
    h_idx = (rows % HEADS)[:, None]
    live = rows[:, None] < total_rows
    nope_idx = tl.arange(0, BLOCK_NOPE)[None, :]
    rope_idx = tl.arange(0, BLOCK_ROPE)[None, :]
    v_idx = tl.arange(0, BLOCK_V)[None, :]
    nope_mask = live & (nope_idx < QK_NOPE)
    rope_mask = live & (rope_idx < QK_ROPE)
    v_mask = live & (v_idx < V_HEAD)

    # PDL: wait for the kv_b_proj GEMM (and k_pe producer) to finish writing
    # before we read.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    nope_off = t_idx * k_nope_stride_t + h_idx * k_nope_stride_h + nope_idx
    k_nope = tl.load(k_nope_ptr + nope_off, mask=nope_mask, other=0)

    pe_off = t_idx * k_pe_stride_t + rope_idx
    k_pe = tl.load(k_pe_ptr + pe_off, mask=rope_mask, other=0)

    v_off = t_idx * v_stride_t + h_idx * v_stride_h + v_idx
    v = tl.load(v_ptr + v_off, mask=v_mask, other=0)

    k_nope_fp8 = (k_nope.to(tl.float32) * k_scale_inv).to(FP8_DTYPE)
    k_pe_fp8 = (k_pe.to(tl.float32) * k_scale_inv).to(FP8_DTYPE)
    v_fp8 = (v.to(tl.float32) * v_scale_inv).to(FP8_DTYPE)

    k_out_base = t_idx * k_out_stride_t + h_idx * k_out_stride_h
    tl.store(k_out_ptr + k_out_base + nope_idx, k_nope_fp8, mask=nope_mask)
    tl.store(
        k_out_ptr + k_out_base + QK_NOPE + rope_idx,
        k_pe_fp8,
        mask=rope_mask,
    )

    v_out_off = t_idx * v_out_stride_t + h_idx * v_out_stride_h + v_idx
    tl.store(v_out_ptr + v_out_off, v_fp8, mask=v_mask)

    # PDL: signal that dependent kernels (FMHA) can begin their preamble while
    # our writes finalize.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def mla_kv_pack_quantize_fp8(
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    k_scale_inv: float = 1.0,
    v_scale_inv: float = 1.0,
    k_out: Optional[torch.Tensor] = None,
    v_out: Optional[torch.Tensor] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse cat(k_nope, k_pe-broadcast) + FP8 quantize for K, plus FP8 quantize for V.

    Args:
        k_nope: BF16/FP16 [s, h, qk_nope]. May be a strided view (last-dim stride = 1).
        k_pe:   BF16/FP16 [s, 1, qk_rope] or [s, qk_rope]. Shared across heads.
        v:      BF16/FP16 [s, h, v_head]. May be a strided view (last-dim stride = 1).
        k_scale_inv: scalar multiplier applied before the FP8 cast on K.
        v_scale_inv: scalar multiplier applied before the FP8 cast on V.
        k_out, v_out: optional pre-allocated FP8 output buffers.
        fp8_dtype: FP8 dtype, torch.float8_e4m3fn (default) or torch.float8_e5m2.

    Returns:
        (k_fp8 [s, h, qk_nope + qk_rope], v_fp8 [s, h, v_head]).
    """
    s, num_heads, qk_nope = k_nope.shape
    qk_rope = k_pe.shape[-1]
    v_head = v.shape[-1]

    assert (
        v.shape[0] == s and v.shape[1] == num_heads
    ), f"v shape {tuple(v.shape)} mismatches k_nope {tuple(k_nope.shape)}"
    assert (
        k_pe.shape[0] == s
    ), f"k_pe first dim {k_pe.shape[0]} mismatches k_nope first dim {s}"
    assert k_nope.stride(-1) == 1, "k_nope must have stride-1 inner dim"
    assert v.stride(-1) == 1, "v must have stride-1 inner dim"
    assert k_pe.stride(-1) == 1, "k_pe must have stride-1 inner dim"
    assert fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    if k_out is None:
        k_out = torch.empty(
            (s, num_heads, qk_nope + qk_rope), dtype=fp8_dtype, device=k_nope.device
        )
    if v_out is None:
        v_out = torch.empty((s, num_heads, v_head), dtype=fp8_dtype, device=v.device)

    # Squeeze head dim of k_pe to [s, qk_rope]; preserve per-token stride.
    if k_pe.dim() == 3:
        assert k_pe.shape[1] == 1, f"k_pe head dim must be 1, got {k_pe.shape[1]}"
        k_pe_2d = k_pe.squeeze(1)
    else:
        k_pe_2d = k_pe

    if s == 0:
        return k_out, v_out

    fp8_dtype_const = tl.float8e4nv if fp8_dtype is torch.float8_e4m3fn else tl.float8e5

    _mla_kv_pack_quantize_fp8_kernel[(triton.cdiv(s * num_heads, 8),)](
        k_nope,
        k_pe_2d,
        v,
        k_out,
        v_out,
        k_scale_inv,
        v_scale_inv,
        s * num_heads,
        k_nope.stride(0),
        k_nope.stride(1),
        k_pe_2d.stride(0),
        v.stride(0),
        v.stride(1),
        k_out.stride(0),
        k_out.stride(1),
        v_out.stride(0),
        v_out.stride(1),
        HEADS=num_heads,
        QK_NOPE=qk_nope,
        QK_ROPE=qk_rope,
        V_HEAD=v_head,
        FP8_DTYPE=fp8_dtype_const,
        BLOCK_NOPE=triton.next_power_of_2(qk_nope),
        BLOCK_ROPE=triton.next_power_of_2(qk_rope),
        BLOCK_V=triton.next_power_of_2(v_head),
        ROWS=8,
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        num_stages=3,
        launch_pdl=enable_pdl,
    )
    return k_out, v_out
