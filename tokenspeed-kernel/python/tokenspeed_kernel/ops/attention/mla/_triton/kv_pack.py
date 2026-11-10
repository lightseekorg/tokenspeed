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


"""MLA key packing and FP8 conversion over adjacent token/head rows."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit(
    do_not_specialize=["total_rows"], do_not_specialize_on_alignment=["total_rows"]
)
def _mla_kv_pack_rows(
    k_nope,
    k_rope,
    value,
    key_out,
    value_out,
    scale_k,
    scale_v,
    total_rows,
    kn_stride_t,
    kn_stride_h,
    kr_stride_t,
    v_stride_t,
    v_stride_h,
    ko_stride_t,
    ko_stride_h,
    vo_stride_t,
    vo_stride_h,
    HEADS: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    VALUE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_V: tl.constexpr,
    ROWS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    token = (rows // HEADS).to(tl.int64)[:, None]
    head = (rows % HEADS).to(tl.int64)[:, None]
    n = tl.arange(0, BLOCK_N)
    r = tl.arange(0, BLOCK_R)
    v = tl.arange(0, BLOCK_V)
    mask_n = (rows[:, None] < total_rows) & (n[None, :] < NOPE)
    mask_r = (rows[:, None] < total_rows) & (r[None, :] < ROPE)
    mask_v = (rows[:, None] < total_rows) & (v[None, :] < VALUE)
    kn = tl.load(
        k_nope + token * kn_stride_t + head * kn_stride_h + n[None, :], mask_n, other=0
    ).to(tl.float32)
    kr = tl.load(k_rope + token * kr_stride_t + r[None, :], mask_r, other=0).to(
        tl.float32
    )
    values = tl.load(
        value + token * v_stride_t + head * v_stride_h + v[None, :], mask_v, other=0
    ).to(tl.float32)
    tl.store(
        key_out + token * ko_stride_t + head * ko_stride_h + n[None, :],
        kn * scale_k,
        mask_n,
    )
    tl.store(
        key_out + token * ko_stride_t + head * ko_stride_h + NOPE + r[None, :],
        kr * scale_k,
        mask_r,
    )
    tl.store(
        value_out + token * vo_stride_t + head * vo_stride_h + v[None, :],
        values * scale_v,
        mask_v,
    )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def triton_mla_kv_pack_quantize_fp8(
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
    """Pack shared RoPE keys and convert K/V into caller-owned FP8 buffers.

    Inputs are K-nope [T,H,N], shared K-rope [T,1,R] or [T,R], and V [T,H,V].
    Inner strides must be one. Scalar multipliers apply in FP32 before the cast.
    Optional outputs have shapes [T,H,N+R] and [T,H,V]; omission allocates them.
    Both E4M3 and E5M2 output formats are supported. Returns both output tensors.
    """
    if k_nope.ndim != 3 or v.ndim != 3 or k_pe.ndim not in (2, 3):
        raise ValueError("MLA packing requires 3-D K/V and 2-D or 3-D RoPE keys")
    tokens, heads, nope = k_nope.shape
    rope, values = k_pe.shape[-1], v.shape[-1]
    if v.shape[:2] != (tokens, heads) or k_pe.shape[0] != tokens:
        raise ValueError("MLA K/V token and head dimensions must agree")
    if k_pe.ndim == 3:
        if k_pe.shape[1] != 1:
            raise ValueError("MLA RoPE keys must have one shared head")
        k_pe = k_pe[:, 0]
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("MLA packing requires E4M3 or E5M2 output")
    for tensor in (k_nope, k_pe, v):
        if tensor.stride(-1) != 1 or tensor.device != k_nope.device:
            raise ValueError("MLA inputs require unit inner stride on the same device")
    if k_out is None:
        k_out = torch.empty(
            (tokens, heads, nope + rope), device=k_nope.device, dtype=fp8_dtype
        )
    if v_out is None:
        v_out = torch.empty((tokens, heads, values), device=v.device, dtype=fp8_dtype)
    for output, shape in (
        (k_out, (tokens, heads, nope + rope)),
        (v_out, (tokens, heads, values)),
    ):
        if (
            output.shape != shape
            or output.dtype != fp8_dtype
            or output.device != k_nope.device
            or output.stride(-1) != 1
        ):
            raise ValueError(
                "MLA output shape, dtype, device or inner stride is invalid"
            )
    if tokens == 0:
        return k_out, v_out
    _mla_kv_pack_rows[(triton.cdiv(tokens * heads, 8),)](
        k_nope,
        k_pe,
        v,
        k_out,
        v_out,
        k_scale_inv,
        v_scale_inv,
        tokens * heads,
        *k_nope.stride()[:2],
        k_pe.stride(0),
        *v.stride()[:2],
        *k_out.stride()[:2],
        *v_out.stride()[:2],
        HEADS=heads,
        NOPE=nope,
        ROPE=rope,
        VALUE=values,
        BLOCK_N=triton.next_power_of_2(nope),
        BLOCK_R=triton.next_power_of_2(rope),
        BLOCK_V=triton.next_power_of_2(values),
        ROWS=8,
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        num_stages=3,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return k_out, v_out


@triton.jit(do_not_specialize=["rows"], do_not_specialize_on_alignment=["rows"])
def _mla_rope_broadcast_rows(
    k_rope,
    key_rope,
    rows,
    kr_stride_t,
    ko_stride_t,
    ko_stride_h,
    HEADS: tl.constexpr,
    ROPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    first_head = tl.program_id(1) * BLOCK_H
    r = tl.arange(0, ROPE)
    mask = t[:, None] < rows
    token = t.to(tl.int64)[:, None]
    rope = tl.load(k_rope + token * kr_stride_t + r[None, :], mask).to(tl.float32)
    out = key_rope + token * ko_stride_t + r[None, :]
    for h in tl.static_range(BLOCK_H):
        head = first_head + h
        tl.store(out + head * ko_stride_h, rope, mask & (head < HEADS))


def triton_mla_rope_broadcast_fp8(k_pe: torch.Tensor, key_rope: torch.Tensor) -> None:
    """Write the shared RoPE key into every head of an FP8 key buffer.

    Args:
        k_pe: Floating RoPE keys [T,R] or [T,1,R] with unit inner stride.
        key_rope: E4M3 or E5M2 destination [T,H,R] with unit inner stride, such
            as the RoPE columns key[..., nope:] of a packed key buffer.

    Values convert to FP32 and then to the destination type, as in
    triton_mla_kv_pack_quantize_fp8.
    """
    if k_pe.ndim == 3:
        if k_pe.shape[1] != 1:
            raise ValueError("MLA RoPE keys must have one shared head")
        k_pe = k_pe[:, 0]
    tokens, heads, rope = key_rope.shape
    if (
        k_pe.shape != (tokens, rope)
        or key_rope.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
        or k_pe.stride(-1) != 1
        or key_rope.stride(-1) != 1
        or k_pe.device != key_rope.device
    ):
        raise ValueError("MLA RoPE broadcast shapes, dtypes or strides are invalid")
    if tokens == 0:
        return
    block_t, block_h = 16, heads
    _mla_rope_broadcast_rows[
        (triton.cdiv(tokens, block_t), triton.cdiv(heads, block_h))
    ](
        k_pe,
        key_rope,
        tokens,
        k_pe.stride(0),
        key_rope.stride(0),
        key_rope.stride(1),
        HEADS=heads,
        ROPE=rope,
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        num_warps=8,
    )
